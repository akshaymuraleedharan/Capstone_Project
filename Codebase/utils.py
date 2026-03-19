"""
utils.py - Shared utility functions for the CV Creation pipeline.
Provides JSON cleaning/validation, terminal display formatting,
user interaction helpers, and common operations used across modules.
"""

import json
import re
import os


def format_gpa_label(gpa_value):
    """
    Determine the correct label (GPA, CGPA, Percentage, etc.) from
    the extracted gpa string and return a formatted display string.

    The extraction prompt asks the LLM to preserve the original label,
    so the value may already contain it (e.g. "CGPA: 8.6/10", "72%",
    "GPA: 3.8/4.0"). This function detects the format and returns
    a clean, correctly labelled string.

    Args:
        gpa_value (str): Raw GPA/percentage string from extracted data

    Returns:
        str: Formatted string like "Percentage: 72%" or "CGPA: 8.6/10"
             Returns the value as-is if it already contains a known label.
             Returns empty string if input is empty/None.
    """
    if not gpa_value or not str(gpa_value).strip():
        return ""

    val = str(gpa_value).strip()

    # If the value already starts with a known label, return it cleaned up
    label_pattern = re.match(
        r'^(cgpa|gpa|percentage|percent|score)\s*[:]\s*(.+)',
        val, re.IGNORECASE
    )
    if label_pattern:
        label_raw = label_pattern.group(1).upper()
        score = label_pattern.group(2).strip()
        # Normalize label
        if label_raw in ("PERCENTAGE", "PERCENT"):
            return f"Percentage: {score}"
        elif label_raw == "CGPA":
            return f"CGPA: {score}"
        elif label_raw == "GPA":
            return f"GPA: {score}"
        elif label_raw == "SCORE":
            # "Score: 72%" → detect if it's a percentage
            if "%" in score:
                return f"Percentage: {score}"
            return f"Score: {score}"

    # No label prefix — detect from the value itself
    # Percentage: contains % sign or the word "percent"
    if "%" in val or re.search(r'percent', val, re.IGNORECASE):
        # Clean up: if it's just a number with %, add label
        clean = val.replace("percent", "%").strip()
        return f"Percentage: {clean}"

    # CGPA: contains "cgpa" anywhere or is a number out of 10
    if re.search(r'cgpa', val, re.IGNORECASE):
        # Strip "cgpa" text and format
        score = re.sub(r'cgpa\s*[-:.]?\s*', '', val, flags=re.IGNORECASE).strip()
        return f"CGPA: {score}" if score else f"CGPA: {val}"

    # GPA on a 4.0 scale: number <= 4.0 (with or without /4.0)
    match_4 = re.match(r'^(\d+\.?\d*)\s*/\s*4\.?0?$', val)
    if match_4:
        return f"GPA: {val}"

    # Number out of 10: likely CGPA (common in India)
    match_10 = re.match(r'^(\d+\.?\d*)\s*/\s*10\.?0?$', val)
    if match_10:
        return f"CGPA: {val}"

    match_10_desc = re.match(r'^(\d+\.?\d*)\s+out\s+of\s+10', val, re.IGNORECASE)
    if match_10_desc:
        return f"CGPA: {val}"

    # Plain number: infer from scale
    plain_num = re.match(r'^(\d+\.?\d*)$', val)
    if plain_num:
        num = float(plain_num.group(1))
        if num <= 4.0:
            return f"GPA: {val}"
        elif num <= 10.0:
            return f"CGPA: {val}"
        elif num <= 100.0:
            return f"Percentage: {val}%"

    # Fallback: return with a generic label
    return val


def _repair_truncated_json(fragment):
    """
    Attempt to repair a truncated JSON string by closing open strings,
    arrays, and objects. Used when the LLM runs out of tokens mid-output.

    Heuristic approach:
    1. If we're inside an open string (odd number of unescaped quotes),
       close it with a quote.
    2. Walk through the fragment tracking open braces/brackets.
    3. Append the necessary closing brackets/braces in reverse order.

    Args:
        fragment (str): A JSON fragment starting with '{' that may be truncated

    Returns:
        dict or None: Parsed JSON if repair succeeds, None otherwise
    """
    # Remove any trailing incomplete escape or whitespace
    text = fragment.rstrip()

    # Count unescaped quotes to detect if we're inside a string
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\\' and in_string:
            i += 2  # skip escaped character
            continue
        if ch == '"':
            in_string = not in_string
        i += 1

    # If we ended inside a string, close it
    if in_string:
        text += '"'

    # Remove any trailing comma or colon (invalid before closing bracket)
    text = text.rstrip()
    while text and text[-1] in (',', ':'):
        text = text[:-1].rstrip()

    # Track open braces/brackets (outside of strings) to know what to close
    stack = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\\' and in_string:
            i += 2
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == '{':
                stack.append('}')
            elif ch == '[':
                stack.append(']')
            elif ch in ('}', ']'):
                if stack and stack[-1] == ch:
                    stack.pop()
        i += 1

    # Close all open containers
    text += ''.join(reversed(stack))

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def clean_json_response(response_text):
    """
    Extract valid JSON from an LLM response that may contain markdown
    code blocks, preamble text, or other non-JSON content.

    Tries multiple strategies in order:
    1. Direct json.loads() on the raw response
    2. Extract from ```json ... ``` markdown code blocks
    3. Extract from ``` ... ``` generic code blocks
    4. Find the first { to last } substring

    Args:
        response_text (str): Raw LLM response text

    Returns:
        dict: Parsed JSON dictionary

    Raises:
        ValueError: If no valid JSON can be extracted from the response
    """
    # Strategy 1: Try direct parsing
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from ```json ... ``` code blocks
    json_block_pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(json_block_pattern, response_text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Strategy 3: Extract from ``` ... ``` generic code blocks
    code_block_pattern = r'```\s*(.*?)\s*```'
    matches = re.findall(code_block_pattern, response_text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Strategy 4: Find first { to last } substring
    first_brace = response_text.find('{')
    last_brace = response_text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = response_text[first_brace:last_brace + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Strategy 5: Handle double-brace wrapping (e.g., "{\n{\n  ...}\n}")
    # Some models wrap JSON in an extra pair of braces
    if first_brace != -1:
        second_brace = response_text.find('{', first_brace + 1)
        second_last_brace = response_text.rfind('}', 0, last_brace)
        if (second_brace != -1 and second_last_brace != -1
                and second_last_brace > second_brace):
            json_str = response_text[second_brace:second_last_brace + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

    # Strategy 6: Attempt to repair truncated JSON
    # When the model runs out of tokens, the JSON is cut off mid-value.
    # Try to close open strings, arrays, and objects to recover partial data.
    if first_brace != -1:
        json_fragment = response_text[first_brace:]
        repaired = _repair_truncated_json(json_fragment)
        if repaired is not None:
            return repaired

    raise ValueError(
        "Could not extract valid JSON from LLM response. "
        f"Response preview: {response_text[:200]}..."
    )


def validate_resume_json(data, schema):
    """
    Validate extracted resume data against the expected schema.
    Fills in missing keys with default empty values from the schema.

    Args:
        data (dict): Extracted JSON data from LLM
        schema (dict): Expected schema with default values

    Returns:
        dict: Validated and completed data with all schema keys present
    """
    validated = {}
    for key, default_value in schema.items():
        if key not in data:
            # Key missing entirely — use default
            validated[key] = default_value
        elif isinstance(default_value, dict) and isinstance(data.get(key), dict):
            # Recursively validate nested dicts
            validated[key] = validate_resume_json(data[key], default_value)
        elif isinstance(default_value, list) and not isinstance(data.get(key), list):
            # Expected list but got something else — wrap or default
            if data[key]:
                validated[key] = [data[key]]
            else:
                validated[key] = default_value
        else:
            validated[key] = data[key]
    return validated


def parse_combined_cv_response(response_text):
    """
    Parse a combined CV generation response that contains multiple sections
    separated by ===SECTION_NAME=== markers.

    Expected markers: ===PROFESSIONAL_SUMMARY===, ===EXPERIENCE===, ===PROJECTS===
    Also handles variations like **PROFESSIONAL SUMMARY**, ## PROFESSIONAL SUMMARY,
    or PROFESSIONAL SUMMARY: that smaller models sometimes produce.

    Args:
        response_text (str): Raw LLM response with section markers

    Returns:
        dict: Dictionary with keys 'professional_summary', 'experience', 'projects'.
              Values are the stripped text content for each section.
              Missing sections default to empty string.
    """
    sections = {
        "professional_summary": "",
        "experience": "",
        "projects": ""
    }

    # Map marker names to dict keys
    marker_map = {
        "PROFESSIONAL_SUMMARY": "professional_summary",
        "EXPERIENCE": "experience",
        "PROJECTS": "projects"
    }

    # Strategy 1: Split on ===MARKER=== patterns (strict)
    parts = re.split(r'===([A-Z_]+)===', response_text)

    # parts alternates: [preamble, marker1, content1, marker2, content2, ...]
    i = 1
    while i < len(parts) - 1:
        marker = parts[i].strip()
        content = parts[i + 1].strip()
        if marker in marker_map:
            # Treat "NONE" as empty
            if content.upper() == "NONE":
                content = ""
            sections[marker_map[marker]] = content
        i += 2

    # Strategy 2: If strict parsing found nothing, try flexible marker detection
    # Handles: **PROFESSIONAL SUMMARY**, ## PROFESSIONAL SUMMARY, PROFESSIONAL SUMMARY:
    if not any(sections.values()):
        # Build a flexible pattern that matches various heading styles
        section_patterns = [
            (r'(?:^|\n)\s*(?:={2,3}\s*|#+\s*|\*{2,})\s*PROFESSIONAL[_ ]SUMMARY\s*(?:={2,3}|\*{2,}|:)?\s*\n(.*?)(?=(?:\n\s*(?:={2,3}\s*|#+\s*|\*{2,})\s*(?:EXPERIENCE|PROJECTS))|$)',
             "professional_summary"),
            (r'(?:^|\n)\s*(?:={2,3}\s*|#+\s*|\*{2,})\s*EXPERIENCE\s*(?:={2,3}|\*{2,}|:)?\s*\n(.*?)(?=(?:\n\s*(?:={2,3}\s*|#+\s*|\*{2,})\s*PROJECTS)|$)',
             "experience"),
            (r'(?:^|\n)\s*(?:={2,3}\s*|#+\s*|\*{2,})\s*PROJECTS\s*(?:={2,3}|\*{2,}|:)?\s*\n(.*?)$',
             "projects"),
        ]
        for pattern, key in section_patterns:
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content.upper() != "NONE":
                    sections[key] = content

    # Strategy 3: If still nothing, use the whole response as professional_summary
    # (better than showing nothing)
    if not any(sections.values()) and response_text.strip():
        cleaned = response_text.strip()
        if cleaned.upper() != "NONE":
            sections["professional_summary"] = cleaned

    return sections


def display_banner():
    """
    Print the application title banner to the terminal.
    Shows project name and brief description.
    """
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                   CV CREATION USING LLMs                     ║
║                  Capstone Project - CS01                     ║
║                                                              ║
║          Qwen2.5-1.5B-Instruct (extraction & analysis)       ║
║                             +                                ║
║            Llama-3.2-3B-Instruct (content generation)        ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def display_json_pretty(data, title=""):
    """
    Display a JSON dictionary in formatted, readable terminal output.
    Uses indentation for readability.

    Args:
        data (dict): Data to display
        title (str): Optional header title to print above the data
    """
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _wrap_text(text, width=54):
    """
    Word-wrap text to fit within a given width. Preserves existing
    line breaks and handles bullet points. Strips markdown bold markers.

    Args:
        text (str): Text to wrap
        width (int): Maximum line width (default: 54)

    Returns:
        list: List of wrapped lines
    """
    # Strip markdown bold markers (**text** → text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    result = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            result.append("")
            continue
        # Detect indent for bullet continuation
        indent = ""
        if line.startswith("- ") or line.startswith("• "):
            indent = "  "  # continuation indent for bullets
        words = line.split()
        current = ""
        for word in words:
            if current and len(current) + 1 + len(word) > width:
                result.append(current)
                current = indent + word
            else:
                current = current + " " + word if current else word
        if current:
            result.append(current)
    return result


def display_cv_preview(cv_content, contact_info):
    """
    Display the generated CV content in a formatted terminal preview.
    Shows each section with proper headings and formatting using
    box-drawing characters for clear visual hierarchy.

    Args:
        cv_content (dict): Generated CV sections (keys are section names,
                           values are content strings or lists)
        contact_info (dict): Contact details dict with keys like
                             name, email, phone, location, linkedin
    """
    W = 60  # total box width

    # ── Top border ──
    print(f"\n  ╔{'═' * (W - 2)}╗")
    print(f"  ║{'CV PREVIEW':^{W - 2}}║")
    print(f"  ╠{'═' * (W - 2)}╣")

    # ── Header: Name and contact ──
    name = contact_info.get("name", "N/A")
    print(f"  ║{' ' * (W - 2)}║")
    print(f"  ║  {name.upper():<{W - 4}}║")

    contact_parts = []
    placeholder_values = {"fill_in", "fill in", "n/a", "na", "none", "nil", "-", "."}
    for field in ["email", "phone", "location", "linkedin"]:
        val = contact_info.get(field, "")
        if val and val.strip().lower() not in placeholder_values:
            contact_parts.append(val)
    if contact_parts:
        contact_line = " | ".join(contact_parts)
        # Try to fit all on one line, else split into two balanced lines
        if len(contact_line) <= W - 4:
            print(f"  ║  {contact_line:<{W - 4}}║")
        else:
            # Split into two rows at the best midpoint
            mid = len(contact_parts) // 2
            row1 = " | ".join(contact_parts[:mid])
            row2 = " | ".join(contact_parts[mid:])
            print(f"  ║  {row1:<{W - 4}}║")
            print(f"  ║  {row2:<{W - 4}}║")

    # Section display order
    section_order = [
        "professional_summary",
        "experience",
        "education",
        "skills",
        "projects",
        "certifications",
        "achievements"
    ]

    for section in section_order:
        if section not in cv_content:
            continue
        content = cv_content[section]

        # Skip empty sections
        if not content or content == "" or content == [] or content == {}:
            continue

        # Section divider with heading
        heading = section.replace("_", " ").upper()
        print(f"  ╟{'─' * (W - 2)}╢")
        pad = W - 4 - len(heading)
        print(f"  ║  {heading}{' ' * pad}║")
        print(f"  ╟{'─' * (W - 2)}╢")

        # Format content based on type
        if isinstance(content, str):
            for line in _wrap_text(content, W - 6):
                if not line:
                    print(f"  ║{' ' * (W - 2)}║")
                else:
                    print(f"  ║  {line:<{W - 4}}║")
        elif isinstance(content, list):
            for idx, item in enumerate(content):
                if isinstance(item, dict):
                    _display_dict_item(item, W)
                    # Add blank line between items (but not after last)
                    if idx < len(content) - 1:
                        print(f"  ║{' ' * (W - 2)}║")
                else:
                    item_str = f"• {item}"
                    for line in _wrap_text(item_str, W - 6):
                        print(f"  ║  {line:<{W - 4}}║")
        elif isinstance(content, dict):
            for sub_key, sub_val in content.items():
                label = sub_key.replace("_", " ").title()
                if isinstance(sub_val, list) and sub_val:
                    val_str = ", ".join(str(s) for s in sub_val)
                    full = f"{label}: {val_str}"
                    for line in _wrap_text(full, W - 6):
                        print(f"  ║  {line:<{W - 4}}║")
                elif sub_val:
                    full = f"{label}: {sub_val}"
                    for line in _wrap_text(full, W - 6):
                        print(f"  ║  {line:<{W - 4}}║")

    # ── Bottom border ──
    print(f"  ╚{'═' * (W - 2)}╝\n")


def _display_dict_item(item, width=60):
    """
    Format and display a dictionary item (e.g., experience entry, education entry)
    in the terminal preview, using box-drawing borders.

    Args:
        item (dict): Dictionary with fields like title, company, dates, etc.
        width (int): Total box width for alignment
    """
    W = width

    # Experience format
    if "title" in item and "company" in item:
        title_line = f"{item['title']} | {item['company']}"
        if len(title_line) <= W - 4:
            print(f"  ║  {title_line:<{W - 4}}║")
        else:
            print(f"  ║  {item['title']:<{W - 4}}║")
            print(f"  ║  {item['company']:<{W - 4}}║")

        dates = ""
        if item.get("start_date"):
            dates = item["start_date"]
            if item.get("end_date"):
                dates += f" – {item['end_date']}"
        if dates:
            print(f"  ║  {dates:<{W - 4}}║")

        if item.get("description"):
            for line in _wrap_text(item["description"], W - 8):
                print(f"  ║    {line:<{W - 6}}║")

        if item.get("achievements"):
            for ach in item["achievements"]:
                if ach:
                    bullet = f"• {ach}"
                    for line in _wrap_text(bullet, W - 8):
                        print(f"  ║    {line:<{W - 6}}║")

    # Education format
    elif "degree" in item and "institution" in item:
        degree_line = f"{item['degree']}"
        print(f"  ║  {degree_line:<{W - 4}}║")
        inst = item["institution"]
        year = item.get("year", "")
        inst_line = f"{inst}  {year}".strip() if year else inst
        print(f"  ║  {inst_line:<{W - 4}}║")
        if item.get("gpa"):
            gpa_line = format_gpa_label(item["gpa"])
            print(f"  ║  {gpa_line:<{W - 4}}║")
        if item.get("details"):
            for line in _wrap_text(item["details"], W - 6):
                print(f"  ║  {line:<{W - 4}}║")

    # Certification format
    elif "name" in item:
        issuer = item.get("issuer", "")
        year = item.get("year", "")
        parts = [item["name"]]
        if issuer:
            parts.append(issuer)
        if year:
            parts.append(f"({year})")
        cert_line = f"• {' – '.join(parts)}"
        for line in _wrap_text(cert_line, W - 6):
            print(f"  ║  {line:<{W - 4}}║")

    # Generic dict display
    else:
        for k, v in item.items():
            kv_line = f"{k}: {v}"
            for line in _wrap_text(kv_line, W - 6):
                print(f"  ║  {line:<{W - 4}}║")


def display_ats_report(ats_report):
    """
    Display the ATS compatibility report in a formatted terminal view.
    Shows rubric category breakdown when available, plus keyword lists
    and LLM-generated suggestions.

    Args:
        ats_report (dict): ATS report with keys: ats_score, rubric (dict),
                           matched_keywords, missing_keywords, suggestions
    """
    print("\n" + "=" * 60)
    print("         ATS COMPATIBILITY REPORT")
    print("=" * 60)

    score = ats_report.get("ats_score", "N/A")

    # Score bar visual (e.g. "████████░░ 80/100")
    if isinstance(score, (int, float)):
        filled = round(score / 5)  # 20 chars for 100
        bar = "█" * filled + "░" * (20 - filled)
        print(f"\n  Overall Score: {bar}  {score}/100")
    else:
        print(f"\n  Overall Score: {score}/100")

    # Rubric breakdown (if present)
    rubric = ats_report.get("rubric")
    if rubric:
        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  SCORE BREAKDOWN            Score /25   │")
        print(f"  ├─────────────────────────────────────────┤")
        categories = [
            ("Keyword Match", "keyword_match"),
            ("Completeness", "completeness"),
            ("Impact Quality", "impact_quality"),
            ("Role Alignment", "role_alignment"),
        ]
        for label, key in categories:
            val = rubric.get(key, 0)
            mini_bar = "█" * round(val / 2.5) + "░" * (10 - round(val / 2.5))
            print(f"  │  {label:<20s} {mini_bar} {val:>2}/25   │")
        print(f"  └─────────────────────────────────────────┘")

    matched = ats_report.get("matched_keywords", [])
    if matched:
        print(f"\n  ✓ Matched Keywords ({len(matched)}):")
        print(f"    {', '.join(matched)}")

    missing = ats_report.get("missing_keywords", [])
    if missing:
        print(f"\n  [X] Missing Keywords ({len(missing)}):")
        print(f"    {', '.join(missing)}")

    suggestions = ats_report.get("suggestions", [])
    if suggestions:
        print(f"\n  Suggestions:")
        for i, s in enumerate(suggestions, 1):
            print(f"    {i}. {s}")

    print(f"\n{'=' * 60}")


def get_user_choice(prompt_text, valid_options):
    """
    Display a prompt and validate user input against allowed options.
    Loops until valid input is provided.

    Args:
        prompt_text (str): The prompt message to display
        valid_options (list): List of valid string inputs (e.g., ["1", "2", "3"])

    Returns:
        str: The validated user choice
    """
    while True:
        choice = input(prompt_text).strip()
        if choice in valid_options:
            return choice
        print(f"  Invalid choice. Please enter one of: {', '.join(valid_options)}")


def clear_screen():
    """
    Clear the terminal screen (cross-platform: cls on Windows, clear on Unix).
    """
    os.system("cls" if os.name == "nt" else "clear")


def truncate_text(text, max_length=500):
    """
    Truncate text to a maximum length with ellipsis indicator.
    Used to prevent overly long content in prompts or displays.

    Args:
        text (str): Text to truncate
        max_length (int): Maximum character count (default: 500)

    Returns:
        str: Truncated text with '...' appended if it was truncated
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def strip_llm_commentary(text):
    """
    Remove preamble and postamble commentary that instruction-tuned LLMs
    (like Llama 3.2) often add around generated CV content.

    Handles patterns like:
    - "Here is the rewritten experience entry:"
    - "Rewritten Experience Entries:"
    - "---" separator lines at start/end
    - "Note: I've rewritten...", "Please note...", "Best regards..."
    - "Please let me know if you need any further assistance."
    - "Let me know if you'd like me to make any changes!"
    - Trailing markdown code fences
    - Mid-content commentary lines from the model

    Args:
        text (str): Raw LLM-generated text that may contain commentary

    Returns:
        str: Cleaned text with commentary removed
    """
    if not text or not text.strip():
        return text

    lines = text.splitlines()

    # Patterns that identify a preamble/header line to skip
    preamble_pat = re.compile(
        r"^(here (is|are|'s)\b|below is\b|following is\b|"
        r"(rewritten|revised|updated|enhanced|tailored)\b.*(entries|entry|experience|section|content|below)\b.*:?\s*$|"
        r"i (have|'ve) (rewritten|revised|updated|enhanced|written)\b|"
        r"please find (below|the)\b)",
        re.IGNORECASE
    )

    # Patterns that identify a postamble/disclaimer line to strip from the end
    postamble_pat = re.compile(
        r"^(please note\b|note:\s|note that\b|"
        r"i (have|'ve) (rewritten|revised|updated|enhanced|kept|assumed|removed)\b|"
        r"i did not\b|i only\b|i've\b|i've maintained\b|"
        r"also,\s*i'?ve?\b|"
        r"please let me know\b|let me know\b|"
        r"if you('d| would) like\b|"
        r"feel free to\b|"
        r"hope this helps\b|"
        r"best regards\b|sincerely\b|"
        r"this (rewrite|revision|version)\b|"
        r"\[your (name|ai assistant)\]|"
        r"please confirm\b|confirm your\b|"
        r"i('m|\s+am) waiting\b|i'?ll wait\b|"
        r"i'?ll provide\b|i'?ll not add\b|"
        r"\*\*please confirm\b)",
        re.IGNORECASE
    )

    # Mid-content commentary lines — these can appear between real content
    midcontent_pat = re.compile(
        r"^(let me know if\b|please let me know\b|"
        r"if you('d| would) like (me to|any)\b|"
        r"i hope (this|these)\b|hope this helps\b|"
        r"feel free to (let|reach|contact)\b|"
        r"is there anything else\b|"
        r"don't hesitate to\b|"
        r"here (is|are) the (rewritten|revised|updated)\b|"
        r"please confirm\b|confirm your confirmation\b|"
        r"i('m|\s+am) waiting for your\b|i'?ll wait for your\b|"
        r"i'?ll provide the\b|i'?ll not add anything\b|"
        r"\*\*please confirm\b|"
        r"best regards\b|sincerely\b|"
        r"\[your (name|ai assistant)\]|"
        r"no (impact|outcome|result|detail|information)[/ ]*(outcome|impact|result|detail|information)?\s*mentioned\b|"
        r"^NONE$)",
        re.IGNORECASE
    )

    # --- Step 1: Strip leading preamble/separators ---
    start = 0
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped in ("", "---", "```", "---\n", "```\n"):
            i += 1
            continue
        if preamble_pat.match(stripped):
            start = i + 1
            i += 1
            continue
        # If we hit real content, stop looking for preamble
        break
    start = max(start, i)

    # --- Step 2: Strip trailing postamble/disclaimers ---
    # Find where the postamble block begins (searching from the end)
    end = len(lines)
    for j in range(len(lines) - 1, start - 1, -1):
        stripped = lines[j].strip()
        if stripped in ("", "---", "```"):
            end = j
            continue
        if postamble_pat.match(stripped):
            end = j
            continue
        # Real content line — stop stripping
        break

    cleaned_lines = lines[start:end]

    # --- Step 3: Remove any remaining leading/trailing blank/separator lines ---
    while cleaned_lines and cleaned_lines[0].strip() in ("", "---", "```"):
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1].strip() in ("", "---", "```"):
        cleaned_lines.pop()

    # --- Step 4: Remove markdown horizontal rules (---, ***, ___) from the body ---
    cleaned_lines = [
        ln for ln in cleaned_lines
        if not (set(ln.strip()) <= {'-', '*', '_'} and len(ln.strip()) >= 3)
    ]

    # --- Step 5: Remove mid-content commentary lines ---
    cleaned_lines = [
        ln for ln in cleaned_lines
        if not midcontent_pat.match(ln.strip())
    ]

    # --- Step 6: Remove echoed prompt data (long lines with "| Experience:" or "| Skills:") ---
    cleaned_lines = [
        ln for ln in cleaned_lines
        if not (len(ln.strip()) > 150 and
                re.search(r'\|\s*(Experience|Skills|Education):', ln))
    ]

    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned if cleaned else text.strip()


def deduplicate_content(text):
    """
    Detect and remove duplicate blocks in LLM output. Some models repeat
    the entire content with commentary in between (e.g. output the experience
    twice). This function detects when the second half is largely a repeat
    of the first half and keeps only the first occurrence.

    Args:
        text (str): Text that may contain duplicated blocks

    Returns:
        str: Deduplicated text
    """
    if not text or len(text) < 100:
        return text

    lines = text.splitlines()
    n = len(lines)

    # Only attempt dedup on content with a reasonable number of lines
    if n < 6:
        return text

    # Try splitting at various midpoints and check if the second half
    # is a near-duplicate of the first half
    best_split = None
    best_ratio = 0.0

    for mid in range(n // 3, 2 * n // 3):
        first_half = "\n".join(lines[:mid]).strip()
        second_half = "\n".join(lines[mid:]).strip()

        if not first_half or not second_half:
            continue

        # Quick length check — second half should be roughly similar length
        len_ratio = len(second_half) / len(first_half) if first_half else 0
        if len_ratio < 0.5 or len_ratio > 1.5:
            continue

        # Count how many lines from the second half appear in the first half
        second_lines = [l.strip() for l in second_half.splitlines() if l.strip()]
        if not second_lines:
            continue

        matches = sum(1 for l in second_lines if l in first_half)
        ratio = matches / len(second_lines)

        if ratio > best_ratio:
            best_ratio = ratio
            best_split = mid

    # If more than 60% of the second half's lines are found in the first half,
    # it's likely a duplicate — keep only the first half
    if best_ratio >= 0.6 and best_split is not None:
        return "\n".join(lines[:best_split]).strip()

    return text


def format_cv_content_as_text(cv_content, contact_info):
    """
    Convert CV content dictionary to a plain text string.
    Used for ATS scoring and other text-based analysis.

    Args:
        cv_content (dict): Generated CV sections
        contact_info (dict): Contact information dict

    Returns:
        str: Full CV content as a single text string
    """
    lines = []

    # Name and contact
    name = contact_info.get("name", "")
    if name:
        lines.append(name.upper())
    contact_parts = []
    placeholder_values = {"fill_in", "fill in", "n/a", "na", "none", "nil", "-", "."}
    for field in ["email", "phone", "location", "linkedin"]:
        val = contact_info.get(field, "")
        if val and val.strip().lower() not in placeholder_values:
            contact_parts.append(val)
    if contact_parts:
        lines.append(" | ".join(contact_parts))
    lines.append("")

    # Sections
    section_order = [
        "professional_summary", "experience", "education",
        "skills", "projects", "certifications", "achievements"
    ]

    for section in section_order:
        if section not in cv_content:
            continue
        content = cv_content[section]
        if not content:
            continue

        heading = section.replace("_", " ").upper()
        lines.append(heading)

        if isinstance(content, str):
            lines.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    lines.append(json.dumps(item))
                else:
                    lines.append(f"- {item}")
        elif isinstance(content, dict):
            for k, v in content.items():
                if isinstance(v, list):
                    lines.append(f"{k}: {', '.join(str(s) for s in v)}")
                else:
                    lines.append(f"{k}: {v}")
        lines.append("")

    return "\n".join(lines)
