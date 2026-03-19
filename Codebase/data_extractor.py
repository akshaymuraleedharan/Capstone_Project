"""
data_extractor.py - Uses Qwen 2.5 (extraction LLM) to parse raw resume text
into structured JSON with standardized fields.

This module implements Step 2 of the pipeline: Resume Data Extraction.
It takes unstructured resume text from any input source and produces
a clean, validated JSON dictionary that can be used for CV generation.
"""

import json
from llm_handler import LLMHandler
from prompts import (RESUME_EXTRACTION_PROMPT, RESUME_HEADER_PROMPT,
                     RESUME_EXPERIENCE_PROMPT, FOLLOW_UP_PROMPT_JD,
                     FOLLOW_UP_PROMPT_GENERAL)
from utils import clean_json_response, validate_resume_json, display_json_pretty


class ResumeDataExtractor:
    """
    Extracts structured resume data from raw text using the Qwen 2.5 model.
    Outputs a standardized JSON dictionary with all CV sections.
    """

    # Standard schema for extracted resume data — used for validation
    RESUME_SCHEMA = {
        "name": "",
        "contact": {
            "email": "",
            "phone": "",
            "location": "",
            "linkedin": "",
            "github": "",
            "portfolio": ""
        },
        "professional_summary": "",
        "years_experience": "",
        "education": [],
        "experience": [],
        "skills": {
            "technical": [],
            "soft": [],
            "tools": [],
            "languages": []
        },
        "certifications": [],
        "projects": [],
        "achievements": [],
        "publications": []
    }

    def __init__(self, llm_handler):
        """
        Initialize the extractor with an LLM handler instance.

        Args:
            llm_handler (LLMHandler): Configured LLM handler with extraction model
        """
        self.llm = llm_handler

    def _extract_pass(self, prompt, pass_label=""):
        """
        Execute a single extraction pass: call the LLM and parse the JSON
        response.  Retries once on JSON parse failure.  Returns None if both
        attempts fail, allowing the caller to fall back gracefully instead
        of aborting the entire extraction.

        Args:
            prompt (str): The fully-formatted prompt to send to the extraction model
            pass_label (str): Human-readable label for progress/warning messages
                              (e.g. "pass 1 (header)")

        Returns:
            dict or None: Parsed JSON data from the LLM, or None on failure
        """
        try:
            response = self.llm.extract(prompt)
            return clean_json_response(response)
        except (ValueError, RuntimeError) as e:
            # First attempt failed — retry once because the model may
            # produce valid output on a second try (non-deterministic sampling)
            print(f"  Warning: {pass_label} first attempt failed: {e}")
            print(f"  Retrying {pass_label}...")
            try:
                response = self.llm.extract(prompt)
                return clean_json_response(response)
            except (ValueError, RuntimeError) as e2:
                print(f"  Warning: {pass_label} retry also failed: {e2}")
                return None

    def extract_from_text(self, raw_text):
        """
        Send raw resume text to Qwen 2.5 for structured extraction.

        Uses a two-pass approach to avoid token truncation with the small
        Qwen 2.5 1.5B model:
          Pass 1 — Extract header data (name, contact, education, skills,
                   certifications, projects, achievements, publications)
          Pass 2 — Extract experience data (job roles and achievement bullets)

        Each pass receives the full resume text for context but returns a
        smaller JSON — fitting comfortably within the 2048 max_new_tokens
        limit.  Results are deep-merged, then all existing post-processing
        (validation, sanitization, flattening, recovery, inference) runs
        once on the merged output.

        If both passes fail, falls back to the original single-pass prompt
        as a last resort.

        Args:
            raw_text (str): Unstructured resume text from any input source

        Returns:
            dict: Structured resume data matching RESUME_SCHEMA

        Raises:
            ValueError: If all extraction attempts fail to produce valid JSON
            RuntimeError: If all LLM calls fail after retries
        """
        print("\n  Extracting resume data using Qwen 2.5 (multi-pass)...")

        # ── Pass 1: Header data ──────────────────────────────────────
        # Extracts everything EXCEPT work experience.  The prompt explicitly
        # tells the model to skip experience so it doesn't waste output
        # tokens on the largest section.
        print("  Extracting personal info, education, and skills (pass 1/2)...")
        header_data = self._extract_pass(
            RESUME_HEADER_PROMPT.format(resume_text=raw_text),
            pass_label="pass 1 (header)"
        )

        # ── Pass 2: Experience data ──────────────────────────────────
        # Extracts ONLY work experience entries with their achievement
        # bullets.  The minimal JSON template lets the model focus its
        # output budget on the actual content.
        print("  Extracting work experience (pass 2/2)...")
        experience_data = self._extract_pass(
            RESUME_EXPERIENCE_PROMPT.format(resume_text=raw_text),
            pass_label="pass 2 (experience)"
        )

        # ── Merge results from both passes ───────────────────────────
        # _deep_merge_resume combines the two partial dicts into one
        # unified structure matching RESUME_SCHEMA.
        if header_data and experience_data:
            merged = self._deep_merge_resume(header_data, experience_data)
        elif header_data:
            # Pass 2 failed — proceed without experience; the user can
            # add experience manually via the edit flow
            print("  Warning: Experience extraction failed. Using header data only.")
            merged = header_data
        elif experience_data:
            # Pass 1 failed — proceed without header fields; validation
            # will fill empty defaults
            print("  Warning: Header extraction failed. Using experience data only.")
            merged = experience_data
        else:
            # Both passes failed — fall back to the original single-pass
            # extraction prompt as a last resort (same behavior as before
            # the multi-pass change)
            print("  Warning: Both passes failed. Attempting single-pass fallback...")
            prompt = RESUME_EXTRACTION_PROMPT.format(resume_text=raw_text)
            response = self.llm.extract(prompt)
            merged = clean_json_response(response)

        # ── Post-processing (runs once on merged result) ─────────────

        # Validate against schema and fill any missing keys with defaults
        validated_data = validate_resume_json(merged, self.RESUME_SCHEMA)

        # Remove placeholder values (FILL_IN, N/A, etc.) left by the model
        validated_data = self._sanitize_extracted_data(validated_data)

        # Flatten achievements: the model sometimes returns dicts with
        # {"title": "...", "company": "...", "date": "..."} instead of strings.
        # Normalize them to plain strings so downstream code is simpler.
        achievements = validated_data.get("achievements", [])
        if achievements:
            flat = []
            for ach in achievements:
                if isinstance(ach, dict):
                    title = ach.get("title", "")
                    if title:
                        flat.append(title)
                elif isinstance(ach, str) and ach:
                    flat.append(ach)
            validated_data["achievements"] = flat

        # Flatten experience achievements too — the model sometimes returns
        # dicts like {"title": "...", "details": "...", "impact": "..."}
        # instead of plain strings inside experience[].achievements[].
        for exp in validated_data.get("experience", []):
            if not isinstance(exp, dict):
                continue
            raw_achs = exp.get("achievements", [])
            if not raw_achs:
                continue
            flat_achs = []
            for ach in raw_achs:
                if isinstance(ach, dict):
                    # Combine title + impact into a clean bullet string
                    parts = []
                    title = ach.get("title", "")
                    details = ach.get("details", "")
                    impact = ach.get("impact", "")
                    if title:
                        parts.append(title)
                    if impact and impact.lower() not in ("", "n/a", "none"):
                        parts.append(f"({impact})")
                    if parts:
                        flat_achs.append(" ".join(parts))
                    elif details:
                        flat_achs.append(details)
                elif isinstance(ach, str) and ach:
                    flat_achs.append(ach)
            exp["achievements"] = flat_achs

        # Recover missing experience achievements from raw resume text.
        # Small models sometimes extract the role title/company/dates but
        # drop the bullet points.  This fallback scans the original text
        # and fills in any experience entry whose achievements list is empty.
        validated_data = self._recover_missing_achievements(validated_data, raw_text)

        # Infer soft skills from experience descriptions if none were extracted
        validated_data = self._infer_soft_skills(validated_data)

        # Infer implied technical skills from experience/project descriptions
        validated_data = self._infer_technical_skills(validated_data)

        # Recover missing contact fields (location, LinkedIn, GitHub) from
        # raw text using regex. Small models often fill name/email/phone but
        # skip URLs and location.
        validated_data = self._recover_missing_contact_fields(validated_data, raw_text)

        # Deduplicate and reclassify certifications vs achievements.
        # Small models often dump the same items into both lists, or
        # misclassify achievements as certifications and vice versa.
        validated_data = self._dedup_certs_and_achievements(validated_data)

        print("  Resume data extracted successfully.")
        return validated_data

    def display_extracted_data(self, resume_data):
        """
        Pretty-print extracted resume data to terminal for user verification.
        Shows each section in a readable format so the user can confirm accuracy.

        Args:
            resume_data (dict): Structured resume JSON matching RESUME_SCHEMA
        """
        print("\n" + "=" * 60)
        print("         EXTRACTED RESUME DATA")
        print("=" * 60)

        # ── Personal Info ────────────────────────────────────────
        print(f"\n  --- Personal Info ---")
        print(f"  Name: {resume_data.get('name', 'N/A')}")
        contact = resume_data.get("contact", {})
        for field in ["email", "phone", "location", "linkedin", "github", "portfolio"]:
            val = contact.get(field, "")
            if val:
                print(f"  {field.title()}: {val}")

        # ── Professional Summary ─────────────────────────────────
        summary = resume_data.get("professional_summary", "")
        if summary:
            print(f"\n  --- Professional Summary ---")
            print(f"  {summary}")

        # ── Education ────────────────────────────────────────────
        education = resume_data.get("education", [])
        if education:
            print(f"\n  --- Education ({len(education)} entries) ---")
            for edu in education:
                if isinstance(edu, dict):
                    degree = edu.get("degree", "N/A")
                    institution = edu.get("institution", "N/A")
                    year = edu.get("year", "")
                    gpa = edu.get("gpa", "")
                    line = f"    • {degree} | {institution} ({year})"
                    if gpa:
                        from utils import format_gpa_label
                        gpa_display = format_gpa_label(gpa)
                        if gpa_display:
                            line += f" — {gpa_display}"
                    print(line)
                else:
                    print(f"    • {edu}")

        # ── Work Experience ───────────────────────────────────────
        experience = resume_data.get("experience", [])
        if experience:
            print(f"\n  --- Work Experience ({len(experience)} entries) ---")
            for exp in experience:
                if isinstance(exp, dict):
                    title = exp.get("title", "N/A")
                    company = exp.get("company", "N/A")
                    dates = f"{exp.get('start_date', '')} - {exp.get('end_date', '')}"
                    print(f"    • {title} at {company} ({dates})")
                    achievements = exp.get("achievements", [])
                    for ach in achievements:
                        if isinstance(ach, dict):
                            # Fallback display for un-flattened dicts
                            title = ach.get("title", "")
                            impact = ach.get("impact", "")
                            line = title
                            if impact and impact.lower() not in ("", "n/a", "none"):
                                line += f" ({impact})"
                            if line:
                                print(f"      ◦ {line}")
                        elif ach:
                            print(f"      ◦ {ach}")
                else:
                    print(f"    • {exp}")

        # ── Skills ────────────────────────────────────────────────
        skills = resume_data.get("skills", {})
        if skills:
            print(f"\n  --- Skills ---")
            for category, skill_list in skills.items():
                if skill_list:
                    print(f"    {category.title()}: {', '.join(str(s) for s in skill_list)}")

        # ── Projects ──────────────────────────────────────────────
        projects = resume_data.get("projects", [])
        if projects:
            print(f"\n  --- Projects ({len(projects)} entries) ---")
            for proj in projects:
                if isinstance(proj, dict):
                    print(f"    • {proj.get('name', 'N/A')}: {proj.get('description', '')}")
                else:
                    print(f"    • {proj}")

        # ── Certifications ────────────────────────────────────────
        certs = resume_data.get("certifications", [])
        if certs:
            print(f"\n  --- Certifications ---")
            for cert in certs:
                if isinstance(cert, dict):
                    print(f"    • {cert.get('name', 'N/A')} ({cert.get('issuer', '')} {cert.get('year', '')})")
                else:
                    print(f"    • {cert}")

        # ── Achievements ──────────────────────────────────────────
        achievements = resume_data.get("achievements", [])
        if achievements:
            print(f"\n  --- Achievements ---")
            for ach in achievements:
                if isinstance(ach, dict):
                    title = ach.get("title", "")
                    company = ach.get("company", "")
                    if title:
                        label = f"{title} ({company})" if company else title
                        print(f"    • {label}")
                elif ach:
                    print(f"    • {ach}")

        print(f"\n{'=' * 60}")

    def get_contact_info(self, resume_data):
        """
        Extract a flat contact info dictionary from the resume data.
        Combines the name and contact fields for use in output generation.
        Filters out placeholder values (FILL_IN, N/A, etc.) so they never
        appear in the rendered CV output.

        Args:
            resume_data (dict): Structured resume JSON

        Returns:
            dict: Flat dict with keys: name, email, phone, location, linkedin, portfolio
        """
        contact = resume_data.get("contact", {})
        raw = {
            "name": resume_data.get("name", ""),
            "email": contact.get("email", ""),
            "phone": contact.get("phone", ""),
            "location": contact.get("location", ""),
            "linkedin": contact.get("linkedin", ""),
            "github": contact.get("github", ""),
            "portfolio": contact.get("portfolio", "")
        }
        return {k: self._clean_placeholder(v) for k, v in raw.items()}

    @staticmethod
    def _clean_placeholder(value):
        """
        Return empty string if the value is a known placeholder left by
        the extraction model (e.g. FILL_IN, N/A, none, -, etc.).

        Args:
            value: The raw extracted value

        Returns:
            str: The original value if real, empty string if placeholder
        """
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        placeholders = {
            "fill_in", "fill in", "n/a", "na", "none", "nil",
            "not applicable", "not provided", "-", ".", ""
        }
        if stripped.lower() in placeholders:
            return ""
        return stripped

    @classmethod
    def _sanitize_extracted_data(cls, data):
        """
        Recursively walk the extracted resume dictionary and replace all
        placeholder values (FILL_IN, N/A, etc.) with proper empty defaults.
        This ensures no raw template values reach the CV output.

        Args:
            data: A dict, list, or scalar value from the extracted JSON

        Returns:
            The sanitized version with placeholders replaced
        """
        if isinstance(data, dict):
            return {k: cls._sanitize_extracted_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            cleaned = [cls._sanitize_extracted_data(item) for item in data]
            # Remove list items that are empty strings or dicts with all
            # values empty (e.g., {"name":"","description":"","technologies":[]})
            result = []
            for item in cleaned:
                if isinstance(item, str) and item == "":
                    continue
                if isinstance(item, dict) and not any(
                    v for v in item.values()
                    if v and v != [] and v != {}
                ):
                    continue
                result.append(item)
            return result
        elif isinstance(data, str):
            return cls._clean_placeholder(data)
        return data

    @staticmethod
    def _deep_merge_resume(base, overlay):
        """
        Deep-merge two partial resume extraction results into a single dict.

        Used to combine Pass 1 (header data) and Pass 2 (experience data)
        from the multi-pass extraction.  Each pass provides mostly disjoint
        sections, but the merge handles overlaps gracefully just in case.

        Merge rules by value type:
          - str:  prefer the non-empty value; if both non-empty, keep base
                  (Pass 1 header data is richer for non-experience fields)
          - list: use whichever is non-empty; concatenate if both have data
          - dict: recurse (handles nested dicts like 'contact' and 'skills')
          - other: prefer non-falsy, then fall back to base

        Args:
            base (dict): First extraction result (typically Pass 1 header data)
            overlay (dict): Second extraction result (typically Pass 2 experience data)

        Returns:
            dict: Merged resume data combining both passes
        """
        # Handle the case where one or both passes returned None / empty
        if not base:
            return overlay or {}
        if not overlay:
            return base or {}

        merged = {}
        all_keys = set(base.keys()) | set(overlay.keys())

        for key in all_keys:
            base_val = base.get(key)
            overlay_val = overlay.get(key)

            # Key exists in only one dict — take whatever we have
            if base_val is None:
                merged[key] = overlay_val
                continue
            if overlay_val is None:
                merged[key] = base_val
                continue

            # Both dicts have this key — merge by type
            if isinstance(base_val, dict) and isinstance(overlay_val, dict):
                # Recurse for nested dicts (e.g. "contact", "skills")
                merged[key] = ResumeDataExtractor._deep_merge_resume(
                    base_val, overlay_val
                )
            elif isinstance(base_val, list) and isinstance(overlay_val, list):
                # For lists (e.g. "experience", "education", "certifications"):
                # use the non-empty one, or concatenate if both have entries
                if base_val and overlay_val:
                    merged[key] = base_val + overlay_val
                else:
                    merged[key] = base_val or overlay_val
            elif isinstance(base_val, str) and isinstance(overlay_val, str):
                # For strings: prefer base (Pass 1) since it has richer header
                # data; fall back to overlay if base is empty
                merged[key] = base_val if base_val.strip() else overlay_val
            else:
                # Mixed types or other — prefer non-falsy, default to base
                merged[key] = base_val if base_val else overlay_val

        return merged

    @staticmethod
    def _dedup_certs_and_achievements(resume_data):
        """
        Remove duplicate items that appear in both certifications and
        achievements. Small extraction models often dump the same items
        into both lists.

        Items are kept in whichever section the model placed them first
        (certifications is checked first). We do NOT reclassify — the
        resume's original section placement is respected.

        Args:
            resume_data (dict): Structured resume data (modified in place)

        Returns:
            dict: Resume data with duplicates removed
        """
        import re

        certs = resume_data.get("certifications", [])
        achievements = resume_data.get("achievements", [])

        def _get_text(item):
            if isinstance(item, dict):
                return item.get("name", "") or item.get("title", "")
            return str(item)

        def _normalize(text):
            # Strip punctuation/spaces so "AWS Cert." matches "AWS Cert" for dedup
            return re.sub(r'[^a-z0-9]', '', text.lower())

        # Keep all certifications, track their normalized text
        seen = set()
        final_certs = []
        for item in certs:
            text = _get_text(item)
            if not text.strip():
                continue
            norm = _normalize(text)
            if norm in seen:
                continue
            seen.add(norm)
            final_certs.append(item)

        # Keep achievements only if not already in certifications
        final_achievements = []
        for item in achievements:
            text = _get_text(item)
            if not text.strip():
                continue
            norm = _normalize(text)
            if norm in seen:
                continue
            seen.add(norm)
            final_achievements.append(item)

        resume_data["certifications"] = final_certs
        resume_data["achievements"] = final_achievements
        return resume_data

    @staticmethod
    def _recover_missing_contact_fields(resume_data, raw_text):
        """
        Recover contact fields (location, LinkedIn, GitHub) that the
        extraction model missed. Small models reliably extract name,
        email, and phone but often skip URLs and location.

        Uses simple regex/keyword matching on the raw resume text —
        no LLM call needed, so it's fast and deterministic.

        Args:
            resume_data (dict): Structured resume data (modified in place)
            raw_text (str): Original raw resume text

        Returns:
            dict: Resume data with recovered contact fields
        """
        import re

        contact = resume_data.get("contact", {})
        if not isinstance(contact, dict):
            contact = {}
            resume_data["contact"] = contact

        # --- LinkedIn ---
        if not contact.get("linkedin"):
            # Match linkedin.com URLs (with or without https://)
            match = re.search(
                r'(https?://)?(?:www\.)?linkedin\.com/in/[\w-]+',
                raw_text, re.IGNORECASE
            )
            if match:
                contact["linkedin"] = match.group(0)

        # --- GitHub ---
        if not contact.get("github"):
            # Match github.com URLs (with or without https://)
            match = re.search(
                r'(https?://)?(?:www\.)?github\.com/[\w-]+',
                raw_text, re.IGNORECASE
            )
            if match:
                contact["github"] = match.group(0)

        # --- Location ---
        if not contact.get("location"):
            # Common patterns: "Based in: City", "Location: City",
            # "City, State, Country" after a label
            for pattern in [
                r'(?:Based in|Location|Address)\s*[:\-]\s*(.+)',
            ]:
                match = re.search(pattern, raw_text, re.IGNORECASE)
                if match:
                    location = match.group(1).strip().rstrip('|').strip()
                    # Stop at newline or pipe
                    location = location.split('\n')[0].strip()
                    if location and len(location) < 100:  # reject regex over-matches
                        contact["location"] = location
                        break

        resume_data["contact"] = contact
        return resume_data

    @staticmethod
    def _infer_soft_skills(resume_data):
        """
        Infer soft skills from experience descriptions, achievements, and
        professional summary when the extraction model returns an empty
        soft skills list. Uses keyword matching against common soft skill
        indicators — no LLM call required.

        Only adds soft skills if the 'soft' list is currently empty, to
        avoid overwriting skills the model did extract.

        Args:
            resume_data (dict): Structured resume data

        Returns:
            dict: Resume data with soft skills populated if they were empty
        """
        import re

        skills = resume_data.get("skills", {})
        if not isinstance(skills, dict):
            return resume_data

        # Only infer if soft skills list is empty
        existing_soft = skills.get("soft", [])
        if existing_soft:
            return resume_data

        # Build a corpus from experience descriptions, achievements, and summary
        text_parts = []

        # Professional summary
        summary = resume_data.get("professional_summary", "")
        if summary:
            text_parts.append(summary)

        # Experience descriptions and achievements
        for exp in resume_data.get("experience", []):
            if isinstance(exp, dict):
                desc = exp.get("description", "")
                if desc:
                    text_parts.append(desc)
                for ach in exp.get("achievements", []):
                    if isinstance(ach, str) and ach:
                        text_parts.append(ach)

        # Project descriptions
        for proj in resume_data.get("projects", []):
            if isinstance(proj, dict):
                desc = proj.get("description", "")
                if desc:
                    text_parts.append(desc)

        corpus = " ".join(text_parts).lower()

        if not corpus:
            return resume_data

        # Keyword-to-skill mapping: pattern → soft skill name
        # Patterns are checked against the combined corpus text
        skill_patterns = [
            (r'\b(managed|led|lead|headed|oversaw|supervised)\b.*\b(team|staff|group|department|people)\b', "Team Leadership"),
            (r'\b(team|staff|group)\b.*\b(managed|led|lead|headed|oversaw|supervised)\b', "Team Leadership"),
            (r'\b(coordinated|collaborated|partnered|worked with)\b.*\b(team|cross[- ]?functional|department|stakeholder|agencies)\b', "Cross-functional Collaboration"),
            (r'\b(communicated|presented|reporting|reports for|briefed)\b.*\b(leadership|management|stakeholder|client|executive)\b', "Communication"),
            (r'\b(built|created|prepared)\b.*\breport', "Analytical Reporting"),
            (r'\bproblem[- ]?solv', "Problem Solving"),
            (r'\b(mentored|coached|trained|onboarded)\b', "Mentoring"),
            (r'\b(prioriti[sz]|multi[- ]?task|juggl|balanc)\b', "Time Management"),
            (r'\b(deadline|on[- ]?time|timely delivery)\b', "Time Management"),
            (r'\b(negotiat|persuad|influenc)\b', "Negotiation"),
            (r'\b(creative|innovat|ideation|brainstorm)\b', "Creativity"),
            (r'\b(adapt|pivot|flex|agile|fast[- ]?paced|dynamic)\b', "Adaptability"),
            (r'\b(analyz|analysis|insight|data[- ]?driven|interpret)\b', "Analytical Thinking"),
            (r'\b(customer|client)[- ]?(service|facing|relationship|success)\b', "Client Relationship Management"),
            (r'\b(attention to detail|thoroughness|meticulous|careful|accuracy)\b', "Attention to Detail"),
            (r'\b(transition|upskill|self[- ]?learn|self[- ]?taught|continuous learning)\b', "Continuous Learning"),
            (r'\b(a/?b test|experiment|optimiz|improv)\b.*\b(campaign|performance|conversion|rate)\b', "Data-Driven Decision Making"),
            (r'\b(budget|cost|resource)\b.*\b(manag|allocat|optimiz)\b', "Budget Management"),
            (r'\b(strategy|strategic|planning)\b', "Strategic Thinking"),
            (r'\b(project manage|manage.*project|agile|scrum|sprint)\b', "Project Management"),
        ]

        inferred = []
        seen = set()

        for pattern, skill_name in skill_patterns:
            if skill_name in seen:
                continue
            if re.search(pattern, corpus):
                inferred.append(skill_name)
                seen.add(skill_name)

        if inferred:
            skills["soft"] = inferred
            resume_data["skills"] = skills

        return resume_data

    @staticmethod
    def _infer_technical_skills(resume_data):
        """
        Infer implied technical skills and tools from experience descriptions,
        project descriptions, and achievements. Adds them to the 'technical'
        and 'tools' lists only if not already present.

        Unlike soft skills inference, this AUGMENTS existing lists rather than
        only running when the list is empty — because extraction models often
        capture some skills but miss ones mentioned only in descriptions.

        Uses keyword matching — no LLM call required, instant execution.

        Args:
            resume_data (dict): Structured resume data

        Returns:
            dict: Resume data with additional technical skills/tools added
        """
        import re

        skills = resume_data.get("skills", {})
        if not isinstance(skills, dict):
            return resume_data

        existing_technical = set(s.lower().strip() for s in skills.get("technical", []))
        existing_tools = set(s.lower().strip() for s in skills.get("tools", []))
        existing_all = existing_technical | existing_tools

        # Build corpus from experience, projects, achievements, certifications
        text_parts = []

        for exp in resume_data.get("experience", []):
            if isinstance(exp, dict):
                for field in ("description", "title"):
                    val = exp.get(field, "")
                    if val:
                        text_parts.append(val)
                for ach in exp.get("achievements", []):
                    if isinstance(ach, str) and ach:
                        text_parts.append(ach)

        for proj in resume_data.get("projects", []):
            if isinstance(proj, dict):
                for field in ("description", "name"):
                    val = proj.get(field, "")
                    if val:
                        text_parts.append(val)
                techs = proj.get("technologies", [])
                if isinstance(techs, list):
                    text_parts.extend(str(t) for t in techs)

        for ach in resume_data.get("achievements", []):
            if isinstance(ach, str) and ach:
                text_parts.append(ach)

        for cert in resume_data.get("certifications", []):
            if isinstance(cert, dict):
                name = cert.get("name", "")
                if name:
                    text_parts.append(name)
            elif isinstance(cert, str):
                text_parts.append(cert)

        corpus = " ".join(text_parts).lower()
        if not corpus:
            return resume_data

        # Pattern → (skill_name, category: "technical" or "tools")
        # These are industry-standard skills that people USE but don't always LIST
        skill_patterns = [
            # --- Programming & Data ---
            (r'\bpython\b', "Python", "technical"),
            (r'\bjava(?:script)?\b', "JavaScript", "technical"),
            (r'\btypescript\b', "TypeScript", "technical"),
            (r'\bjava\b(?!script)', "Java", "technical"),
            (r'\bc\+\+\b', "C++", "technical"),
            (r'\bc#\b', "C#", "technical"),
            (r'\bruby\b', "Ruby", "technical"),
            (r'\bphp\b', "PHP", "technical"),
            (r'\bswift\b', "Swift", "technical"),
            (r'\bkotlin\b', "Kotlin", "technical"),
            (r'\br\b(?:\s+programming|\s+studio|\s+language)', "R", "technical"),  # bare "r" too common; require context
            (r'\bsql\b', "SQL", "technical"),
            (r'\bhtml\b', "HTML", "technical"),
            (r'\bcss\b', "CSS", "technical"),

            # --- Frameworks ---
            (r'\breact(?:\.?js)?\b', "React", "technical"),
            (r'\bangular\b', "Angular", "technical"),
            (r'\bvue(?:\.?js)?\b', "Vue.js", "technical"),
            (r'\bnode(?:\.?js)?\b', "Node.js", "technical"),
            (r'\bdjango\b', "Django", "technical"),
            (r'\bflask\b', "Flask", "technical"),
            (r'\bspring\b', "Spring", "technical"),
            (r'\b\.net\b', ".NET", "technical"),

            # --- Data & ML ---
            (r'\bmachine learning\b', "Machine Learning", "technical"),
            (r'\bdeep learning\b', "Deep Learning", "technical"),
            (r'\bdata analy', "Data Analysis", "technical"),
            (r'\bdata visual', "Data Visualization", "technical"),
            (r'\bstatistic', "Statistical Analysis", "technical"),
            (r'\bpandas\b', "Pandas", "technical"),
            (r'\bnumpy\b', "NumPy", "technical"),
            (r'\btensorflow\b', "TensorFlow", "technical"),
            (r'\bpytorch\b', "PyTorch", "technical"),
            (r'\bscikit[- ]?learn\b', "Scikit-learn", "technical"),
            (r'\bpower\s?bi\b', "Power BI", "tools"),
            (r'\btableau\b', "Tableau", "tools"),

            # --- Cloud & DevOps ---
            (r'\baws\b', "AWS", "tools"),
            (r'\bazure\b', "Azure", "tools"),
            (r'\bgcp\b|google cloud', "Google Cloud", "tools"),
            (r'\bdocker\b', "Docker", "tools"),
            (r'\bkubernetes\b', "Kubernetes", "tools"),
            (r'\bci/?cd\b', "CI/CD", "technical"),
            (r'\bterraform\b', "Terraform", "tools"),
            (r'\bjenkins\b', "Jenkins", "tools"),

            # --- Databases ---
            (r'\bmysql\b', "MySQL", "tools"),
            (r'\bpostgres', "PostgreSQL", "tools"),
            (r'\bmongodb\b', "MongoDB", "tools"),
            (r'\bredis\b', "Redis", "tools"),

            # --- Tools ---
            (r'\bgit(?:hub|lab)?\b', "Git", "tools"),
            (r'\bjira\b', "Jira", "tools"),
            (r'\bconfluence\b', "Confluence", "tools"),
            (r'\bslack\b', "Slack", "tools"),
            (r'\bfigma\b', "Figma", "tools"),
            (r'\bphotoshop\b', "Adobe Photoshop", "tools"),
            (r'\billustrator\b', "Adobe Illustrator", "tools"),

            # --- Marketing & Business ---
            (r'\bgoogle\s*a(ds|dwords)\b', "Google Ads", "tools"),
            (r'\bgoogle\s*analytics\b', "Google Analytics", "tools"),
            (r'\bfacebook\s*(ads|business)\b', "Facebook Ads", "tools"),
            (r'\bseo\b', "SEO", "technical"),
            (r'\bsem\b', "SEM", "technical"),
            (r'\bcontent\s*market', "Content Marketing", "technical"),
            (r'\bemail\s*market', "Email Marketing", "technical"),
            (r'\bsocial\s*media\s*(market|manage|strateg)', "Social Media Marketing", "technical"),
            (r'\bcrm\b', "CRM", "tools"),
            (r'\bsalesforce\b', "Salesforce", "tools"),
            (r'\bhubspot\b', "HubSpot", "tools"),
            (r'\bmailchimp\b', "Mailchimp", "tools"),
            (r'\ba/?b\s*test', "A/B Testing", "technical"),
            (r'\bcampaign\s*manage', "Campaign Management", "technical"),

            # --- Design & Creative ---
            (r'\bui/?ux\b|user\s*(experience|interface)', "UI/UX Design", "technical"),
            (r'\bwireframe', "Wireframing", "technical"),
            (r'\bprototyp', "Prototyping", "technical"),

            # --- Finance & Accounting ---
            (r'\bfinancial\s*(model|analy|report|forecast)', "Financial Analysis", "technical"),
            (r'\bexcel\b', "Microsoft Excel", "tools"),
            (r'\bpowerpoint\b', "PowerPoint", "tools"),

            # --- API & Architecture ---
            (r'\brest\s*api\b', "REST APIs", "technical"),
            (r'\bgraph\s*ql\b', "GraphQL", "technical"),
            (r'\bmicro\s*service', "Microservices", "technical"),
        ]

        new_technical = []
        new_tools = []

        for pattern, skill_name, category in skill_patterns:
            if skill_name.lower().strip() in existing_all:
                continue
            if re.search(pattern, corpus):
                if category == "technical":
                    new_technical.append(skill_name)
                else:
                    new_tools.append(skill_name)
                existing_all.add(skill_name.lower().strip())

        if new_technical:
            skills.setdefault("technical", []).extend(new_technical)
        if new_tools:
            skills.setdefault("tools", []).extend(new_tools)

        if new_technical or new_tools:
            resume_data["skills"] = skills

        return resume_data

    @staticmethod
    def _recover_missing_achievements(resume_data, raw_text):
        """
        Recover experience achievements that the extraction model dropped.
        Small models sometimes output experience entries with correct
        title/company/dates but empty achievements[], even when the source
        resume clearly has responsibilities listed.

        Generic strategy — handles multiple resume formats:
          1. Split the raw resume into lines.
          2. For each experience entry with empty achievements, find the
             anchor line in the raw text that matches the company or title.
          3. Collect the content lines that follow using format-agnostic
             detection: bullets (- • * >), numbered items (1. 2. (a)),
             and indented lines.
          4. Stop when hitting a section header, another experience block,
             or a double blank line.

        Only fills in achievements for entries that currently have none,
        to avoid overwriting what the model did extract.

        Args:
            resume_data (dict): Structured resume data (modified in place)
            raw_text (str): The original raw resume text

        Returns:
            dict: Resume data with recovered achievements
        """
        import re

        experience = resume_data.get("experience", [])
        if not experience or not raw_text:
            return resume_data

        # Check if any experience entry actually needs recovery
        needs_recovery = any(
            isinstance(exp, dict) and not exp.get("achievements")
            for exp in experience
        )
        if not needs_recovery:
            return resume_data

        lines = raw_text.split("\n")

        # ── Helper patterns ──────────────────────────────────────────

        # Section headers: ALL-CAPS lines, or well-known resume section names.
        # Intentionally strict — avoids false positives like "Work done:".
        _SECTION_NAMES = (
            r'EDUCATION|SKILLS|PROJECTS?|CERTIFICATIONS?|ACHIEVEMENTS?'
            r'|PUBLICATIONS?|LANGUAGES?|INTERESTS?|HOBBIES|REFERENCES?'
            r'|PERSONAL\s+(?:INFO|INFORMATION|DETAILS|PROJECTS)'
            r'|TECHNICAL\s+SKILLS|DOMAIN\s+KNOWLEDGE'
            r'|NEW\s+TECHNICAL\s+SKILLS'
            r'|WORK\s+(?:EXPERIENCE|HISTORY)'
            r'|PROFESSIONAL\s+(?:BACKGROUND|EXPERIENCE|SUMMARY)'
            r'|NOTABLE\s+ACHIEVEMENTS|EXTRA\s*CURRICULAR'
        )
        _SECTION_RE = re.compile(
            r'^(?:'
            r'[A-Z][A-Z\s&/]{2,}$'                   # ALL CAPS line (3+ chars)
            r'|(?:' + _SECTION_NAMES + r')(?:\s*:)?\s*$'  # known section name
            r')',
            re.IGNORECASE
        )

        # Date-range patterns (used to identify header lines to skip)
        _DATE_LINE_RE = re.compile(
            r'(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*'
            r'[\s,]+\d{4}|\d{4})\s*(?:[-–—]|\bto\b)',
            re.IGNORECASE
        )

        # List item detection (bullets, numbered items)
        _LIST_RE = re.compile(
            r'^(?:[\-•\*\>▸◦‣➤○]\s+'             # bullet markers
            r'|\d+[\.\)]\s+'                        # numbered: 1. 2)
            r'|\([a-z0-9]+\)\s+'                    # (a) (i) (1)
            r'|[a-z][\.\)]\s+)',                    # a. b)
            re.IGNORECASE
        )

        # Collect all known titles/companies to detect next-entry boundaries.
        # Build word-boundary regex for each to avoid substring false positives
        # like "marketing executive" matching inside "marketing executives".
        known_titles = {}   # lowercase title → compiled regex
        known_companies = {}
        for exp in experience:
            if isinstance(exp, dict):
                t = (exp.get("title") or "").strip().lower()
                c = (exp.get("company") or "").strip().lower()
                # Skip 1-2 char titles to avoid false boundary matches
                if t and len(t) > 2:
                    known_titles[t] = re.compile(
                        r'\b' + re.escape(t) + r'\b', re.IGNORECASE
                    )
                if c and len(c) > 2:
                    known_companies[c] = re.compile(
                        r'\b' + re.escape(c) + r'\b', re.IGNORECASE
                    )

        def _is_list_item(line_text):
            """Check if a line looks like a list item."""
            return bool(_LIST_RE.match(line_text.strip()))

        def _clean_list_marker(line_text):
            """Remove the leading list marker from a line."""
            return _LIST_RE.sub('', line_text.strip()).strip()

        def _is_anchor_candidate(line_text):
            """
            Check if a line could be an experience title/company line.
            Rejects bullets, long paragraphs, and indented lines — these
            are content lines that may contain a title/company as a substring.
            """
            stripped = line_text.strip()
            if not stripped:
                return False
            # Reject list items (bullets, numbered)
            if _is_list_item(stripped):
                return False
            # Reject long lines (>120 chars) — likely paragraph text
            if len(stripped) > 120:
                return False
            # Reject indented lines
            if line_text.startswith(("  ", "\t")):
                return False
            return True

        def _is_stop_line(line_text, current_title, current_company):
            """Check if a line signals we've left the current entry's content."""
            stripped = line_text.strip()
            if not stripped:
                return False

            # Section header
            if _SECTION_RE.match(stripped):
                return True

            # Another experience entry's title or company (word-boundary match)
            for t, pattern in known_titles.items():
                if t != current_title and pattern.search(stripped):
                    # Only treat as stop if line looks like a header, not content
                    if _is_anchor_candidate(line_text):
                        return True
            for c, pattern in known_companies.items():
                if c != current_company and pattern.search(stripped):
                    if _is_anchor_candidate(line_text):
                        return True

            return False

        # ── Main recovery loop ───────────────────────────────────────

        for exp in experience:
            if not isinstance(exp, dict):
                continue
            if exp.get("achievements"):
                continue

            title = (exp.get("title") or "").strip().lower()
            company = (exp.get("company") or "").strip().lower()

            if not title and not company:
                continue

            # Find the anchor line for this experience entry.
            # Strategy (in priority order):
            #   1. Company match on a header-like line (company names are unique)
            #   2. Title+company proximity: title on one line, company within
            #      3 lines — handles multi-line headers like:
            #        Software Engineer
            #        Nexus FinTech Startup, Mumbai
            #   3. Exact title match (line IS the title, not a substring)
            #   4. Title substring match (fallback)
            # Only header-candidate lines are considered (no bullets, no
            # paragraphs, no indented content).
            anchor_idx = None

            # Pass 1: company match (most reliable)
            if company and len(company) > 2:
                for i, line in enumerate(lines):
                    if not _is_anchor_candidate(line):
                        continue
                    if company in line.strip().lower():
                        anchor_idx = i
                        break

            # Pass 2: title + company proximity
            if anchor_idx is None and title and company:
                for i, line in enumerate(lines):
                    if not _is_anchor_candidate(line):
                        continue
                    if title in line.strip().lower():
                        # Asymmetric window: company usually follows title (+3 down, -2 up)
                        for k in range(max(0, i - 2), min(len(lines), i + 4)):
                            if company in lines[k].strip().lower():
                                anchor_idx = i
                                break
                        if anchor_idx is not None:
                            break

            # Pass 3: exact title match (line stripped == title)
            if anchor_idx is None and title and len(title) > 2:
                for i, line in enumerate(lines):
                    if not _is_anchor_candidate(line):
                        continue
                    if line.strip().lower() == title:
                        anchor_idx = i
                        break

            # Pass 4: title substring (fallback)
            if anchor_idx is None and title and len(title) > 2:
                for i, line in enumerate(lines):
                    if not _is_anchor_candidate(line):
                        continue
                    if title in line.strip().lower():
                        anchor_idx = i
                        break

            if anchor_idx is None:
                continue

            # Scan lines after the anchor to collect content.
            # Phase 1: skip header lines (title, company, dates, location)
            # Phase 2: collect content lines (bullets, numbered, indented)
            # Stop on: section header, next experience entry, double blank line
            bullets = []
            consecutive_blanks = 0
            header_skipping = True

            for j in range(anchor_idx + 1, len(lines)):
                raw_line = lines[j]
                stripped = raw_line.strip()

                # Blank line handling
                if not stripped:
                    consecutive_blanks += 1
                    if consecutive_blanks >= 2:  # double blank = section break in most resumes
                        break
                    if bullets:
                        # Peek ahead: if next non-blank line is a bullet, keep going
                        if j + 1 < len(lines) and _is_list_item(lines[j + 1]):
                            continue
                        break
                    continue

                consecutive_blanks = 0

                # Stop if we hit a section header or another entry
                if _is_stop_line(stripped, title, company):
                    break

                # List item — collect it
                if _is_list_item(stripped):
                    header_skipping = False
                    bullet_text = _clean_list_marker(stripped)
                    if bullet_text:
                        bullets.append(bullet_text)
                    continue

                # Still in header mode — skip metadata lines
                if header_skipping:
                    if (_DATE_LINE_RE.search(stripped)
                            or len(stripped) < 60  # short non-bullet lines are likely metadata
                            or stripped.lower().startswith("technologies")
                            or stripped.lower().startswith("duration")
                            or stripped.lower().startswith("work done")
                            or stripped.lower().startswith("location")):
                        continue
                    # Long non-bullet line — treat as plain-text achievement
                    header_skipping = False
                    bullets.append(stripped)
                    continue

                # Already collecting — indented continuation line
                if raw_line.startswith((" ", "\t")) and len(stripped) > 10:
                    bullets.append(stripped)
                else:
                    break

            if bullets:
                exp["achievements"] = bullets

        return resume_data

    @staticmethod
    def _recover_missing_contact_fields(resume_data, raw_text):
        """
        Recover contact fields (location, LinkedIn, GitHub) that the
        extraction model missed, using regex on the raw resume text.
        Small models reliably extract name/email/phone but often skip
        URLs and location strings.

        Only fills in fields that are currently empty — never overwrites
        values the model already extracted.

        Args:
            resume_data (dict): Structured resume data
            raw_text (str): Original resume text

        Returns:
            dict: Resume data with recovered contact fields
        """
        import re

        contact = resume_data.get("contact", {})
        if not isinstance(contact, dict):
            contact = {}
            resume_data["contact"] = contact

        # --- LinkedIn URL ---
        if not contact.get("linkedin", "").strip():
            linkedin_match = re.search(
                r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+',
                raw_text, re.IGNORECASE
            )
            if linkedin_match:
                contact["linkedin"] = linkedin_match.group(0)

        # --- GitHub URL ---
        if not contact.get("github", "").strip():
            github_match = re.search(
                r'(?:https?://)?(?:www\.)?github\.com/[\w-]+',
                raw_text, re.IGNORECASE
            )
            if github_match:
                contact["github"] = github_match.group(0)

        # --- Location ---
        # Common patterns: "Based in: City, State, Country"
        #                   "Location: City, State"
        if not contact.get("location", "").strip():
            loc_match = re.search(
                r'(?:based\s+in|location|address)\s*[:\-]\s*(.+)',
                raw_text, re.IGNORECASE
            )
            if loc_match:
                location = loc_match.group(1).strip().rstrip('|').strip()
                if location and len(location) < 100:
                    contact["location"] = location

        return resume_data

    def generate_follow_ups(self, resume_data, job_data=None):
        """
        Generate targeted follow-up questions based on gaps between the
        candidate's resume and the job requirements (if provided), or
        based on thin/empty sections for a general CV.

        Uses the extraction model (Qwen 1.5B) instead of Llama for speed.
        Sends a compact summary of sections rather than the full JSON dump.

        Args:
            resume_data (dict): Structured resume JSON from extraction step
            job_data (dict or None): Parsed job description, or None for general CV

        Returns:
            list: List of dicts with 'section' and 'question' keys,
                  or empty list if generation fails
        """
        # Build a compact section summary instead of dumping full JSON
        sections_summary = self._build_sections_summary(resume_data)

        if job_data:
            required = ", ".join(job_data.get("required_skills", []))
            preferred = ", ".join(job_data.get("preferred_skills", []))
            job_title = job_data.get("title", "the target role")

            prompt = FOLLOW_UP_PROMPT_JD.format(
                sections_summary=sections_summary,
                required_skills=required or "not specified",
                preferred_skills=preferred or "not specified",
                job_title=job_title
            )
        else:
            prompt = FOLLOW_UP_PROMPT_GENERAL.format(
                sections_summary=sections_summary
            )

        # Use extraction model (Qwen) — much faster than Llama for this task
        response = self.llm.extract(prompt)

        try:
            questions = clean_json_response(response)
            if isinstance(questions, list) and questions:
                return questions
        except (ValueError, KeyError):
            pass

        return []

    @staticmethod
    def _build_sections_summary(resume_data):
        """
        Build a compact text summary of which resume sections are populated,
        thin, or empty. This is sent to the LLM instead of the full JSON,
        keeping the prompt short and focused.

        Args:
            resume_data (dict): Structured resume data

        Returns:
            str: Compact summary like "experience: 2 entries, skills: 8 technical + 3 soft, projects: empty"
        """
        parts = []

        # Experience
        exp = resume_data.get("experience", [])
        if exp:
            titles = [e.get("title", "") for e in exp if isinstance(e, dict)]
            parts.append(f"experience: {len(exp)} entries ({', '.join(t for t in titles if t)})")
        else:
            parts.append("experience: empty")

        # Skills — include actual names so the LLM can check for overlap
        # with job requirements and avoid asking about skills already present
        skills = resume_data.get("skills", {})
        if isinstance(skills, dict):
            tech = skills.get("technical", [])
            soft = skills.get("soft", [])
            tools = skills.get("tools", [])
            if tech or soft or tools:
                skill_parts = []
                if tech:
                    skill_parts.append(f"technical: {', '.join(str(s) for s in tech)}")
                if soft:
                    skill_parts.append(f"soft: {', '.join(str(s) for s in soft)}")
                if tools:
                    skill_parts.append(f"tools: {', '.join(str(s) for s in tools)}")
                parts.append(f"skills: {'; '.join(skill_parts)}")
            else:
                parts.append("skills: empty")
        else:
            parts.append("skills: empty")

        # Projects
        projects = resume_data.get("projects", [])
        if projects:
            names = [p.get("name", "") for p in projects if isinstance(p, dict)]
            parts.append(f"projects: {len(projects)} entries ({', '.join(n for n in names if n)})")
        else:
            parts.append("projects: empty")

        # Education
        edu = resume_data.get("education", [])
        if edu:
            parts.append(f"education: {len(edu)} entries")
        else:
            parts.append("education: empty")

        # Professional summary
        summary = resume_data.get("professional_summary", "")
        if summary and len(summary) > 20:  # <20 chars is too thin to be a real summary
            parts.append(f"professional_summary: present ({len(summary)} chars)")
        else:
            parts.append("professional_summary: empty or thin")

        # Achievements
        achievements = resume_data.get("achievements", [])
        if achievements:
            parts.append(f"achievements: {len(achievements)} entries")
        else:
            parts.append("achievements: empty")

        # Certifications — include names so the LLM doesn't ask about
        # certifications the candidate already holds
        certs = resume_data.get("certifications", [])
        if certs:
            cert_names = [c.get("name", "") if isinstance(c, dict) else str(c)
                          for c in certs]
            cert_names = [n for n in cert_names if n]
            parts.append(f"certifications: {', '.join(cert_names)}" if cert_names
                         else f"certifications: {len(certs)} entries")
        else:
            parts.append("certifications: empty")

        return "\n".join(parts)

    # -------------------------------------------------------------------------
    # Structured Interview — programmatic enrichment questions
    # -------------------------------------------------------------------------
    def generate_structured_interview(self, resume_data):
        """
        Analyze extracted resume data and generate targeted interview
        questions to enrich sections before CV generation. Unlike the
        LLM-based follow-ups (which only target gaps), this probes for
        richer detail in existing sections too — especially measurable
        results in experience entries.

        Purely programmatic — no LLM call needed, so it's fast and
        deterministic.

        Args:
            resume_data (dict): Structured resume data from extraction

        Returns:
            list: List of {"section", "question"} dicts (0-7 questions)
        """
        questions = []

        # 1. Professional summary — if missing or very thin
        summary = resume_data.get("professional_summary", "")
        if not summary or len(summary.strip()) < 30:  # placeholder or single phrase
            questions.append({
                "section": "professional_summary",
                "question": ("How would you describe your professional identity "
                             "and career goals in 2-3 sentences?")
            })

        # 2. Experience enrichment — for the 2 most recent roles that lack
        #    measurable results (no digits in any achievement bullet)
        experience = resume_data.get("experience", [])
        enrich_count = 0
        for i, exp in enumerate(experience):
            if enrich_count >= 2:  # cap at 2 to avoid overwhelming user with questions
                break
            if not isinstance(exp, dict):
                continue
            achievements = exp.get("achievements", [])
            has_metrics = any(
                any(c.isdigit() for c in str(a))
                for a in achievements if a
            )
            if not has_metrics:
                title = exp.get("title", "your role")
                company = exp.get("company", "the company")
                questions.append({
                    "section": f"experience_enrich_{i}",
                    "question": (f"For your role as {title} at {company}, can you "
                                 f"share specific achievements with measurable "
                                 f"results? (e.g., 'Reduced costs by 30%', "
                                 f"'Led team of 8', 'Improved efficiency by 25%')")
                })
                enrich_count += 1

        # 3. Achievements — if section is empty
        achievements = resume_data.get("achievements", [])
        if not achievements:
            questions.append({
                "section": "achievements",
                "question": ("Have you received any awards, recognitions, or "
                             "notable accomplishments? (e.g., Employee of the "
                             "Year, industry award, conference speaker, "
                             "published research)")
            })

        # 4. Projects — if section is empty
        projects = resume_data.get("projects", [])
        if not projects:
            questions.append({
                "section": "projects",
                "question": ("Have you worked on any notable projects worth "
                             "highlighting? (personal, academic, or professional)")
            })

        # 5. Soft skills — if empty
        skills = resume_data.get("skills", {})
        soft_skills = skills.get("soft", []) if isinstance(skills, dict) else []
        if not soft_skills:
            questions.append({
                "section": "skills_soft",
                "question": ("What soft skills would you highlight? "
                             "(e.g., Leadership, Communication, Problem-solving, "
                             "Team Management, Mentoring)")
            })

        # 6. Certifications — if empty
        certs = resume_data.get("certifications", [])
        if not certs:
            questions.append({
                "section": "certifications",
                "question": ("Do you have any professional certifications? "
                             "(e.g., PMP, CPA, Six Sigma, board certification)")
            })

        return questions

    # Keywords that signal a non-answer (user is saying "already provided" in some form)
    _DISMISSAL_KEYWORDS = {
        "already", "given", "provided", "mentioned", "told", "answered",
        "above", "earlier", "before", "resume", "check", "refer", "see",
        "said", "stated", "listed", "wrote", "entered", "submitted",
    }

    # Single-word rejections
    _SKIP_WORDS = {
        "n/a", "na", "none", "no", "nope", "nothing", "skip", "pass",
        "nil", "nah", "nada", "-", ".", "idk", "dunno",
    }

    @classmethod
    def _is_non_answer(cls, text):
        """
        Check if a user's follow-up answer is a non-answer that should
        NOT be merged into the resume data. Uses keyword-based detection
        rather than exact phrase matching, so it catches variations like
        "I already told you", "it's in my resume", "check my resume",
        "mentioned it above", etc.

        A response is considered a non-answer if:
        1. It's a single skip/rejection word ("none", "skip", "no", etc.)
        2. It's short (< 20 words) AND contains 2+ dismissal keywords
           AND lacks concrete content (no commas, no digits, no colons)
        3. It's very short (< 5 words) AND contains any dismissal keyword

        Responses with commas (lists), digits (metrics/years), or colons
        (structured info) are always treated as real answers.

        Args:
            text (str): The user's answer text

        Returns:
            bool: True if the answer should be discarded
        """
        cleaned = text.strip().lower().rstrip("!.,;")
        # Strip leading non-alpha characters (e.g., "^No" → "no")
        cleaned = cleaned.lstrip("^~`!@#$%&*")

        # Single-character answers are never meaningful content
        if len(cleaned) <= 1:
            return True

        # Single-word skip
        if cleaned in cls._SKIP_WORDS:
            return True

        # If it has commas, numbers, or colons, it's likely real content
        # e.g. "Python, SQL, Docker" or "3 years of experience" or "Role: Manager"
        has_concrete = ("," in cleaned or
                        any(c.isdigit() for c in cleaned) or
                        ":" in cleaned)
        if has_concrete:
            return False

        words = cleaned.split()
        word_count = len(words)

        # Count how many dismissal keywords appear
        dismissal_count = sum(1 for w in words if w in cls._DISMISSAL_KEYWORDS)

        # Very short (< 5 words) with any dismissal keyword
        # e.g. "already given", "check above", "see resume"
        if word_count < 5 and dismissal_count >= 1:
            return True

        # Short (< 20 words) with 2+ dismissal keywords
        # e.g. "I have already provided this in my resume above"
        if word_count < 20 and dismissal_count >= 2:
            return True

        return False

    @staticmethod
    def merge_follow_up_answers(resume_data, answered_questions):
        """
        Merge user's follow-up answers into the existing resume data.
        Each answer is appended to the appropriate section based on the
        question's section tag. Filters out non-answers like "I have
        given already" to prevent data pollution.

        Args:
            resume_data (dict): Current structured resume data
            answered_questions (list): List of dicts with 'section',
                                       'question', and 'answer' keys

        Returns:
            dict: Updated resume data with follow-up answers merged in
        """
        for item in answered_questions:
            section = item.get("section", "")
            answer = item.get("answer", "").strip()
            if not answer:
                continue
            # Skip frustrated non-answers to prevent data pollution
            if ResumeDataExtractor._is_non_answer(answer):
                continue

            if section == "skills":
                # Parse comma-separated skills into the technical list
                skills = resume_data.get("skills", {})
                if not isinstance(skills, dict):
                    skills = {"technical": [], "soft": [], "tools": [], "languages": []}
                new_skills = [s.strip() for s in answer.split(",") if s.strip()]
                skills.setdefault("technical", []).extend(new_skills)
                resume_data["skills"] = skills

            elif section == "projects":
                projects = resume_data.get("projects", [])
                projects.append({
                    # Extract first clause as project name; 60-char cap fits PDF layout
                    "name": answer.split(".")[0].split(",")[0][:60],
                    "description": answer,
                    "technologies": []
                })
                resume_data["projects"] = projects

            elif section == "experience":
                experience = resume_data.get("experience", [])
                experience.append({
                    # Parse "Role at Company" pattern; 40-char fallback avoids full sentences
                    "title": answer.split(" at ")[0] if " at " in answer else answer[:40],
                    "company": answer.split(" at ")[1].split(",")[0] if " at " in answer else "",
                    "start_date": "",
                    "end_date": "",
                    "description": answer,
                    "achievements": []
                })
                resume_data["experience"] = experience

            elif section == "education":
                education = resume_data.get("education", [])
                if education:
                    # Append to the first education entry's details
                    edu = education[0]
                    if isinstance(edu, dict):
                        existing = edu.get("details", "")
                        edu["details"] = (existing + " " + answer).strip() if existing else answer
                else:
                    education.append({
                        "degree": answer,
                        "institution": "",
                        "year": "",
                        "gpa": "",
                        "details": ""
                    })
                resume_data["education"] = education

            elif section == "certifications":
                certs = resume_data.get("certifications", [])
                certs.append({"name": answer, "issuer": "", "year": ""})
                resume_data["certifications"] = certs

            elif section == "achievements":
                achievements = resume_data.get("achievements", [])
                achievements.append(answer)
                resume_data["achievements"] = achievements

            elif section == "professional_summary":
                # Append to or replace the professional summary
                existing = resume_data.get("professional_summary", "")
                if existing:
                    resume_data["professional_summary"] = existing + " " + answer
                else:
                    resume_data["professional_summary"] = answer

            elif section.startswith("experience_enrich_"):
                # Enrich an existing experience entry with new achievement
                # bullets (from the structured interview). Split the answer
                # on commas/semicolons to get individual bullet points.
                try:
                    idx = int(section.split("_")[-1])
                except ValueError:
                    continue
                experience = resume_data.get("experience", [])
                if idx < len(experience) and isinstance(experience[idx], dict):
                    new_bullets = [
                        b.strip() for b in
                        # Normalize semicolons so "Led team; Cut costs" splits into bullets
                        answer.replace(";", ",").split(",")
                        if b.strip()
                    ]
                    experience[idx].setdefault("achievements", []).extend(new_bullets)

            elif section == "skills_soft":
                # Route comma-separated values to soft skills sub-list
                skills = resume_data.get("skills", {})
                if not isinstance(skills, dict):
                    skills = {"technical": [], "soft": [], "tools": [], "languages": []}
                new_soft = [s.strip() for s in answer.split(",") if s.strip()]
                skills.setdefault("soft", []).extend(new_soft)
                resume_data["skills"] = skills

        return resume_data
