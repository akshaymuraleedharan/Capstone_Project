"""
main.py - Entry point for the CV Creation using LLMs pipeline.
Capstone Project CS01.

This script orchestrates the complete 5-step pipeline:
1. Project Setup - Initialize LLMs and verify connectivity
2. Resume Data Extraction - Parse input into structured JSON (Qwen 2.5 1.5B)
3. Job Description Parsing - Extract requirements and keywords (Qwen 2.5 1.5B)
4. Resume Tailoring & Generation - Create ATS-optimized CV content (Llama 3.2 3B)
5. User Review & Iterative Revision - Interactive editing and ATS feedback

Models Used (via HuggingFace Transformers - platform agnostic, no server required):
- Qwen 2.5 1.5B (Qwen/Qwen2.5-1.5B-Instruct): Extraction and analysis tasks
- Llama 3.2 3B (meta-llama/Llama-3.2-3B-Instruct): Content generation and rewriting tasks

Usage:
    python main.py --hf-token YOUR_HF_TOKEN
    python main.py --hf-token YOUR_HF_TOKEN --install-deps
"""

import argparse
import sys
import os
import subprocess

# Progress bar suppression is handled in _set_hf_offline_if_cached() below.
# When all models are already cached, bars are hidden (they're just noise).
# When a new model needs downloading, bars are kept visible so the user
# can see download progress.


def check_and_install_dependencies():
    """
    Only runs when --install-deps is passed. Checks for missing Python
    packages, prompts before installing, then restarts the program.

    PyTorch is handled separately since it requires platform-specific
    installation (CPU, CUDA, MPS).

    Without --install-deps this function is never called and imports
    fail naturally with a standard Python traceback.
    """
    if "--install-deps" not in sys.argv:
        return

    # Platform-agnostic dependencies (safe to pip install directly)
    DEPENDENCIES = [
        ("transformers>=4.40.0", "transformers"),
        ("accelerate>=0.26.0",  "accelerate"),
        ("hf_xet>=1.0.0",       "hf_xet"),
        ("pdfplumber>=0.11.0",  "pdfplumber"),
        ("python-docx>=1.1.0",  "docx"),
        ("fpdf2>=2.8.0",        "fpdf"),
    ]

    print("\n  Checking dependencies...")

    # --- Check PyTorch separately (platform-specific) ---
    torch_installed = True
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            __import__("torch")
    except ImportError:
        torch_installed = False

    if not torch_installed:
        print("\n  PyTorch is required but not installed.")
        print("  PyTorch installation is platform-specific (CPU, CUDA, MPS).")
        print("  Please install the correct version for your system:")
        print("    https://pytorch.org/get-started/locally/")
        print("\n  Example commands:")
        print('    macOS (MPS):            pip install "torch>=2.0.0"')
        print('    Windows/Linux (CUDA):   pip install "torch>=2.0.0"')
        print('    Windows/Linux (CPU):    pip install "torch>=2.0.0"')
        print("\n  After installing PyTorch, run this program again.")
        sys.exit(1)

    # --- Check remaining dependencies ---
    missing = []
    for pip_name, import_name in DEPENDENCIES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if not missing:
        print("  All dependencies are available.")
        return

    # --- Prompt and install ---
    print(f"\n  The following {len(missing)} package(s) are missing:")
    for pkg in missing:
        print(f"    - {pkg}")

    print(f"\n  Command that will be run: pip install {' '.join(missing)}")

    choice = input("\n  Install now? (y/n): ").strip().lower()
    if choice != "y":
        print("\n  These dependencies are required to run the pipeline.")
        print("  Install them manually and run this program again.")
        sys.exit(1)

    print()
    failed = []
    for pkg in missing:
        print(f"  Installing {pkg}...", end=" ", flush=True)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )
            print("done.")
        except subprocess.CalledProcessError:
            print("FAILED.")
            failed.append(pkg)

    if failed:
        print(f"\n  ERROR: Failed to install: {', '.join(failed)}")
        print(f"  Please install manually and run this program again.")
        sys.exit(1)

    print("\n  All dependencies installed successfully.")
    print("  Restarting program...\n")
    os.execv(sys.executable, [sys.executable] + sys.argv)


# =============================================================================
# Corporate / proxy SSL fix — MUST run before any import of 'transformers'.
#
# transformers 5.x reads HF_HUB_OFFLINE at import time, not at call time.
# With --install-deps, check_and_install_dependencies() does
# __import__("transformers") to check if the package is present, so the
# env var must be set before that call too.
#
# When both models are already cached locally, we set HF_HUB_OFFLINE=1 to
# suppress all outbound Hub network calls.  This prevents SSL certificate
# errors on corporate networks that use a TLS-inspecting proxy.
# First-run behaviour (models not yet cached) is unaffected: the flag is not
# set, so downloading works normally.
# =============================================================================
def _set_hf_offline_if_cached():
    """
    If all HuggingFace models being used are already in the local cache, set
    HF_HUB_OFFLINE=1 so that transformers never attempts network calls.
    This prevents SSL certificate errors on corporate proxy networks.
    Only sets the flag when ALL models are cached — first-run downloads still work.
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

    # Always suppress progress bars — they clutter the terminal
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    extraction_model = "Qwen/Qwen2.5-1.5B-Instruct"
    generation_model = "meta-llama/Llama-3.2-3B-Instruct"

    models_to_check = [extraction_model, generation_model]
    all_cached = all(
        os.path.isdir(
            # HF Hub stores cached models as "models--org--name" on disk
            os.path.join(cache_dir, "models--" + m.replace("/", "--"))
        )
        for m in models_to_check
    )
    if all_cached:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # setdefault: don't override user's explicit setting

_set_hf_offline_if_cached()


# =============================================================================
# Handle --help before local imports so it works even without dependencies
# =============================================================================
if "--help" in sys.argv or "-h" in sys.argv:
    print("usage: python main.py [--hf-token TOKEN] [--install-deps]")
    print()
    print("CV Creation using LLMs - Automated resume builder (Capstone Project CS01)")
    print()
    print("options:")
    print("  --hf-token TOKEN   HuggingFace access token for gated models (Llama 3.2).")
    print("                     Can also be set via the HF_TOKEN environment variable.")
    print("  --install-deps     Interactively install missing dependencies with")
    print("                     confirmation. Without this flag, missing dependencies")
    print("                     fail naturally on import.")
    print("  -h, --help         Show this help message and exit.")
    sys.exit(0)

# =============================================================================
# With --install-deps: check and install missing packages BEFORE importing
# local modules (they import transformers/torch at load time).
# Without --install-deps: this is a no-op — missing imports fail naturally.
# =============================================================================
check_and_install_dependencies()


# Local module imports
from llm_handler import LLMHandler
from input_parser import InputParser
from data_extractor import ResumeDataExtractor
from job_parser import JobDescriptionParser
from cv_generator import CVGenerator
from output_builder import OutputBuilder
from utils import (
    display_banner,
    display_cv_preview,
    display_ats_report,
    get_user_choice
)


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with hf_token and install_deps fields
    """
    parser = argparse.ArgumentParser(
        description="CV Creation using LLMs - Automated resume builder "
                    "(Capstone Project CS01)"
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace access token for gated models (Llama 3.2). "
             "Can also be set via the HF_TOKEN environment variable. "
             "Get a free token at: https://huggingface.co/settings/tokens"
    )
    parser.add_argument(
        "--install-deps", action="store_true", default=False,
        help="Interactively install missing dependencies with confirmation. "
             "Without this flag, missing dependencies are reported and the "
             "program exits."
    )
    return parser.parse_args()


def get_resume_input(input_parser):
    """
    Acquire resume data from a file or manual input.
    Presents an interactive menu for the user to choose input method.

    Args:
        input_parser (InputParser): Configured input parser instance

    Returns:
        str: Raw resume text content
    """
    # Interactive menu
    print("\n  How would you like to provide your resume data?")
    print("  [1] Load from file (PDF, DOCX, or TXT)")
    print("  [2] Enter manually (free-form text)")
    print("  [3] Enter manually (guided section-by-section)")

    choice = get_user_choice("\n  Enter choice (1-3): ", ["1", "2", "3"])

    if choice == "1":
        # Retry loop: re-prompt until the user provides a valid file or exits
        while True:
            file_path = input("  Enter file path (provide full path if file is not in the current directory): ").strip()
            file_path = file_path.strip("'\"")  # strip quotes from terminal drag-and-drop
            try:
                return input_parser.read_file(file_path)
            except (FileNotFoundError, ValueError) as e:
                print(f"\n  ERROR: {e}")
                print("  [1] Try a different file path")
                print("  [2] Exit")
                retry = get_user_choice("\n  Enter choice (1-2): ", ["1", "2"])
                if retry == "2":
                    print("\n  Exiting.")
                    sys.exit(0)
    elif choice == "2":
        return input_parser.read_manual_input()
    elif choice == "3":
        return input_parser.read_manual_structured()


def _text_similarity(text_a, text_b):
    """
    Compute a simple word-overlap similarity ratio between two texts.
    Uses set intersection of lowercased words — no external libraries needed.
    Returns a float between 0.0 (no overlap) and 1.0 (identical word sets).

    Args:
        text_a (str): First text
        text_b (str): Second text

    Returns:
        float: Jaccard similarity of word sets (intersection / union)
    """
    if not text_a or not text_b:
        return 0.0
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _validate_jd_input(raw_text, resume_text):
    """
    Check if the provided job description text is likely a duplicate of
    the resume already provided. Uses generic text similarity — no
    hardcoded patterns, works for any language or format.

    Args:
        raw_text (str): The text the user entered as a job description
        resume_text (str): The raw resume text from Step 2

    Returns:
        bool: True if the text looks like a duplicate of the resume
    """
    if not raw_text or not resume_text:
        return False
    return _text_similarity(raw_text, resume_text) > 0.5  # >50% Jaccard = likely same document


def get_job_description_input(input_parser, resume_text=None):
    """
    Acquire job description from a file or manual input.
    Presents an interactive menu. User can skip this step.

    Validates input against the already-provided resume text to catch
    accidental pastes (e.g. pasting the CV instead of a JD).

    Args:
        input_parser (InputParser): Configured input parser instance
        resume_text (str or None): Raw resume text from Step 2, used to
                                    detect accidental duplicate pastes

    Returns:
        str or None: Raw job description text, or None if skipped
    """
    while True:
        # Interactive menu
        print("\n  Would you like to provide a job description for tailored CV?")
        print("  [1] Load from file (PDF, DOCX, or TXT)")
        print("  [2] Paste job description text")
        print("  [3] Enter manually (guided section-by-section)")
        print("  [4] Skip (generate a general-purpose CV)")

        choice = get_user_choice("\n  Enter choice (1-4): ", ["1", "2", "3", "4"])

        raw_text = None
        if choice == "1":
            # Retry loop: re-prompt until user provides a valid file or bails out
            while True:
                file_path = input("  Enter file path (provide full path if file is not in the current directory): ").strip()
                file_path = file_path.strip("'\"")  # strip quotes from terminal drag-and-drop
                try:
                    raw_text = input_parser.read_file(file_path)
                    break
                except (FileNotFoundError, ValueError) as e:
                    print(f"\n  ERROR: {e}")
                    print("  [1] Try a different file path")
                    print("  [2] Go back to job description menu")
                    retry = get_user_choice("\n  Enter choice (1-2): ", ["1", "2"])
                    if retry == "2":
                        break
            # If user bailed out of retry loop, re-show the JD menu
            if raw_text is None and choice == "1":
                continue
        elif choice == "2":
            print("\n  Paste the job description below.")
            print("  (Press Enter twice on an empty line to finish)\n")
            raw_text = input_parser.read_manual_input()
        elif choice == "3":
            return input_parser.read_job_description_structured()
        elif choice == "4":
            print("  Skipping job description. Will generate a general-purpose CV.")
            return None

        # Validate: detect if user accidentally pasted their resume again
        if raw_text and _validate_jd_input(raw_text, resume_text):
            print("\n  WARNING: This text looks very similar to your resume.")
            print("     You may have pasted your CV instead of a job description.")
            print()
            print("     [1] Re-enter the job description")
            print("     [2] Use it anyway")
            print("     [3] Skip job description")
            retry = get_user_choice("\n  Enter choice (1-3): ", ["1", "2", "3"])
            if retry == "1":
                continue  # Loop back to JD input menu
            elif retry == "3":
                print("  Skipping job description. Will generate a general-purpose CV.")
                return None
            # retry == "2": fall through and return the text

        return raw_text


def _format_section_value(key, value):
    """
    Format a resume section's current value for display during editing.
    Shows a compact, readable summary so the user knows what they're fixing.

    Args:
        key (str): Section key (e.g., 'name', 'experience', 'skills')
        value: The current value (str, list, or dict)

    Returns:
        str: Human-readable summary of the current value
    """
    if isinstance(value, str):
        return value if value else "(empty)"

    if isinstance(value, dict):
        # Contact or skills — show key: value pairs
        parts = []
        for k, v in value.items():
            if isinstance(v, list) and v:
                parts.append(f"    {k.title()}: {', '.join(str(s) for s in v)}")
            elif isinstance(v, str) and v:
                parts.append(f"    {k.title()}: {v}")
        return "\n".join(parts) if parts else "(empty)"

    if isinstance(value, list):
        if not value:
            return "(empty)"
        lines = []
        for i, item in enumerate(value, 1):
            if isinstance(item, dict):
                # Experience, education, projects, certifications
                if "title" in item:
                    line = f"{item.get('title', '')} at {item.get('company', '')}"
                    achs = item.get("achievements", [])
                    if achs:
                        line += f" ({len(achs)} bullets)"
                elif "degree" in item:
                    line = f"{item.get('degree', '')} | {item.get('institution', '')}"
                elif "name" in item and "description" in item:
                    line = f"{item.get('name', '')}: {item.get('description', '')[:60]}"
                elif "name" in item:
                    line = item.get("name", str(item))
                else:
                    line = str(item)
                lines.append(f"    {i}. {line}")
            else:
                lines.append(f"    {i}. {item}")
        return "\n".join(lines)

    return str(value) if value else "(empty)"


def _edit_string_section(current_value, section_label):
    """
    Edit a simple string section (name, professional_summary, years_experience).

    Args:
        current_value (str): Current value
        section_label (str): Human-readable section name

    Returns:
        str: Updated value
    """
    print(f"\n  Current {section_label}:")
    print(f"    {current_value if current_value else '(empty)'}")
    new_val = input(f"\n  New {section_label} (Enter to keep current): ").strip()
    return new_val if new_val else current_value


def _edit_contact_section(contact):
    """
    Edit individual contact fields.

    Args:
        contact (dict): Current contact info dict

    Returns:
        dict: Updated contact dict
    """
    print("\n  Current contact info:")
    fields = ["email", "phone", "location", "linkedin", "github", "portfolio"]
    for f in fields:
        val = contact.get(f, "")
        if val:
            print(f"    {f.title()}: {val}")

    print("\n  Edit fields (press Enter to keep current):")
    for f in fields:
        current = contact.get(f, "")
        prompt = f"  {f.title()} [{current}]: " if current else f"  {f.title()}: "
        new_val = input(prompt).strip()
        if new_val:
            contact[f] = new_val
    return contact


def _edit_skills_section(skills):
    """
    Edit skills by category. Shows current skills per category and lets
    the user replace them with comma-separated input.

    Args:
        skills (dict): Current skills dict with category keys

    Returns:
        dict: Updated skills dict
    """
    categories = ["technical", "soft", "tools", "languages"]
    print("\n  Current skills:")
    for cat in categories:
        vals = skills.get(cat, [])
        if vals:
            print(f"    {cat.title()}: {', '.join(str(s) for s in vals)}")

    print("\n  Edit by category (comma-separated, Enter to keep current):")
    for cat in categories:
        current = skills.get(cat, [])
        current_str = ", ".join(str(s) for s in current)
        prompt = f"  {cat.title()} [{current_str}]: " if current_str else f"  {cat.title()}: "
        new_val = input(prompt).strip()
        if new_val:
            skills[cat] = [s.strip() for s in new_val.split(",") if s.strip()]
    return skills


def _display_list_items(items, section_label):
    """Display numbered items in a list section."""
    print(f"\n  Current {section_label}:")
    if not items:
        print("    (empty)")
        return
    for i, item in enumerate(items, 1):
        if isinstance(item, dict):
            if "title" in item:
                achs = item.get("achievements", [])
                print(f"    [{i}] {item.get('title', '')} at {item.get('company', '')}")
                for ach in achs[:3]:
                    print(f"        - {ach}")
                if len(achs) > 3:
                    print(f"        ... and {len(achs) - 3} more")
            elif "degree" in item:
                print(f"    [{i}] {item.get('degree', '')} | {item.get('institution', '')} ({item.get('year', '')})")
            elif "name" in item and "description" in item:
                print(f"    [{i}] {item.get('name', '')}: {item.get('description', '')[:80]}")
            elif "name" in item:
                print(f"    [{i}] {item.get('name', '')}")
            else:
                print(f"    [{i}] {item}")
        else:
            print(f"    [{i}] {item}")


def _edit_list_section(items, section_label, item_label, resume_data=None):
    """
    Edit a list section (experience, education, projects, certifications,
    achievements). Shows numbered items and lets the user edit, remove, add,
    move to another section, or finish via a numbered menu.
    Re-displays items after each action.

    Args:
        items (list): Current list of items
        section_label (str): Human-readable section name
        item_label (str): Singular item name (e.g., 'entry', 'certification')
        resume_data (dict or None): Full resume data, needed for move operation

    Returns:
        list: Updated list
    """
    # Sections that can receive moved items (label → resume_data key)
    _MOVE_TARGETS = {
        "certifications": "Certifications",
        "achievements": "Achievements",
        "projects": "Projects",
    }

    while True:
        _display_list_items(items, section_label)

        print(f"\n  Options:")
        print(f"    [1] Edit a {item_label} (by number)")
        print(f"    [2] Remove a {item_label} (by number)")
        print(f"    [3] Add a new {item_label}")
        if resume_data and section_label in _MOVE_TARGETS:
            print(f"    [4] Move a {item_label} to another section")
            print(f"    [5] Done editing {section_label}")
            valid_choices = ["1", "2", "3", "4", "5"]
        else:
            print(f"    [4] Done editing {section_label}")
            valid_choices = ["1", "2", "3", "4"]

        action = input(f"\n  Enter choice (1-{len(valid_choices)}): ").strip()

        # "Done" is the last option (4 or 5 depending on move support)
        done_choice = valid_choices[-1]
        if action == done_choice:
            break

        elif action == "1" and items:
            num = input(f"  Edit which {item_label}? (number): ").strip()
            try:
                idx = int(num) - 1
                if 0 <= idx < len(items):
                    item = items[idx]
                    if isinstance(item, dict):
                        # Edit dict fields one at a time
                        print(f"  Editing {item_label} [{num}]. Press Enter to keep current value.")
                        for field_key in list(item.keys()):
                            if field_key == "achievements":
                                # Special handling for achievement lists
                                current_achs = item.get("achievements", [])
                                if current_achs:
                                    print(f"    Current achievements:")
                                    for ai, ach in enumerate(current_achs, 1):
                                        print(f"      {ai}. {ach}")
                                edit_achs = input(f"    Re-enter achievements? (y/N): ").strip().lower()
                                if edit_achs == "y":
                                    print("    New achievements (one per line, blank to finish):")
                                    new_achs = []
                                    while True:
                                        ach = input("      - ").strip()
                                        if not ach:
                                            break
                                        new_achs.append(ach)
                                    item["achievements"] = new_achs
                            elif isinstance(item[field_key], list):
                                current = ", ".join(str(v) for v in item[field_key])
                                label = field_key.replace("_", " ").title()
                                new_val = input(f"    {label} [{current}]: ").strip()
                                if new_val:
                                    item[field_key] = [v.strip() for v in new_val.split(",") if v.strip()]
                            else:
                                current = item[field_key]
                                label = field_key.replace("_", " ").title()
                                new_val = input(f"    {label} [{current}]: ").strip()
                                if new_val:
                                    item[field_key] = new_val
                    else:
                        # Edit plain string
                        new_val = input(f"  New value [{item}]: ").strip()
                        if new_val:
                            items[idx] = new_val
                    print(f"  Updated.")
                else:
                    print(f"  Invalid number. Enter 1 to {len(items)}.")
            except ValueError:
                print("  Please enter a number.")

        elif action == "2" and items:
            num = input(f"  Remove which {item_label}? (number): ").strip()
            try:
                idx = int(num) - 1
                if 0 <= idx < len(items):
                    removed = items.pop(idx)
                    label = removed
                    if isinstance(removed, dict):
                        label = removed.get("title", removed.get("name", removed.get("degree", str(removed))))
                    print(f"  Removed: {label}")
                else:
                    print(f"  Invalid number. Enter 1 to {len(items)}.")
            except ValueError:
                print("  Please enter a number.")

        elif action == "3":
            if section_label in ("experience", "work experience"):
                title = input("  Job Title: ").strip()
                if title:
                    company = input("  Company: ").strip()
                    start = input("  Start Date: ").strip()
                    end = input("  End Date (or 'Present'): ").strip()
                    print("  Achievements (one per line, blank to finish):")
                    achs = []
                    while True:
                        ach = input("    - ").strip()
                        if not ach:
                            break
                        achs.append(ach)
                    items.append({
                        "title": title, "company": company,
                        "start_date": start, "end_date": end,
                        "achievements": achs
                    })

            elif section_label == "education":
                degree = input("  Degree: ").strip()
                if degree:
                    institution = input("  Institution: ").strip()
                    year = input("  Year: ").strip()
                    gpa = input("  GPA (optional): ").strip()
                    items.append({
                        "degree": degree, "institution": institution,
                        "year": year, "gpa": gpa
                    })

            elif section_label == "projects":
                name = input("  Project Name: ").strip()
                if name:
                    desc = input("  Description: ").strip()
                    tech = input("  Technologies (comma-separated): ").strip()
                    items.append({
                        "name": name, "description": desc,
                        "technologies": [t.strip() for t in tech.split(",") if t.strip()]
                    })

            elif section_label == "certifications":
                name = input("  Certification: ").strip()
                if name:
                    issuer = input("  Issuer (optional): ").strip()
                    year = input("  Year (optional): ").strip()
                    items.append({"name": name, "issuer": issuer, "year": year})

            else:
                # achievements, publications, or other string lists
                text = input(f"  New {item_label}: ").strip()
                if text:
                    items.append(text)

        elif action == "4" and resume_data and section_label in _MOVE_TARGETS and items:
            # Move an item to another section
            num = input(f"  Move which {item_label}? (number): ").strip()
            try:
                idx = int(num) - 1
                if 0 <= idx < len(items):
                    # Show target sections (exclude current)
                    targets = [(k, v) for k, v in _MOVE_TARGETS.items()
                               if k != section_label]
                    print("  Move to which section?")
                    for ti, (tkey, tlabel) in enumerate(targets, 1):
                        print(f"    [{ti}] {tlabel}")
                    tchoice = input(f"  Enter choice (1-{len(targets)}): ").strip()
                    try:
                        tidx = int(tchoice) - 1
                        if 0 <= tidx < len(targets):
                            target_key = targets[tidx][0]
                            moved = items.pop(idx)
                            # Convert to appropriate format for target
                            text = moved
                            if isinstance(moved, dict):
                                text = moved.get("name", moved.get("title", str(moved)))
                            # Add to target section
                            target_list = resume_data.get(target_key, [])
                            if target_key == "certifications":
                                target_list.append({"name": text, "issuer": "", "year": ""})
                            else:
                                target_list.append(text if isinstance(text, str) else str(text))
                            resume_data[target_key] = target_list
                            print(f"  Moved to {targets[tidx][1]}.")
                        else:
                            print("  Invalid choice.")
                    except ValueError:
                        print("  Please enter a number.")
                else:
                    print(f"  Invalid number. Enter 1 to {len(items)}.")
            except ValueError:
                print("  Please enter a number.")

        else:
            print(f"  Invalid choice. Enter 1-{len(valid_choices)}.")

    return items


def verify_extracted_data(extractor, resume_data):
    """
    Display extracted resume data and let the user verify or correct it.
    Offers section-level editing so the user can fix individual sections
    without re-entering everything.

    The editable sections are built dynamically from the extracted data
    keys — not hardcoded — so new sections are automatically supported.

    Args:
        extractor (ResumeDataExtractor): The data extractor instance
        resume_data (dict): Extracted resume data JSON

    Returns:
        dict: Verified (and possibly corrected) resume data
    """
    extractor.display_extracted_data(resume_data)

    print("\n  The extracted resume data is displayed above.")
    print("  Please review it and choose an option:")
    print("  [1] Looks good, continue")
    print("  [2] Edit specific sections")
    print("  [3] Re-enter everything")

    choice = get_user_choice("\n  Enter choice (1-3): ", ["1", "2", "3"])

    if choice == "1":
        return resume_data

    if choice == "3":
        print("\n  Please re-enter your resume information.")
        parser = InputParser()
        raw_text = parser.read_manual_structured()
        resume_data = extractor.extract_from_text(raw_text)
        extractor.display_extracted_data(resume_data)
        return resume_data

    # ── Section-level editing ─────────────────────────────────────
    # Build the editable sections dynamically from the data keys.
    # Human-readable labels for known keys; unknown keys get auto-labelled.
    _LABELS = {
        "name": "Name",
        "contact": "Contact Info",
        "professional_summary": "Professional Summary",
        "years_experience": "Years of Experience",
        "education": "Education",
        "experience": "Work Experience",
        "skills": "Skills",
        "certifications": "Certifications",
        "projects": "Projects",
        "achievements": "Achievements",
        "publications": "Publications",
    }

    while True:
        # Build menu from data keys that have content
        editable = []
        for key, value in resume_data.items():
            # Skip empty sections
            if not value or value == {} or value == []:
                continue
            # Skip internal metadata fields (e.g. _sections_summary)
            if key.startswith("_"):
                continue
            label = _LABELS.get(key, key.replace("_", " ").title())
            editable.append((key, label))

        if not editable:
            print("  No editable sections found.")
            break

        print("\n  Which section would you like to edit?")
        for i, (key, label) in enumerate(editable, 1):
            print(f"  [{i}] {label}")
        print(f"  [{len(editable) + 1}] Done editing")

        valid = [str(i) for i in range(1, len(editable) + 2)]
        pick = get_user_choice(f"\n  Enter choice (1-{len(editable) + 1}): ", valid)
        pick_idx = int(pick) - 1

        if pick_idx >= len(editable):
            break  # Done

        key, label = editable[pick_idx]
        value = resume_data[key]

        # Dispatch to the right editor based on value type
        if key == "contact" and isinstance(value, dict):
            resume_data[key] = _edit_contact_section(value)

        elif key == "skills" and isinstance(value, dict):
            resume_data[key] = _edit_skills_section(value)

        elif isinstance(value, str):
            resume_data[key] = _edit_string_section(value, label)

        elif isinstance(value, list):
            # Determine item label for display
            if key == "experience":
                item_label = "entry"
            elif key == "education":
                item_label = "entry"
            elif key == "certifications":
                item_label = "certification"
            elif key == "projects":
                item_label = "project"
            else:
                item_label = "item"
            resume_data[key] = _edit_list_section(value, key, item_label, resume_data)

        elif isinstance(value, dict):
            # Generic dict — edit as key-value pairs
            print(f"\n  Current {label}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
            print("\n  Edit fields (press Enter to keep current):")
            for k in list(value.keys()):
                current = value[k]
                new_val = input(f"  {k} [{current}]: ").strip()
                if new_val:
                    value[k] = new_val
            resume_data[key] = value

        # Show updated data after each edit
        print("\n  --- Updated ---")
        print(f"  {label}:")
        print(f"  {_format_section_value(key, resume_data[key])}")

    return resume_data


def follow_up_round(extractor, resume_data, job_data=None):
    """
    Use the extraction LLM (Qwen) to analyze gaps in the candidate's
    resume data — optionally compared against a job description — and
    ask targeted follow-up questions. The user's answers are merged
    back into the resume data to strengthen it before CV generation.

    When a JD is provided, questions focus on gaps between the resume
    and job requirements. Without a JD, questions target empty or thin
    resume sections for a general-purpose CV.

    Args:
        extractor (ResumeDataExtractor): The data extractor instance
        resume_data (dict): Verified resume data from extraction step
        job_data (dict or None): Parsed job description, or None

    Returns:
        dict: Updated resume data with follow-up answers merged in
    """
    follow_ups = extractor.generate_follow_up_questions(resume_data, job_data)

    if not follow_ups:
        print("  No additional questions needed — profile looks complete.")
        return resume_data

    if job_data:
        print("\n" + "=" * 60)
        print("  STRENGTHENING YOUR PROFILE FOR THIS ROLE")
        print("=" * 60)
        print("\n  Based on the job requirements, a few more details would")
        print("  help tailor your CV. Press Enter to skip any question.\n")
    else:
        print("\n" + "=" * 60)
        print("  STRENGTHENING YOUR PROFILE")
        print("=" * 60)
        print("\n  A few more details would help generate a stronger CV.")
        print("  Press Enter to skip any question.\n")

    answered = []
    for i, item in enumerate(follow_ups, 1):
        question = item.get("question", "")
        section = item.get("section", "")
        if not question:
            continue

        print(f"  [{i}/{len(follow_ups)}] {question}")
        answer = input("  > ").strip()

        if answer:
            answered.append({
                "section": section,
                "question": question,
                "answer": answer
            })
        print()

    if answered:
        # Filter out non-answers before merging and counting
        from data_extractor import ResumeDataExtractor
        real_answers = [a for a in answered
                        if not ResumeDataExtractor._is_non_answer(a["answer"])]
        resume_data = extractor.merge_follow_up_answers(resume_data, answered)
        if real_answers:
            print(f"  Profile updated with {len(real_answers)} additional detail(s).")
        else:
            print("  No usable details provided. Continuing with current data.")
    else:
        print("  No additional details provided. Continuing with current data.")

    return resume_data


def structured_interview_round(extractor, resume_data):
    """
    Run a programmatic structured interview to enrich resume data
    before CV generation. Unlike the LLM-based follow-ups which
    target gaps, this probes for richer detail in existing sections
    — especially measurable results in experience entries.

    Questions are generated deterministically (no LLM call) by
    analyzing the extracted data for thin or metric-free sections.

    Args:
        extractor (ResumeDataExtractor): The data extractor instance
        resume_data (dict): Verified resume data from extraction step

    Returns:
        dict: Updated resume data with interview answers merged in
    """
    questions = extractor.generate_structured_interview(resume_data)

    if not questions:
        print("  Profile looks comprehensive — no enrichment questions needed.")
        return resume_data

    print("\n" + "=" * 60)
    print("  PROFILE ENRICHMENT INTERVIEW")
    print("=" * 60)
    print("\n  A few quick questions to strengthen your CV.")
    print("  Press Enter to skip any question.\n")

    answered = []
    for i, item in enumerate(questions, 1):
        question = item.get("question", "")
        section = item.get("section", "")
        if not question:
            continue

        print(f"  [{i}/{len(questions)}] {question}")
        answer = input("  > ").strip()

        if answer:
            answered.append({
                "section": section,
                "question": question,
                "answer": answer
            })
        print()

    if answered:
        # Filter out non-answers before merging and counting
        from data_extractor import ResumeDataExtractor
        real_answers = [a for a in answered
                        if not ResumeDataExtractor._is_non_answer(a["answer"])]
        resume_data = extractor.merge_follow_up_answers(resume_data, answered)
        if real_answers:
            print(f"  Profile enriched with {len(real_answers)} additional detail(s).")
        else:
            print("  No usable details provided. Continuing with current data.")
    else:
        print("  No additional details provided. Continuing with current data.")

    return resume_data


def revision_loop(cv_generator, cv_content, contact_info, resume_data,
                  job_data, ats_report):
    """
    Interactive two-phase review loop for the generated CV.

    Phase 1 (ATS Review): Shown only when a JD was provided. Displays
    the ATS compatibility report and offers ATS-specific actions
    (optimize keywords, re-score). User proceeds to Phase 2 when done.

    Phase 2 (CV Review): Shows the full CV preview by default, then
    offers editing actions (edit sections, regenerate summary, accept).

    Args:
        cv_generator (CVGenerator): The CV generator instance
        cv_content (dict): Generated CV content by section
        contact_info (dict): Contact information for the candidate
        resume_data (dict): Original extracted resume data
        job_data (dict or None): Parsed job description (None if not provided)
        ats_report (dict or None): ATS scoring report (None if no JD)

    Returns:
        dict: Final approved CV content dictionary
    """
    # ── Phase 1: ATS Review (only when JD was provided) ──────────────
    if ats_report:
        display_ats_report(ats_report)
        while True:
            print("\n  Options:")
            print("  [1] Auto-optimize for missing ATS keywords")
            print("  [2] Re-score ATS compatibility")
            print("  [3] Preview generated CV")
            print("  [4] Continue to CV review")

            choice = get_user_choice("\n  Enter choice (1-4): ", ["1", "2", "3", "4"])

            if choice == "1":
                if ats_report.get("missing_keywords"):
                    cv_content = cv_generator.revise_for_keywords(
                        cv_content, ats_report["missing_keywords"]
                    )
                    print("  CV updated. Re-scoring ATS compatibility...")
                    ats_report = cv_generator.score_ats_compatibility(
                        cv_content, job_data, contact_info
                    )
                    display_ats_report(ats_report)
                else:
                    print("  No missing keywords identified. Nothing to optimize.")

            elif choice == "2":
                ats_report = cv_generator.score_ats_compatibility(
                    cv_content, job_data, contact_info
                )
                display_ats_report(ats_report)

            elif choice == "3":
                display_cv_preview(cv_content, contact_info)

            elif choice == "4":
                break

    # ── Phase 2: CV Review ───────────────────────────────────────────
    while True:
        print("\n" + "=" * 60)
        print("         CV REVIEW")
        print("=" * 60)
        display_cv_preview(cv_content, contact_info)

        print("\n  Options:")
        print("  [1] Edit a section (provide revision instructions)")
        print("  [2] Regenerate professional summary")
        print("  [3] Accept CV and generate output files")
        print("  [4] Exit without saving")

        choice = get_user_choice("\n  Enter choice (1-4): ",
                                 ["1", "2", "3", "4"])

        if choice == "1":
            cv_content = _edit_section(cv_generator, cv_content)

        elif choice == "2":
            print("\n  Regenerating professional summary...")
            cv_content["professional_summary"] = \
                cv_generator.generate_professional_summary(resume_data, job_data)
            print("  Professional summary updated.")

        elif choice == "3":
            return cv_content

        elif choice == "4":
            print("\n  Exiting without saving.")
            sys.exit(0)


def _edit_section(cv_generator, cv_content):
    """
    Let the user select a CV section and provide revision instructions.
    Sends the revision to Llama 3.2 for rewriting.

    Args:
        cv_generator (CVGenerator): The CV generator instance
        cv_content (dict): Current CV content dictionary

    Returns:
        dict: Updated CV content with the revised section
    """
    # Build list of non-empty sections
    available_sections = []
    for section in cv_generator.CV_SECTIONS:
        if section in cv_content and cv_content[section]:
            available_sections.append(section)

    if not available_sections:
        print("  No sections available to edit.")
        return cv_content

    print("\n  Available sections:")
    valid_choices = []
    for i, section in enumerate(available_sections, 1):
        heading = section.replace("_", " ").title()
        print(f"    [{i}] {heading}")
        valid_choices.append(str(i))

    sec_choice = get_user_choice(
        "\n  Which section to edit? (number): ", valid_choices
    )
    section_name = available_sections[int(sec_choice) - 1]

    # Show current content preview
    current = cv_content[section_name]
    if isinstance(current, (list, dict)):
        import json
        preview = json.dumps(current, indent=2)[:300]
    else:
        preview = str(current)[:300]
    print(f"\n  Current content of '{section_name.replace('_', ' ').title()}':")
    print(f"  {preview}")
    if len(str(current)) > 300:
        print("  ...")

    # Get user feedback
    feedback = input("\n  Your revision instructions: ").strip()
    if not feedback:
        print("  No instructions provided. Section unchanged.")
        return cv_content

    # Send to AI model for revision
    print(f"\n  Revising section...")
    current_text = str(current)
    revised = cv_generator.revise_section(section_name, current_text, feedback)
    cv_content[section_name] = revised
    print("  Section updated successfully.")

    return cv_content


def _resolve_filename(directory, base_name, extension):
    """
    Check whether the target file already exists. If it does, ask the user
    whether to overwrite it or save with a numbered suffix (_2, _3, etc.).

    Args:
        directory (str): Directory where the file will be saved
        base_name (str): Desired filename without extension
        extension (str): File extension including the dot (e.g. '.pdf')

    Returns:
        str: A filename (without extension) to use for saving
    """
    target = os.path.join(directory, f"{base_name}{extension}")

    if not os.path.isfile(target):
        return base_name

    # File exists — ask the user what to do
    print(f"\n  WARNING: '{base_name}{extension}' already exists.")
    print("     [1] Overwrite the existing file")
    print("     [2] Save as a new file")

    choice = get_user_choice("  Enter choice (1-2): ", ["1", "2"])

    if choice == "1":
        print(f"  Overwriting '{base_name}{extension}'.")
        return base_name

    # Find the next available suffix
    candidate = base_name
    counter = 2
    while os.path.isfile(os.path.join(directory, f"{candidate}{extension}")):
        candidate = f"{base_name}_{counter}"
        counter += 1

    print(f"  Saving as '{candidate}{extension}' instead.")
    return candidate


def run_pipeline(hf_token=None):
    """
    Main pipeline function that executes all 5 project steps in sequence.

    Steps:
    1. Initialize LLMs via HuggingFace Transformers (Qwen 2.5 + Llama 3.2)
    2. Acquire and extract resume data (Qwen 2.5 1.5B)
    3. Acquire and parse job description (Qwen 2.5 1.5B)
    4. Generate tailored CV content (Llama 3.2 3B)
    5. Interactive revision loop and final output generation

    Args:
        hf_token (str or None): HuggingFace access token for gated models
    """
    # Display welcome banner
    display_banner()

    import time
    import gc
    pipeline_start = time.time()

    # Flush any stale GPU/MPS memory from previous interrupted runs
    # Flush stale GPU/MPS memory from any previous interrupted run
    gc.collect()
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass  # best-effort; torch may not be installed yet

    # =========================================================================
    # STEP 1: Project Setup — Initialize LLMs
    # =========================================================================
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  STEP 1 - Initializing LLMs             │")
    print("  └─────────────────────────────────────────┘")
    llm_handler = LLMHandler(
        hf_token=hf_token
    )
    llm_handler.verify_connection()
    print(f"  ✓ Step 1 completed.")

    # Initialize all pipeline components
    input_parser = InputParser()
    extractor = ResumeDataExtractor(llm_handler)
    job_parser = JobDescriptionParser(llm_handler)
    cv_generator = CVGenerator(llm_handler)
    output_builder = OutputBuilder(output_dir=".")

    # =========================================================================
    # STEP 2: Resume Data Extraction
    # =========================================================================
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  STEP 2 - Resume Data Extraction        │")
    print("  └─────────────────────────────────────────┘")

    # Get resume input (file or manual)
    raw_resume_text = get_resume_input(input_parser)
    if not raw_resume_text:
        print("  ERROR: No resume data provided. Exiting.")
        sys.exit(1)

    # Extract structured data using Qwen 2.5
    step_start = time.time()
    resume_data = extractor.extract_from_text(raw_resume_text)
    print(f"  ✓ Extraction completed. ({time.time() - step_start:.1f}s)")

    # Let user verify extracted data
    resume_data = verify_extracted_data(extractor, resume_data)

    # Get contact info for output generation
    contact_info = extractor.get_contact_info(resume_data)

    # =========================================================================
    # STEP 3: Job Description Parsing (Optional)
    # =========================================================================
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  STEP 3 - Job Description Parsing       │")
    print("  └─────────────────────────────────────────┘")

    raw_job_text = get_job_description_input(input_parser, resume_text=raw_resume_text)
    job_data = None
    ats_keywords = []

    if raw_job_text:
        # Parse job description using Qwen 2.5
        step_start = time.time()
        job_data = job_parser.parse_job_description(raw_job_text)
        print(f"  ✓ JD parsing completed. ({time.time() - step_start:.1f}s)")
        ats_keywords = job_parser.extract_ats_keywords(job_data)
        print(f"  ✓ Extracted ATS keywords for CV tailoring.")

    # =========================================================================
    # =========================================================================
    # STEP 3b: Structured Interview (enrich data before generation)
    # =========================================================================
    step_start = time.time()
    resume_data = structured_interview_round(extractor, resume_data)
    print(f"  ✓ Structured interview completed. ({time.time() - step_start:.1f}s)")
    # Refresh contact info in case interview updated it
    contact_info = extractor.get_contact_info(resume_data)

    # =========================================================================
    # STEP 3c: JD-Specific Follow-up Questions (only when JD is provided)
    # =========================================================================
    if job_data:
        print("\n  Analyzing your profile against the job requirements...")
        step_start = time.time()
        resume_data = follow_up_round(extractor, resume_data, job_data)
        print(f"  ✓ JD follow-up round completed. ({time.time() - step_start:.1f}s)")
        contact_info = extractor.get_contact_info(resume_data)

    # =========================================================================
    # STEP 4: Resume Tailoring & Generation
    # =========================================================================
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  STEP 4 - CV Content Generation         │")
    print("  └─────────────────────────────────────────┘")

    # Generate tailored CV content using Llama 3.2
    step_start = time.time()
    cv_content = cv_generator.generate_full_cv(resume_data, job_data)
    print(f"  ✓ Step 4 completed. ({time.time() - step_start:.1f}s)")

    # CV preview is available via option [1] in the revision loop

    # =========================================================================
    # STEP 5: User Review & Iterative Revision
    # =========================================================================
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  STEP 5 - Review and Revision           │")
    print("  └─────────────────────────────────────────┘")

    # ATS Compatibility Scoring (if JD provided)
    ats_report = None
    if job_data:
        step_start = time.time()
        ats_report = cv_generator.score_ats_compatibility(
            cv_content, job_data, contact_info
        )
        print(f"  ✓ ATS scoring completed. ({time.time() - step_start:.1f}s)")

    final_cv_content = revision_loop(
        cv_generator, cv_content, contact_info,
        resume_data, job_data, ats_report
    )

    # =========================================================================
    # OUTPUT: Ask format preference and generate files
    # =========================================================================
    print("\n  What output format would you like?")
    print("  [1] PDF only")
    print("  [2] DOCX only")
    print("  [3] Both PDF and DOCX")

    format_choice = get_user_choice("\n  Enter choice (1-3): ", ["1", "2", "3"])

    # Build filename from candidate name
    candidate_name = contact_info.get("name", "").strip()
    if candidate_name:
        # Convert "John Doe" -> "John_Doe_CV"
        base_name = candidate_name.replace(" ", "_") + "_CV"
    else:
        base_name = "generated_cv"

    # Show default filename and let user customize
    output_dir = os.path.abspath(output_builder.output_dir)
    print(f"\n  Output directory: {output_dir}")
    print(f"  Default filename: {base_name}")
    print(f"\n  Would you like to use a custom filename?")
    print(f"  [1] Use default ({base_name})")
    print(f"  [2] Enter a custom filename")

    name_choice = get_user_choice("\n  Enter choice (1-2): ", ["1", "2"])

    if name_choice == "2":
        custom_name = input("  Enter filename (without extension): ").strip()
        if custom_name:
            # Sanitize: replace spaces with underscores, remove unsafe chars
            base_name = custom_name.replace(" ", "_")
            base_name = "".join(c for c in base_name if c.isalnum() or c in "_-")
            print(f"  Using filename: {base_name}")
        else:
            print(f"  No name entered. Using default: {base_name}")

    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  Generating Output Files                │")
    print("  └─────────────────────────────────────────┘")

    generated_files = []

    # Generate DOCX first: if PDF (fpdf2) fails, user still gets the DOCX
    if format_choice in ("2", "3"):
        docx_filename = _resolve_filename(output_builder.output_dir, base_name, ".docx")
        docx_path = output_builder.build_docx(
            final_cv_content, contact_info, docx_filename
        )
        print(f"  ✓ DOCX saved: {docx_path}")
        generated_files.append(docx_path)

    if format_choice in ("1", "3"):
        pdf_filename = _resolve_filename(output_builder.output_dir, base_name, ".pdf")
        pdf_path = output_builder.build_pdf(
            final_cv_content, contact_info, pdf_filename
        )
        print(f"  ✓ PDF saved:  {pdf_path}")
        generated_files.append(pdf_path)

    # =========================================================================
    # COMPLETION
    # =========================================================================
    total_time = time.time() - pipeline_start
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    ext_model = 'Qwen/Qwen2.5-1.5B-Instruct'
    gen_model = 'meta-llama/Llama-3.2-3B-Instruct'

    print(f"\n  {'=' * 60}")
    print(f"  CV GENERATION COMPLETE!")
    print(f"  {'=' * 60}")
    print(f"\n  Output files:")
    for fpath in generated_files:
        print(f"    - {fpath}")
    print(f"\n  Models used:")
    print(f"    - Extraction: {ext_model}")
    print(f"    - Generation: {gen_model}")
    print(f"\n  Total time: {minutes}m {seconds}s")
    print(f"\n  {'=' * 60}")


if __name__ == "__main__":
    args = parse_arguments()
    try:
        while True:
            run_pipeline(hf_token=args.hf_token)
            # After CV generation, ask the user if they want to create another
            print("\n  What would you like to do next?")
            print("  [1] Create another CV")
            print("  [2] Exit")
            next_choice = get_user_choice("\n  Enter choice (1-2): ", ["1", "2"])
            if next_choice == "2":
                print("\n  Thank you for using CV Creation using LLMs! Goodbye.\n")
                sys.exit(0)
            # Loop back to create another CV
            print("\n  Starting a new CV generation...\n")
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user. Exiting.")
        sys.exit(0)
    except ModuleNotFoundError:
        raise  # Let missing dependencies show the default Python traceback
    except Exception as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)
