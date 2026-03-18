"""
main.py - Entry point for the CV Creation using LLMs pipeline.
Capstone Project CS01.

This script orchestrates the complete 5-step pipeline:
1. Project Setup - Initialize LLMs and verify connectivity
2. Resume Data Extraction - Parse input into structured JSON (Qwen 2.5 3B)
3. Job Description Parsing - Extract requirements and keywords (Qwen 2.5 3B)
4. Resume Tailoring & Generation - Create ATS-optimized CV content (Llama 3.2 3B)
5. User Review & Iterative Revision - Interactive editing and ATS feedback

Models Used (via HuggingFace Transformers - platform agnostic, no server required):
- Qwen 2.5 3B (Qwen/Qwen2.5-3B-Instruct): Extraction and analysis tasks
- Llama 3.2 3B (meta-llama/Llama-3.2-3B-Instruct): Content generation and rewriting tasks

Usage:
    python main.py
    python main.py --hf-token YOUR_HF_TOKEN
    python main.py --resume path/to/resume.pdf --job path/to/job.txt --hf-token YOUR_TOKEN
    python main.py --resume resume.pdf --job job.txt --output my_cv
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
    Automatically install all required Python packages from requirements.txt
    if they are not already present. This ensures the evaluator does not need
    to manually run 'pip install -r requirements.txt' before running the project.

    Reads requirements.txt from the same directory as main.py.
    Exits with an error if installation fails.
    """
    # Path to requirements.txt (same directory as this script)
    req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")

    if not os.path.isfile(req_file):
        print("  WARNING: requirements.txt not found. Skipping auto-install.")
        return

    print("  Checking Python dependencies...")

    # Read required packages
    with open(req_file, "r") as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not packages:
        return

    # Check which packages are missing
    missing = []
    for pkg in packages:
        # Extract package name without version specifier for import check
        pkg_name = pkg.split(">=")[0].split("==")[0].split("!=")[0].strip()
        # Map pip package names to importable module names
        import_name_map = {
            "python-docx": "docx",
            "fpdf2": "fpdf",
            "pdfplumber": "pdfplumber",
            "transformers": "transformers",
            "torch": "torch",
            "accelerate": "accelerate"
        }
        import_name = import_name_map.get(pkg_name, pkg_name.replace("-", "_"))
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if not missing:
        print("  All dependencies are already installed.")
        return

    print(f"  Installing missing packages: {', '.join(missing)}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + missing,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        print("  Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n  ERROR: Failed to install dependencies automatically.")
        print(f"  Please run manually: pip install -r requirements.txt")
        sys.exit(1)


# =============================================================================
# Corporate / proxy SSL fix — MUST run before any import of 'transformers'.
#
# transformers 5.x reads HF_HUB_OFFLINE at import time, not at call time.
# check_and_install_dependencies() does __import__("transformers") to verify
# the package is installed, so the env var must be set before that call too.
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

    Checks both default models and any custom models passed via CLI args.
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

    # Start with the default model IDs
    extraction_model = "Qwen/Qwen2.5-3B-Instruct"
    generation_model = "meta-llama/Llama-3.2-3B-Instruct"

    # Check CLI args for custom models (before argparse runs)
    for i, arg in enumerate(sys.argv[:-1]):
        if arg == "--gemma-model":
            extraction_model = sys.argv[i + 1]
        elif arg == "--llama-model":
            generation_model = sys.argv[i + 1]

    models_to_check = [extraction_model, generation_model]
    all_cached = all(
        os.path.isdir(
            os.path.join(cache_dir, "models--" + m.replace("/", "--"))
        )
        for m in models_to_check
    )
    if all_cached:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        # Suppress progress bars when loading from cache (they're just noise).
        # Must be set before tqdm is first imported by transformers.
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

_set_hf_offline_if_cached()


# =============================================================================
# Auto-install dependencies BEFORE importing local modules
# (local modules import transformers/torch at load time, so they must exist first)
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
    Parse command line arguments for optional file paths, model configuration,
    and HuggingFace authentication token.

    Returns:
        argparse.Namespace: Parsed arguments with resume, job, output,
                            gemma_model, llama_model, and hf_token fields
    """
    parser = argparse.ArgumentParser(
        description="CV Creation using LLMs - Automated resume builder "
                    "(Capstone Project CS01)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to existing resume file (PDF, DOCX, or TXT)"
    )
    parser.add_argument(
        "--job", type=str, default=None,
        help="Path to job description file (PDF, DOCX, or TXT)"
    )
    parser.add_argument(
        "--output", type=str, default="generated_cv",
        help="Output filename without extension (default: generated_cv)"
    )
    parser.add_argument(
        "--gemma-model", type=str, default=None,
        help="HuggingFace model ID for extraction tasks "
             "(default: Qwen/Qwen2.5-3B-Instruct)"
    )
    parser.add_argument(
        "--llama-model", type=str, default=None,
        help="HuggingFace model ID for generation tasks "
             "(default: meta-llama/Llama-3.2-3B-Instruct)"
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace access token for gated models (Llama 3.2). "
             "Can also be set via the HF_TOKEN environment variable. "
             "Get a free token at: https://huggingface.co/settings/tokens"
    )
    return parser.parse_args()


def get_resume_input(input_parser, args):
    """
    Acquire resume data from a file or manual input.
    If --resume arg is provided, reads that file directly.
    Otherwise, presents an interactive menu for the user to choose input method.

    Args:
        input_parser (InputParser): Configured input parser instance
        args (argparse.Namespace): Parsed CLI arguments

    Returns:
        str: Raw resume text content
    """
    # If file path provided via CLI arg, use it directly
    if args.resume:
        print(f"\n  Loading resume from: {args.resume}")
        return input_parser.read_file(args.resume)

    # Interactive menu
    print("\n  How would you like to provide your resume data?")
    print("  [1] Load from PDF file")
    print("  [2] Load from DOCX file")
    print("  [3] Load from text file")
    print("  [4] Enter manually (free-form text)")
    print("  [5] Enter manually (guided section-by-section)")

    choice = get_user_choice("\n  Enter choice (1-5): ", ["1", "2", "3", "4", "5"])

    if choice in ("1", "2", "3"):
        file_path = input("  Enter file path: ").strip()
        # Remove surrounding quotes if present
        file_path = file_path.strip("'\"")
        return input_parser.read_file(file_path)
    elif choice == "4":
        return input_parser.read_manual_input()
    elif choice == "5":
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
    return _text_similarity(raw_text, resume_text) > 0.5


def get_job_description_input(input_parser, args, resume_text=None):
    """
    Acquire job description from a file or manual input.
    If --job arg is provided, reads that file directly.
    Otherwise, presents an interactive menu. User can skip this step.

    Validates input against the already-provided resume text to catch
    accidental pastes (e.g. pasting the CV instead of a JD).

    Args:
        input_parser (InputParser): Configured input parser instance
        args (argparse.Namespace): Parsed CLI arguments
        resume_text (str or None): Raw resume text from Step 2, used to
                                    detect accidental duplicate pastes

    Returns:
        str or None: Raw job description text, or None if skipped
    """
    # If file path provided via CLI arg, use it directly
    if args.job:
        print(f"\n  Loading job description from: {args.job}")
        text = input_parser.read_file(args.job)
        if text and _validate_jd_input(text, resume_text):
            print("\n  WARNING: This file looks very similar to your resume.")
            print("     Did you provide the same file for both?")
            print("     [1] Use it anyway")
            print("     [2] Skip job description")
            retry = get_user_choice("\n  Enter choice (1-2): ", ["1", "2"])
            if retry == "2":
                print("  Skipping job description. Will generate a general-purpose CV.")
                return None
        return text

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
            file_path = input("  Enter file path: ").strip()
            file_path = file_path.strip("'\"")
            raw_text = input_parser.read_file(file_path)
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


def verify_extracted_data(extractor, resume_data):
    """
    Display extracted resume data and let the user verify or correct it.

    Args:
        extractor (ResumeDataExtractor): The data extractor instance
        resume_data (dict): Extracted resume data JSON

    Returns:
        dict: Verified (and possibly corrected) resume data
    """
    extractor.display_extracted_data(resume_data)

    print("\n  Is the extracted information correct?")
    print("  [1] Yes, continue")
    print("  [2] No, let me re-enter my resume data")

    choice = get_user_choice("\n  Enter choice (1-2): ", ["1", "2"])

    if choice == "2":
        print("\n  Please re-enter your resume information.")
        parser = InputParser()
        raw_text = parser.read_manual_structured()
        resume_data = extractor.extract_from_text(raw_text)
        extractor.display_extracted_data(resume_data)

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
    follow_ups = extractor.generate_follow_ups(resume_data, job_data)

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
        resume_data = extractor.merge_follow_up_answers(resume_data, answered)
        print(f"  Profile updated with {len(answered)} additional detail(s).")
    else:
        print("  No additional details provided. Continuing with current data.")

    return resume_data


def revision_loop(cv_generator, cv_content, contact_info, resume_data,
                  job_data, ats_report):
    """
    Interactive terminal loop for user review and iterative CV revision.
    Implements Step 5 of the pipeline: User Review and Iterative Revision.

    Allows the user to:
    - View the full CV preview
    - Edit specific sections with natural language instructions
    - Auto-optimize for missing ATS keywords
    - Regenerate the professional summary
    - Re-score ATS compatibility
    - Accept and generate final output files

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
    while True:
        print("\n" + "=" * 60)
        print("         CV REVIEW AND REVISION")
        print("=" * 60)

        # Show ATS score summary if available
        if ats_report:
            score = ats_report.get("ats_score", "N/A")
            print(f"\n  ATS Compatibility Score: {score}/100")
            rubric = ats_report.get("rubric")
            if rubric:
                print(f"    Keywords: {rubric.get('keyword_match', 0)}/25 | "
                      f"Completeness: {rubric.get('completeness', 0)}/25 | "
                      f"Impact: {rubric.get('impact_quality', 0)}/25 | "
                      f"Alignment: {rubric.get('role_alignment', 0)}/25")
            missing = ats_report.get("missing_keywords", [])
            if missing:
                print(f"  Missing Keywords: {', '.join(missing[:5])}")
                if len(missing) > 5:
                    print(f"    ... and {len(missing) - 5} more")

        print("\n  Options:")
        print("  [1] View full CV preview")
        print("  [2] Edit a section (provide revision instructions)")
        print("  [3] Auto-optimize for missing ATS keywords")
        print("  [4] Regenerate professional summary")
        print("  [5] Re-score ATS compatibility")
        print("  [6] Accept CV and generate output files")
        print("  [7] Exit without saving")

        choice = get_user_choice("\n  Enter choice (1-7): ",
                                 ["1", "2", "3", "4", "5", "6", "7"])

        if choice == "1":
            # View full CV preview
            display_cv_preview(cv_content, contact_info)

        elif choice == "2":
            # Edit a specific section
            cv_content = _edit_section(cv_generator, cv_content)

        elif choice == "3":
            # Auto-optimize for missing keywords
            if ats_report and ats_report.get("missing_keywords"):
                cv_content = cv_generator.revise_for_keywords(
                    cv_content, ats_report["missing_keywords"]
                )
                print("  CV updated. Consider re-scoring ATS (option 5).")
            else:
                print("  No missing keywords identified. Nothing to optimize.")

        elif choice == "4":
            # Regenerate professional summary
            print("\n  Regenerating professional summary...")
            cv_content["professional_summary"] = \
                cv_generator.generate_professional_summary(resume_data, job_data)
            print("  Professional summary updated.")

        elif choice == "5":
            # Re-score ATS compatibility
            if job_data:
                ats_report = cv_generator.score_ats_compatibility(
                    cv_content, job_data, contact_info
                )
                display_ats_report(ats_report)
            else:
                print("  No job description provided. Cannot score ATS compatibility.")

        elif choice == "6":
            # Accept and generate output
            return cv_content

        elif choice == "7":
            # Exit without saving
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


def run_pipeline(args):
    """
    Main pipeline function that executes all 5 project steps in sequence.

    Steps:
    1. Initialize LLMs via HuggingFace Transformers (Qwen 2.5 + Llama 3.2)
    2. Acquire and extract resume data (Qwen 2.5 3B)
    3. Acquire and parse job description (Qwen 2.5 3B)
    4. Generate tailored CV content (Llama 3.2 3B)
    5. Interactive revision loop and final output generation

    Args:
        args (argparse.Namespace): Parsed CLI arguments
    """
    # Display welcome banner
    display_banner()

    import time
    import gc
    pipeline_start = time.time()

    # Flush any stale GPU/MPS memory from previous interrupted runs
    gc.collect()
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # =========================================================================
    # STEP 1: Project Setup — Initialize LLMs
    # =========================================================================
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  STEP 1 - Initializing LLMs              │")
    print("  └─────────────────────────────────────────┘")
    llm_handler = LLMHandler(
        extraction_model=args.gemma_model,
        generation_model=args.llama_model,
        hf_token=args.hf_token
    )
    llm_handler.verify_connection()
    print(f"  [OK] Step 1 completed.")

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
    print("  │  STEP 2 - Resume Data Extraction         │")
    print("  └─────────────────────────────────────────┘")

    # Get resume input (file or manual)
    raw_resume_text = get_resume_input(input_parser, args)
    if not raw_resume_text:
        print("  ERROR: No resume data provided. Exiting.")
        sys.exit(1)

    # Extract structured data using Qwen 2.5
    step_start = time.time()
    resume_data = extractor.extract_from_text(raw_resume_text)
    print(f"  [OK] Extraction completed. ({time.time() - step_start:.1f}s)")

    # Let user verify extracted data
    resume_data = verify_extracted_data(extractor, resume_data)

    # Get contact info for output generation
    contact_info = extractor.get_contact_info(resume_data)

    # =========================================================================
    # STEP 3: Job Description Parsing (Optional)
    # =========================================================================
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  STEP 3 - Job Description Parsing         │")
    print("  └─────────────────────────────────────────┘")

    raw_job_text = get_job_description_input(input_parser, args, resume_text=raw_resume_text)
    job_data = None
    ats_keywords = []

    if raw_job_text:
        # Parse job description using Qwen 2.5
        step_start = time.time()
        job_data = job_parser.parse_job_description(raw_job_text)
        print(f"  [OK] JD parsing completed. ({time.time() - step_start:.1f}s)")
        ats_keywords = job_parser.extract_ats_keywords(job_data)
        print(f"  [OK] Extracted ATS keywords for CV tailoring.")

    # =========================================================================
    # STEP 3b: Smart Follow-up Questions (after JD so we know the gaps)
    # =========================================================================
    if job_data:
        print("\n  Analyzing your profile against the job requirements...")
    else:
        print("\n  Analyzing your profile for gaps to strengthen your CV...")

    step_start = time.time()
    resume_data = follow_up_round(extractor, resume_data, job_data)
    print(f"  [OK] Follow-up round completed. ({time.time() - step_start:.1f}s)")
    # Refresh contact info in case follow-ups updated it
    contact_info = extractor.get_contact_info(resume_data)

    # =========================================================================
    # STEP 4: Resume Tailoring & Generation
    # =========================================================================
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  STEP 4 - CV Content Generation           │")
    print("  └─────────────────────────────────────────┘")

    # Generate tailored CV content using Llama 3.2
    step_start = time.time()
    cv_content = cv_generator.generate_full_cv(resume_data, job_data)
    print(f"  [OK] Step 4 completed. ({time.time() - step_start:.1f}s)")

    # Display generated CV preview
    display_cv_preview(cv_content, contact_info)

    # =========================================================================
    # STEP 4b: ATS Compatibility Scoring (if JD provided)
    # =========================================================================
    ats_report = None
    if job_data:
        step_start = time.time()
        ats_report = cv_generator.score_ats_compatibility(
            cv_content, job_data, contact_info
        )
        print(f"  [OK] ATS scoring completed. ({time.time() - step_start:.1f}s)")
        display_ats_report(ats_report)

    # =========================================================================
    # STEP 5: User Review & Iterative Revision
    # =========================================================================
    print("\n  ┌─────────────────────────────────────────┐")
    print("  │  STEP 5 - Review and Revision             │")
    print("  └─────────────────────────────────────────┘")
    print("  " + "-" * 40)

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
        base_name = args.output

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
    print("  │  Generating Output Files                 │")
    print("  └─────────────────────────────────────────┘")

    generated_files = []

    if format_choice in ("2", "3"):
        docx_filename = _resolve_filename(output_builder.output_dir, base_name, ".docx")
        docx_path = output_builder.build_docx(
            final_cv_content, contact_info, docx_filename
        )
        print(f"  [OK] DOCX saved: {docx_path}")
        generated_files.append(docx_path)

    if format_choice in ("1", "3"):
        pdf_filename = _resolve_filename(output_builder.output_dir, base_name, ".pdf")
        pdf_path = output_builder.build_pdf(
            final_cv_content, contact_info, pdf_filename
        )
        print(f"  [OK] PDF saved:  {pdf_path}")
        generated_files.append(pdf_path)

    # =========================================================================
    # COMPLETION
    # =========================================================================
    total_time = time.time() - pipeline_start
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    ext_model = args.gemma_model or 'Qwen/Qwen2.5-1.5B-Instruct'
    gen_model = args.llama_model or 'meta-llama/Llama-3.2-3B-Instruct'

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
    print(f"\n  Thank you for using CV Creation using LLMs!\n")


if __name__ == "__main__":
    args = parse_arguments()
    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)
