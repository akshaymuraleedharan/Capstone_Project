"""
job_parser.py - Uses Qwen 2.5 (extraction LLM) to parse job descriptions
into structured JSON with requirements, keywords, and responsibilities.

This module implements Step 3 of the pipeline: Job Description Parsing.
It extracts structured information from raw job posting text, including
required/preferred skills, responsibilities, and ATS-relevant keywords.
"""

import json
from llm_handler import LLMHandler
from prompts import JOB_PARSING_PROMPT
from utils import clean_json_response


class JobDescriptionParser:
    """
    Parses job descriptions into structured data using Qwen 2.5.
    Extracts keywords, requirements, and responsibilities for CV tailoring.
    """

    # Standard schema for parsed job description data
    JOB_SCHEMA = {
        "job_title": "",
        "company": "",
        "required_skills": [],
        "preferred_skills": [],
        "experience_requirements": "",
        "education_requirements": "",
        "key_responsibilities": [],
        "keywords": [],
        "industry": "",
        "seniority_level": ""
    }

    def __init__(self, llm_handler):
        """
        Initialize the parser with an LLM handler instance.

        Args:
            llm_handler (LLMHandler): Configured LLM handler with extraction model
        """
        self.llm = llm_handler

    def parse_job_description(self, raw_text):
        """
        Send raw job description text to Qwen 2.5 for structured parsing.
        Parses the LLM response into JSON and fills missing fields with defaults.

        Args:
            raw_text (str): Raw job description text (copy-pasted or from file)

        Returns:
            dict: Structured job data matching JOB_SCHEMA

        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON
            RuntimeError: If the LLM call fails after all retries
        """
        print("\n  Parsing job description using Qwen 2.5...")

        # Build the prompt with the job description text
        prompt = JOB_PARSING_PROMPT.format(job_text=raw_text)

        # Call the extraction model (Qwen 2.5, low temperature)
        response = self.llm.extract(prompt)

        # Parse JSON from the response
        try:
            parsed_data = clean_json_response(response)
        except ValueError as e:
            print(f"  Warning: Could not parse job parsing response as JSON: {e}")
            print("  Retrying parsing...")
            response = self.llm.extract(prompt)
            parsed_data = clean_json_response(response)

        # Fill missing fields with defaults from schema
        for key, default_value in self.JOB_SCHEMA.items():
            if key not in parsed_data:
                parsed_data[key] = default_value

        print("  Job description parsed successfully.")
        return parsed_data

    def extract_ats_keywords(self, job_data):
        """
        Derive a flat, deduplicated list of ATS-critical keywords from parsed job data.
        Combines required_skills, preferred_skills, keywords, and key terms from
        responsibilities into a single list sorted by importance.

        Args:
            job_data (dict): Parsed job description dictionary

        Returns:
            list: Deduplicated list of ATS keywords (required skills listed first)
        """
        keywords = []

        # Required skills are highest priority
        required = job_data.get("required_skills", [])
        for skill in required:
            if skill and skill not in keywords:
                keywords.append(skill)

        # Explicit keywords from the job posting
        explicit_keywords = job_data.get("keywords", [])
        for kw in explicit_keywords:
            if kw and kw not in keywords:
                keywords.append(kw)

        # Preferred skills are next priority
        preferred = job_data.get("preferred_skills", [])
        for skill in preferred:
            if skill and skill not in keywords:
                keywords.append(skill)

        return keywords

    def display_parsed_job(self, job_data):
        """
        Pretty-print parsed job description to terminal for user review.
        Shows all extracted fields in a readable format.

        Args:
            job_data (dict): Structured job description JSON
        """
        print("\n" + "=" * 60)
        print("         PARSED JOB DESCRIPTION")
        print("=" * 60)

        print(f"\n  Job Title: {job_data.get('job_title', 'N/A')}")
        print(f"  Company: {job_data.get('company', 'N/A')}")
        print(f"  Industry: {job_data.get('industry', 'N/A')}")
        print(f"  Seniority: {job_data.get('seniority_level', 'N/A')}")

        exp_req = job_data.get("experience_requirements", "")
        if exp_req:
            print(f"  Experience Required: {exp_req}")

        edu_req = job_data.get("education_requirements", "")
        if edu_req:
            print(f"  Education Required: {edu_req}")

        required = job_data.get("required_skills", [])
        if required:
            print(f"\n  Required Skills ({len(required)}):")
            print(f"  {', '.join(required)}")

        preferred = job_data.get("preferred_skills", [])
        if preferred:
            print(f"\n  Preferred Skills ({len(preferred)}):")
            print(f"  {', '.join(preferred)}")

        responsibilities = job_data.get("key_responsibilities", [])
        if responsibilities:
            print(f"\n  Key Responsibilities ({len(responsibilities)}):")
            for i, resp in enumerate(responsibilities, 1):
                print(f"    {i}. {resp}")

        print(f"\n{'=' * 60}")
