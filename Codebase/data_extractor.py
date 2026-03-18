"""
data_extractor.py - Uses Qwen 2.5 (extraction LLM) to parse raw resume text
into structured JSON with standardized fields.

This module implements Step 2 of the pipeline: Resume Data Extraction.
It takes unstructured resume text from any input source and produces
a clean, validated JSON dictionary that can be used for CV generation.
"""

import json
from llm_handler import LLMHandler
from prompts import RESUME_EXTRACTION_PROMPT, FOLLOW_UP_PROMPT_JD, FOLLOW_UP_PROMPT_GENERAL
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

    def extract_from_text(self, raw_text):
        """
        Send raw resume text to Qwen 2.5 for structured extraction.
        Parses the LLM response into JSON, validates against the schema,
        and fills in missing fields with defaults.

        Args:
            raw_text (str): Unstructured resume text from any input source

        Returns:
            dict: Structured resume data matching RESUME_SCHEMA

        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON
            RuntimeError: If the LLM call fails after all retries
        """
        print("\n  Extracting resume data using Qwen 2.5...")

        # Build the prompt with the raw resume text
        prompt = RESUME_EXTRACTION_PROMPT.format(resume_text=raw_text)

        # Call the extraction model (Qwen 2.5, low temperature)
        response = self.llm.extract(prompt)

        # Parse JSON from the response
        try:
            extracted_data = clean_json_response(response)
        except ValueError as e:
            print(f"  Warning: Could not parse extraction response as JSON: {e}")
            print("  Retrying extraction (attempt 2)...")
            # Retry once — the second attempt often completes within the token limit
            response = self.llm.extract(prompt)
            try:
                extracted_data = clean_json_response(response)
            except ValueError as e2:
                print(f"  Warning: Second attempt also failed: {e2}")
                raise

        # Validate and fill in missing fields
        validated_data = validate_resume_json(extracted_data, self.RESUME_SCHEMA)

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

        # Infer soft skills from experience descriptions if none were extracted
        validated_data = self._infer_soft_skills(validated_data)

        # Infer implied technical skills from experience/project descriptions
        validated_data = self._infer_technical_skills(validated_data)

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

        # Name and contact
        print(f"\n  Name: {resume_data.get('name', 'N/A')}")
        contact = resume_data.get("contact", {})
        for field in ["email", "phone", "location", "linkedin", "portfolio"]:
            val = contact.get(field, "")
            if val:
                print(f"  {field.title()}: {val}")

        # Professional summary
        summary = resume_data.get("professional_summary", "")
        if summary:
            print(f"\n  Professional Summary:")
            print(f"  {summary}")

        # Education
        education = resume_data.get("education", [])
        if education:
            print(f"\n  Education ({len(education)} entries):")
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

        # Experience
        experience = resume_data.get("experience", [])
        if experience:
            print(f"\n  Work Experience ({len(experience)} entries):")
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

        # Skills
        skills = resume_data.get("skills", {})
        if skills:
            print(f"\n  Skills:")
            for category, skill_list in skills.items():
                if skill_list:
                    print(f"    {category.title()}: {', '.join(str(s) for s in skill_list)}")

        # Projects
        projects = resume_data.get("projects", [])
        if projects:
            print(f"\n  Projects ({len(projects)} entries):")
            for proj in projects:
                if isinstance(proj, dict):
                    print(f"    • {proj.get('name', 'N/A')}: {proj.get('description', '')}")
                else:
                    print(f"    • {proj}")

        # Certifications
        certs = resume_data.get("certifications", [])
        if certs:
            print(f"\n  Certifications:")
            for cert in certs:
                if isinstance(cert, dict):
                    print(f"    • {cert.get('name', 'N/A')} ({cert.get('issuer', '')} {cert.get('year', '')})")
                else:
                    print(f"    • {cert}")

        # Achievements
        achievements = resume_data.get("achievements", [])
        if achievements:
            print(f"\n  Achievements:")
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
            # Remove list items that are just placeholder strings
            return [
                item for item in cleaned
                if not (isinstance(item, str) and item == "")
            ]
        elif isinstance(data, str):
            return cls._clean_placeholder(data)
        return data

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
            (r'\br\b(?:\s+programming|\s+studio|\s+language)', "R", "technical"),
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
        print("\n  Analyzing your profile for gaps...")

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

        # Skills
        skills = resume_data.get("skills", {})
        if isinstance(skills, dict):
            tech_count = len(skills.get("technical", []))
            soft_count = len(skills.get("soft", []))
            tools_count = len(skills.get("tools", []))
            if tech_count + soft_count + tools_count > 0:
                parts.append(f"skills: {tech_count} technical, {soft_count} soft, {tools_count} tools")
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
        if summary and len(summary) > 20:
            parts.append(f"professional_summary: present ({len(summary)} chars)")
        else:
            parts.append("professional_summary: empty or thin")

        # Achievements
        achievements = resume_data.get("achievements", [])
        if achievements:
            parts.append(f"achievements: {len(achievements)} entries")
        else:
            parts.append("achievements: empty")

        # Certifications
        certs = resume_data.get("certifications", [])
        if certs:
            parts.append(f"certifications: {len(certs)} entries")
        else:
            parts.append("certifications: empty")

        return "\n".join(parts)

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
                    "name": answer.split(".")[0].split(",")[0][:60],
                    "description": answer,
                    "technologies": []
                })
                resume_data["projects"] = projects

            elif section == "experience":
                experience = resume_data.get("experience", [])
                experience.append({
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

        return resume_data
