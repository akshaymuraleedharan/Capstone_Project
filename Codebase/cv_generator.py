"""
cv_generator.py - Uses Llama 3.2 (generation LLM) to create tailored,
ATS-optimized CV content by aligning candidate data with job requirements.

This module implements Step 4 (Resume Tailoring and Generation) and
Step 5 (User Review and Iterative Revision) of the pipeline.

It also uses Qwen 2.5 (via the same LLMHandler) for ATS scoring,
since scoring is an analysis task best suited for the extraction model.
"""

import json
from llm_handler import LLMHandler
from prompts import (
    CV_GENERATION_PROMPT,
    SECTION_REWRITE_PROMPT,
    ATS_SCORING_PROMPT,
    REVISION_PROMPT,
    PROFESSIONAL_SUMMARY_PROMPT,
    EXPERIENCE_TAILORING_PROMPT,
    COMBINED_CV_GENERATION_PROMPT
)
from utils import clean_json_response, format_cv_content_as_text, strip_llm_commentary, parse_combined_cv_response, deduplicate_content


class CVGenerator:
    """
    Generates tailored CV content using Llama 3.2 by combining
    extracted resume data with parsed job description requirements.
    Also provides ATS scoring using Qwen 2.5.
    """

    # Sections in the final CV and their processing order
    CV_SECTIONS = [
        "professional_summary",
        "experience",
        "education",
        "skills",
        "projects",
        "certifications",
        "achievements"
    ]

    def __init__(self, llm_handler):
        """
        Initialize the generator with an LLM handler instance.

        Args:
            llm_handler (LLMHandler): Configured LLM handler with both
                                       extraction and generation models
        """
        self.llm = llm_handler

    def generate_full_cv(self, resume_data, job_data=None):
        """
        Generate complete CV content, optionally tailored to a job description.
        Uses a single combined LLM call for summary, experience, and projects
        to minimize inference time, then handles skills and pass-through sections.

        Args:
            resume_data (dict): Structured resume data from extraction step
            job_data (dict or None): Parsed job description (None if no JD provided)

        Returns:
            dict: Generated CV content organized by section name
        """
        print("\n  Generating CV content using Llama 3.2...")
        cv_content = {}

        # Build job context string for prompts (empty if no JD)
        job_context = self._build_job_context(job_data)

        # --- Prepare data for combined prompt ---
        name = resume_data.get("name", "the candidate")

        # Experience (filtered — small models hallucinate placeholder entries like "N/A")
        experience = resume_data.get("experience", [])
        real_experience = [e for e in experience if self._is_real_experience(e)]
        experience_text = self._format_experience_for_prompt(real_experience) if real_experience else "No work experience."

        # Most recent role
        recent_role = "N/A"
        if real_experience:
            first_exp = real_experience[0]
            if isinstance(first_exp, dict):
                recent_role = f"{first_exp.get('title', '')} at {first_exp.get('company', '')}"
            else:
                recent_role = str(first_exp)

        # Years of experience
        years_experience = self._estimate_years(experience, resume_data)

        # Skills
        skills = resume_data.get("skills", {})
        all_skills = []
        if isinstance(skills, dict):
            for category_skills in skills.values():
                if isinstance(category_skills, list):
                    all_skills.extend(category_skills)
        key_skills = ", ".join(all_skills[:10]) if all_skills else "N/A"  # cap to stay within small-model context

        # Education
        education = resume_data.get("education", [])
        edu_summary = "N/A"
        if education:
            first_edu = education[0]
            if isinstance(first_edu, dict):
                edu_summary = f"{first_edu.get('degree', '')} from {first_edu.get('institution', '')}"
            else:
                edu_summary = str(first_edu)

        # Projects
        projects = resume_data.get("projects", [])
        projects_text = self._format_projects_for_prompt(projects) if projects else "No projects."

        # --- Single combined LLM call for summary + experience + projects ---
        print("    - Generating professional summary, experience, and projects...")
        prompt = COMBINED_CV_GENERATION_PROMPT.format(
            name=name,
            recent_role=recent_role,
            years_experience=years_experience,
            key_skills=key_skills,
            education=edu_summary,
            experience_text=experience_text,
            projects_text=projects_text,
            job_context=job_context if job_context else "No specific job target. Write a general-purpose, impactful CV."
        )

        response = self.llm.generate(prompt)
        parsed = parse_combined_cv_response(response)

        # Apply each parsed section (with cleanup and deduplication)
        cv_content["professional_summary"] = strip_llm_commentary(
            deduplicate_content(parsed.get("professional_summary", "").strip())
        )

        if real_experience and parsed.get("experience"):
            cv_content["experience"] = strip_llm_commentary(
                deduplicate_content(parsed["experience"].strip())
            )
        else:
            cv_content["experience"] = ""

        if projects and parsed.get("projects"):
            cv_content["projects"] = strip_llm_commentary(
                deduplicate_content(parsed["projects"].strip())
            )
        else:
            cv_content["projects"] = []

        # --- Education — pass through ---
        cv_content["education"] = education

        # --- Skills — optimize if JD provided (separate call, returns JSON) ---
        print("    - Processing skills...")
        if job_data and skills:
            cv_content["skills"] = self.optimize_skills_section(skills, job_data)
        else:
            cv_content["skills"] = skills

        # --- Pass-through sections ---
        cv_content["certifications"] = resume_data.get("certifications", [])
        cv_content["achievements"] = resume_data.get("achievements", [])

        print("  ✓ CV content generated successfully.")
        return cv_content

    def generate_professional_summary(self, resume_data, job_data=None):
        """
        Generate a compelling professional summary/objective statement
        using Llama 3.2.

        Args:
            resume_data (dict): Full resume data dictionary
            job_data (dict or None): Job description data for targeting

        Returns:
            str: Professional summary paragraph (3-4 sentences)
        """
        # Extract relevant info for the prompt
        name = resume_data.get("name", "the candidate")

        # Determine most recent role (only from real experience)
        experience = resume_data.get("experience", [])
        real_experience = [e for e in experience if self._is_real_experience(e)]
        recent_role = "N/A"
        if real_experience:
            first_exp = real_experience[0]
            if isinstance(first_exp, dict):
                recent_role = f"{first_exp.get('title', '')} at {first_exp.get('company', '')}"
            else:
                recent_role = str(first_exp)

        # Estimate years of experience (uses user-provided value if available)
        years_experience = self._estimate_years(experience, resume_data)

        # Collect key skills
        skills = resume_data.get("skills", {})
        all_skills = []
        if isinstance(skills, dict):
            for category_skills in skills.values():
                if isinstance(category_skills, list):
                    all_skills.extend(category_skills)
        key_skills = ", ".join(all_skills[:10]) if all_skills else "N/A"  # cap to stay within small-model context

        # Education summary
        education = resume_data.get("education", [])
        edu_summary = "N/A"
        if education:
            first_edu = education[0]
            if isinstance(first_edu, dict):
                edu_summary = f"{first_edu.get('degree', '')} from {first_edu.get('institution', '')}"
            else:
                edu_summary = str(first_edu)

        # Build job context if JD available
        job_context = ""
        if job_data:
            job_title = job_data.get("job_title", "")
            company = job_data.get("company", "")
            required = job_data.get("required_skills", [])
            job_context = (
                f"Target Job: {job_title} at {company}\n"
                f"Key requirements: {', '.join(required[:8])}\n"
                f"Write the summary to align with this target role."
            )

        prompt = PROFESSIONAL_SUMMARY_PROMPT.format(
            name=name,
            recent_role=recent_role,
            years_experience=years_experience,
            key_skills=key_skills,
            education=edu_summary,
            job_context=job_context
        )

        return strip_llm_commentary(deduplicate_content(self.llm.generate(prompt).strip()))

    def tailor_experience(self, experience_list, job_data):
        """
        Rewrite work experience entries to emphasize job-relevant skills
        using Llama 3.2.

        Args:
            experience_list (list): List of experience dicts from resume_data
            job_data (dict): Parsed job description with keywords

        Returns:
            str: Rewritten experience section as formatted text
        """
        # Format experience entries into text for the prompt
        experience_text = self._format_experience_for_prompt(experience_list)

        required_skills = ", ".join(job_data.get("required_skills", []))
        responsibilities = ", ".join(job_data.get("key_responsibilities", [])[:5])
        keywords = ", ".join(job_data.get("keywords", []))

        prompt = EXPERIENCE_TAILORING_PROMPT.format(
            experience_text=experience_text,
            required_skills=required_skills,
            responsibilities=responsibilities,
            keywords=keywords
        )

        return strip_llm_commentary(deduplicate_content(self.llm.generate(prompt).strip()))

    def optimize_skills_section(self, skills_data, job_data):
        """
        Reorganize skills to prioritize job-relevant ones deterministically.
        No LLM call — uses fuzzy string matching to sort skills that appear
        in the job description's required/preferred lists to the front.

        Args:
            skills_data (dict): Skills from resume_data with categories
            job_data (dict): Parsed job description

        Returns:
            dict: Reorganized skills dict with job-relevant skills first
        """
        required = [s.lower().strip() for s in job_data.get("required_skills", [])]
        preferred = [s.lower().strip() for s in job_data.get("preferred_skills", [])]

        # Build a combined set of job keywords for fuzzy matching
        job_keywords = set(required + preferred)

        def relevance_score(skill):
            """
            Score a skill by how relevant it is to the job description.
            Higher score = more relevant = sorted first.
            - Exact match in required skills: 3
            - Exact match in preferred skills: 2
            - Substring match (skill in keyword or keyword in skill): 1
            - No match: 0
            """
            s = skill.lower().strip()
            if s in required:
                return 3
            if s in preferred:
                return 2
            # Fuzzy: check if any job keyword is a substring or vice versa
            for kw in job_keywords:
                if s in kw or kw in s:
                    return 1
            return 0

        def sort_skills(skill_list):
            """Sort a list of skills by job relevance (highest first),
            preserving original order among equally-scored skills."""
            if not skill_list:
                return skill_list
            return sorted(skill_list, key=lambda s: -relevance_score(s))

        optimized = {
            "technical": sort_skills(skills_data.get("technical", [])),
            "soft": sort_skills(skills_data.get("soft", [])),
            "tools": sort_skills(skills_data.get("tools", [])),
            "languages": sort_skills(skills_data.get("languages", [])),
        }

        return optimized

    # ------------------------------------------------------------------
    # Deterministic ATS Rubric Scorer
    # ------------------------------------------------------------------
    @staticmethod
    def _rubric_score(cv_text, cv_content, job_data):
        """
        Score the CV against the job description using a deterministic
        rubric.  Returns a dict with four category scores (each 0-25,
        totalling 0-100) plus matched/missing keyword lists.

        Categories:
          1. Keyword Match   (0-25): % of required + preferred skills found
          2. Completeness    (0-25): Are key CV sections populated?
          3. Impact Quality  (0-25): Do bullets have metrics & action verbs?
          4. Role Alignment  (0-25): Does experience reference the job domain?

        Args:
            cv_text (str): Full CV as plain text (lowered internally)
            cv_content (dict): Structured CV content dict
            job_data (dict): Parsed job description

        Returns:
            dict: {keyword_match, completeness, impact_quality,
                   role_alignment, rubric_total, matched, missing}
        """
        import re

        cv_lower = cv_text.lower()

        # ---- 1. Keyword Match (0-25) ----
        required = [s.strip().lower() for s in job_data.get("required_skills", []) if s.strip()]
        preferred = [s.strip().lower() for s in job_data.get("preferred_skills", []) if s.strip()]
        all_keywords = list(dict.fromkeys(required + preferred))  # dedupe, keep order

        matched = [kw for kw in all_keywords if kw in cv_lower]
        missing = [kw for kw in all_keywords if kw not in cv_lower]

        if all_keywords:
            # Required hits weighted 2x: ATS systems penalize missing "must-haves" harder
            req_hits = sum(1 for kw in required if kw in cv_lower)
            pref_hits = sum(1 for kw in preferred if kw in cv_lower)
            weighted_hits = req_hits * 2 + pref_hits
            weighted_total = len(required) * 2 + len(preferred)
            keyword_pct = weighted_hits / weighted_total if weighted_total else 0
        else:
            keyword_pct = 0
        keyword_score = round(keyword_pct * 25)

        # ---- 2. Completeness (0-25) ----
        expected_sections = [
            "professional_summary", "experience", "skills",
            "education", "projects"
        ]
        present = 0
        for sec in expected_sections:
            val = cv_content.get(sec)
            if val and val != "NONE" and val != [] and val != {}:
                if isinstance(val, str) and val.strip().upper() == "NONE":
                    continue
                present += 1
        completeness_score = round((present / len(expected_sections)) * 25)

        # ---- 3. Impact Quality (0-25) ----
        # Check experience + projects text for: numbers, %, $, action verbs
        impact_text = ""
        for sec in ["experience", "projects"]:
            val = cv_content.get(sec, "")
            if isinstance(val, str):
                impact_text += val + "\n"
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        impact_text += item + "\n"
                    elif isinstance(item, dict):
                        impact_text += " ".join(str(v) for v in item.values()) + "\n"

        bullets = [l.strip() for l in impact_text.split("\n")
                   if l.strip().startswith("-") or l.strip().startswith("•")]
        total_bullets = max(len(bullets), 1)  # floor at 1 to avoid division-by-zero

        # Metric check: numbers, %, $ in bullet
        metric_count = sum(1 for b in bullets if re.search(r'\d+[\.\d]*\s*[%$]|\$\s*\d|[\d,]+\s*(users|clients|customers|team|months|years)', b.lower()))
        # Action verb check: starts with strong verb (after the bullet marker)
        _ACTION_VERBS = {
            "developed", "implemented", "spearheaded", "orchestrated",
            "streamlined", "engineered", "designed", "managed", "led",
            "created", "built", "deployed", "optimized", "delivered",
            "launched", "executed", "established", "administered",
            "analyzed", "automated", "cultivated", "directed", "facilitated",
            "improved", "initiated", "negotiated", "resolved", "transformed",
            "collaborated", "coordinated", "maintained", "mentored",
            "pioneered", "reduced", "revamped", "scaled", "supervised"
        }
        verb_count = 0
        for b in bullets:
            first_word = re.sub(r'^[\-\*•\s]+', '', b).split()[0].lower() if b.strip() else ""
            if first_word in _ACTION_VERBS:
                verb_count += 1

        metric_pct = metric_count / total_bullets
        verb_pct = verb_count / total_bullets
        # Equal weight: metrics prove results, action verbs prove ownership
        impact_score = round(((metric_pct * 0.5) + (verb_pct * 0.5)) * 25)

        # ---- 4. Role Alignment (0-25) ----
        job_title = job_data.get("job_title", "").lower()
        responsibilities = [r.lower() for r in job_data.get("key_responsibilities", []) if r.strip()]

        alignment_hits = 0
        alignment_total = 0

        # Check job title words appear in experience
        if job_title:
            title_words = [w for w in job_title.split() if len(w) > 2]  # skip stop-words
            alignment_total += len(title_words)
            alignment_hits += sum(1 for w in title_words if w in cv_lower)

        # Check responsibilities keywords appear in CV
        if responsibilities:
            for resp in responsibilities[:10]:  # cap to keep scoring fast and avoid filler noise
                key_words = [w for w in resp.split() if len(w) > 3][:3]  # top 3 content words per resp
                alignment_total += len(key_words)
                alignment_hits += sum(1 for w in key_words if w in cv_lower)

        if alignment_total:
            role_score = round((alignment_hits / alignment_total) * 25)
        else:
            role_score = 12  # 12/25 = midpoint; don't penalize when JD lacks responsibilities

        rubric_total = keyword_score + completeness_score + impact_score + role_score

        return {
            "keyword_match": keyword_score,
            "completeness": completeness_score,
            "impact_quality": impact_score,
            "role_alignment": role_score,
            "rubric_total": rubric_total,
            "matched": matched,
            "missing": missing,
        }

    def score_ats_compatibility(self, cv_content, job_data, contact_info):
        """
        Score ATS compatibility using a two-layer approach:
          1. Deterministic rubric (instant) — 4 categories, 0-25 each
          2. LLM analysis via Qwen 2.5 — keyword-level detail & suggestions

        The final ats_score is the deterministic rubric total; the LLM
        provides supplementary suggestions and may refine matched/missing lists.

        Args:
            cv_content (dict): Generated CV content dictionary
            job_data (dict): Parsed job description
            contact_info (dict): Contact info for building full CV text

        Returns:
            dict: ATS report with rubric breakdown + LLM suggestions
        """
        # Convert CV content to plain text for analysis
        cv_text = format_cv_content_as_text(cv_content, contact_info)

        # --- Layer 1: Deterministic rubric (instant) ---
        rubric = self._rubric_score(cv_text, cv_content, job_data)

        print("\n  Scoring ATS compatibility...")

        # --- Layer 2: LLM suggestions via Qwen 2.5 ---
        print("    - Generating suggestions via Qwen 2.5...")

        prompt = ATS_SCORING_PROMPT.format(
            cv_text=cv_text,
            required_skills=", ".join(job_data.get("required_skills", [])),
            preferred_skills=", ".join(job_data.get("preferred_skills", [])),
            responsibilities=", ".join(job_data.get("key_responsibilities", [])),
            keywords=", ".join(job_data.get("keywords", []))
        )

        # Use extraction model (Qwen 2.5) for analysis
        response = self.llm.extract(prompt)

        llm_suggestions = []
        try:
            llm_report = clean_json_response(response)
            llm_suggestions = llm_report.get("suggestions", [])
        except (ValueError, KeyError):
            pass

        # Build the combined report
        return {
            "ats_score": rubric["rubric_total"],
            "rubric": {
                "keyword_match": rubric["keyword_match"],
                "completeness": rubric["completeness"],
                "impact_quality": rubric["impact_quality"],
                "role_alignment": rubric["role_alignment"],
            },
            "matched_keywords": rubric["matched"],
            "missing_keywords": rubric["missing"],
            "suggestions": llm_suggestions,
        }

    def revise_section(self, section_name, current_content, user_feedback):
        """
        Revise a specific CV section based on user feedback using Llama 3.2.

        Args:
            section_name (str): Name of the section to revise
            current_content (str): Current text content of the section
            user_feedback (str): User's revision instructions

        Returns:
            str: Revised section content
        """
        prompt = SECTION_REWRITE_PROMPT.format(
            section_name=section_name.replace("_", " ").title(),
            current_content=current_content,
            user_feedback=user_feedback
        )

        return strip_llm_commentary(deduplicate_content(self.llm.generate(prompt).strip()))

    def revise_for_keywords(self, cv_content, missing_keywords):
        """
        Automatically enhance CV content to incorporate missing ATS keywords
        where naturally appropriate, using Llama 3.2.

        Args:
            cv_content (dict): Current CV content dictionary
            missing_keywords (list): Keywords not yet present in CV

        Returns:
            dict: Updated CV content with keywords woven in where possible
        """
        print("\n  Auto-optimizing CV for missing keywords...")
        keywords_str = ", ".join(missing_keywords)

        # Revise sections that can naturally incorporate keywords
        revisable_sections = ["professional_summary", "experience", "projects"]

        for section in revisable_sections:
            if section not in cv_content or not cv_content[section]:
                continue

            current = cv_content[section]
            # Convert to string if needed
            if isinstance(current, (list, dict)):
                current = json.dumps(current, indent=2)

            prompt = REVISION_PROMPT.format(
                current_content=current,
                missing_keywords=keywords_str
            )

            revised = strip_llm_commentary(deduplicate_content(self.llm.generate(prompt).strip()))
            cv_content[section] = revised

        print("  Auto-optimization complete.")
        return cv_content

    def _build_job_context(self, job_data):
        """
        Build a job context string for inclusion in generation prompts.
        Returns empty string if no job data is provided.

        Args:
            job_data (dict or None): Parsed job description

        Returns:
            str: Formatted job context string for prompt injection
        """
        if not job_data:
            return ""

        parts = []
        job_title = job_data.get("job_title", "")
        company = job_data.get("company", "")
        if job_title:
            parts.append(f"Target Job: {job_title}")
        if company:
            parts.append(f"Company: {company}")

        required = job_data.get("required_skills", [])
        if required:
            parts.append(f"Required Skills: {', '.join(required)}")

        keywords = job_data.get("keywords", [])
        if keywords:
            parts.append(f"Key Keywords: {', '.join(keywords)}")

        if parts:
            return "Job Context:\n" + "\n".join(parts)
        return ""

    def _enhance_experience(self, experience_list):
        """
        Enhance experience entries without a specific job target.
        Uses Llama 3.2 to improve wording and add action verbs.

        Args:
            experience_list (list): List of experience dicts

        Returns:
            str: Enhanced experience section as formatted text
        """
        experience_text = self._format_experience_for_prompt(experience_list)

        prompt = CV_GENERATION_PROMPT.format(
            section_name="Work Experience",
            original_content=experience_text,
            job_context="No specific job target. Write a general-purpose, impactful CV."
        )

        return strip_llm_commentary(deduplicate_content(self.llm.generate(prompt).strip()))

    def _enhance_projects(self, projects_list, job_context):
        """
        Enhance project descriptions using Llama 3.2.

        Args:
            projects_list (list): List of project dicts
            job_context (str): Job context string for targeting

        Returns:
            str: Enhanced projects section as formatted text
        """
        # Format projects into text
        project_lines = []
        for proj in projects_list:
            if isinstance(proj, dict):
                name = proj.get("name", "Unnamed Project")
                desc = proj.get("description", "")
                tech = proj.get("technologies", [])
                tech_str = ", ".join(tech) if isinstance(tech, list) else str(tech)
                project_lines.append(f"{name}: {desc} (Technologies: {tech_str})")
            else:
                project_lines.append(str(proj))

        project_text = "\n".join(project_lines)

        if not project_text.strip():
            return []

        prompt = CV_GENERATION_PROMPT.format(
            section_name="Projects",
            original_content=project_text,
            job_context=job_context if job_context else "No specific job target."
        )

        return strip_llm_commentary(deduplicate_content(self.llm.generate(prompt).strip()))

    def _format_experience_for_prompt(self, experience_list):
        """
        Format experience entries into a text string suitable for LLM prompts.

        Args:
            experience_list (list): List of experience dicts

        Returns:
            str: Formatted experience text
        """
        lines = []
        for exp in experience_list:
            if isinstance(exp, dict):
                title = exp.get("title", "N/A")
                company = exp.get("company", "N/A")
                start = exp.get("start_date", "")
                end = exp.get("end_date", "")
                lines.append(f"{title} | {company} | {start} - {end}")
                desc = exp.get("description", "")
                if desc:
                    lines.append(f"  {desc}")
                achievements = exp.get("achievements", [])
                for ach in achievements:
                    if ach:
                        lines.append(f"  - {ach}")
                lines.append("")
            else:
                lines.append(str(exp))
                lines.append("")
        return "\n".join(lines)

    def _format_projects_for_prompt(self, projects_list):
        """
        Format project entries into a text string suitable for LLM prompts.

        Args:
            projects_list (list): List of project dicts

        Returns:
            str: Formatted projects text
        """
        lines = []
        for proj in projects_list:
            if isinstance(proj, dict):
                name = proj.get("name", "Unnamed Project")
                desc = proj.get("description", "")
                tech = proj.get("technologies", [])
                tech_str = ", ".join(tech) if isinstance(tech, list) else str(tech)
                lines.append(f"{name}: {desc} (Technologies: {tech_str})")
            else:
                lines.append(str(proj))
        return "\n".join(lines)

    def _estimate_years(self, experience_list, resume_data=None):
        """
        Return years of experience. Uses the user-provided value from
        resume_data['years_experience'] if available (set during guided
        input). Otherwise falls back to counting real experience entries.
        Returns "0" for freshers or when entries are placeholder/empty.

        Args:
            experience_list (list): List of experience dicts
            resume_data (dict or None): Full resume data that may contain
                                        a 'years_experience' field

        Returns:
            str: Years string (e.g., "7", "5+", "0")
        """
        # Prefer user-provided years if available
        if resume_data:
            user_years = str(resume_data.get("years_experience", "")).strip()
            if user_years and user_years != "0":
                return user_years

        if not experience_list:
            return "0"

        # Filter out placeholder/empty entries before counting
        real_entries = [e for e in experience_list if self._is_real_experience(e)]

        if not real_entries:
            return "0"

        # Rough heuristic (~2 yrs/role avg); only used when user didn't provide years
        num_roles = len(real_entries)
        if num_roles >= 5:
            return "10+"
        elif num_roles >= 3:
            return "5+"
        elif num_roles >= 2:
            return "3+"
        else:
            return "1+"

    @staticmethod
    def _is_real_experience(entry):
        """
        Check whether an experience entry contains real data or is a
        placeholder / empty entry. Returns False for entries like
        "No experience", "N/A", "no", "none", single-char entries, etc.

        Args:
            entry: An experience dict or string

        Returns:
            bool: True if the entry appears to describe a real job
        """
        placeholder_tokens = {
            "no", "n/a", "na", "none", "nil", "no experience",
            "not applicable", "-", ".", ""
        }

        if isinstance(entry, dict):
            title = (entry.get("title") or "").strip().lower()
            company = (entry.get("company") or "").strip().lower()
            # Both title and company must look real
            if title in placeholder_tokens or company in placeholder_tokens:
                return False
            if len(title) <= 2 and len(company) <= 2:
                return False
            return True
        elif isinstance(entry, str):
            return entry.strip().lower() not in placeholder_tokens
        return False
