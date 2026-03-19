"""
prompts.py - Centralized prompt templates for all LLM interactions.
Separates prompt engineering from business logic for maintainability.

QWEN 2.5 PROMPTS: Used for extraction/analysis tasks (structured JSON output)
LLAMA 3.2 PROMPTS: Used for generation/rewriting tasks (natural language output)

Each prompt includes clear instructions, output format specification,
and constraints to ensure consistent, high-quality results.
"""


# =============================================================================
# QWEN 2.5 PROMPTS (Extraction & Analysis - low temperature, structured output)
# =============================================================================

RESUME_EXTRACTION_PROMPT = """Extract ALL resume data into compact JSON. Do NOT skip entries. Use "" for missing strings, [] for missing lists. Do NOT invent data.

RULES:
- Extract EVERY job, project, and certification — do NOT skip any.
- "certifications" = ONLY formal credentials with an issuing body (e.g., AWS Certified, PMP, CPA). Do NOT put awards, ratings, rankings, or competition results here.
- "achievements" = Awards, rankings, competition results, honors, hackathons, and other accomplishments.
- CRITICAL: For EACH experience entry, extract ALL bullet points / responsibilities / achievements listed under it. Do NOT return empty achievements for any role that has bullets in the resume.
- Keep achievement bullets SHORT (max 15 words each). Capture the key point only.
- Keep descriptions to 1 short sentence max.
- "languages" = SPOKEN languages (English, Hindi), NOT programming languages.
- Do NOT invent LinkedIn, GitHub, portfolio, or any URLs not in the resume.
- "gpa" field: Copy the EXACT score with its original label from the resume. Examples: "CGPA: 8.6/10", "72%", "GPA: 3.8/4.0", "85 percentile". Do NOT convert between formats.

Resume:
{resume_text}

Return ONLY valid JSON (no markdown, no explanation):
{{"name":"","contact":{{"email":"","phone":"","location":"","linkedin":"","github":"","portfolio":""}},"professional_summary":"","years_experience":"","education":[{{"degree":"","institution":"","year":"","gpa":""}}],"experience":[{{"title":"","company":"","start_date":"","end_date":"","achievements":[]}}],"skills":{{"technical":[],"soft":[],"tools":[],"languages":[]}},"certifications":[{{"name":"","issuer":"","year":""}}],"projects":[{{"name":"","description":"","technologies":[]}}],"achievements":[]}}"""


# -----------------------------------------------------------------------------
# MULTI-PASS EXTRACTION PROMPTS
# The single RESUME_EXTRACTION_PROMPT above asks for ALL sections in one JSON
# blob.  For long resumes the small Qwen 2.5 1.5B model runs out of output
# tokens and silently produces truncated JSON — losing entire sections.
#
# To fix this, extraction is split into two focused passes:
#   Pass 1 (RESUME_HEADER_PROMPT)     — everything EXCEPT work experience
#   Pass 2 (RESUME_EXPERIENCE_PROMPT) — ONLY work experience
#
# Each pass receives the full resume text (so the model has context) but
# outputs a much smaller JSON — comfortably fitting within the 2048 token
# limit.  Results are merged in data_extractor.py.
#
# The original RESUME_EXTRACTION_PROMPT is kept as a single-pass fallback
# in case both multi-pass attempts fail.
# -----------------------------------------------------------------------------

# Pass 1: Header data — personal info, education, skills, certifications,
# projects, achievements, publications.  Explicitly excludes work experience
# so the model doesn't waste output tokens on it.
RESUME_HEADER_PROMPT = """Extract personal info, education, skills, certifications, projects, achievements, and publications from this resume into compact JSON. Do NOT extract work experience — that will be extracted separately. Use "" for missing strings, [] for missing lists. Do NOT invent data.

RULES:
- Extract EVERY education entry, certification, project, and achievement — do NOT skip any.
- "certifications" = ONLY formal credentials with an issuing body (e.g., AWS Certified, PMP, CPA). Do NOT put awards, ratings, rankings, or competition results here.
- "achievements" = Awards, rankings, competition results, honors, hackathons, and other accomplishments.
- Keep descriptions to 1 short sentence max.
- "languages" = SPOKEN languages (English, Hindi), NOT programming languages.
- Do NOT invent LinkedIn, GitHub, portfolio, or any URLs not in the resume.
- "gpa" field: Copy the EXACT score with its original label from the resume. Examples: "CGPA: 8.6/10", "72%", "GPA: 3.8/4.0". Do NOT convert between formats.

Resume:
{resume_text}

Return ONLY valid JSON (no markdown, no explanation):
{{"name":"","contact":{{"email":"","phone":"","location":"","linkedin":"","github":"","portfolio":""}},"professional_summary":"","years_experience":"","education":[{{"degree":"","institution":"","year":"","gpa":""}}],"skills":{{"technical":[],"soft":[],"tools":[],"languages":[]}},"certifications":[{{"name":"","issuer":"","year":""}}],"projects":[{{"name":"","description":"","technologies":[]}}],"achievements":[],"publications":[]}}"""


# Pass 2: Work experience only — job roles with titles, companies, dates,
# and ALL achievement/responsibility bullet points.  The JSON template is
# minimal so the model can spend its output tokens on the actual content.
RESUME_EXPERIENCE_PROMPT = """Extract ONLY the work experience section from this resume into compact JSON. Do NOT extract personal info, education, skills, or other sections.

RULES:
- Extract EVERY job/role — do NOT skip any.
- CRITICAL: For EACH role, extract ALL bullet points / responsibilities / achievements listed under it. Do NOT return empty achievements for any role that has bullets in the resume.
- Keep achievement bullets SHORT (max 15 words each). Capture the key point only.
- Include title, company, start_date, end_date, and achievements for each role.

Resume:
{resume_text}

Return ONLY valid JSON (no markdown, no explanation):
{{"experience":[{{"title":"","company":"","start_date":"","end_date":"","achievements":[]}}]}}"""


FOLLOW_UP_PROMPT_JD = """The candidate is applying for a specific role. Compare their resume against the job requirements and identify 3-5 gaps where additional information would strengthen their application.

Resume sections present: {sections_summary}
Job requires: {required_skills}
Job prefers: {preferred_skills}
Job title: {job_title}

CRITICAL RULES:
- ONLY ask about genuine GAPS between the resume and job requirements.
- Do NOT ask about skills, projects, certifications, or experience already listed above.
- Focus on job requirements the candidate MIGHT have but didn't mention.
- Keep questions short and specific.

Return a JSON array:
[{{"section":"skills","question":"The role requires experience with budgeting. Have you managed budgets or financial planning?"}},{{"section":"experience","question":"The role involves stakeholder management. Have you worked directly with clients or senior leadership?"}}]

Valid sections: "skills", "projects", "experience", "achievements", "professional_summary"
Return ONLY the JSON array."""


FOLLOW_UP_PROMPT_GENERAL = """The candidate wants a general-purpose CV. Their resume has some thin sections that could be strengthened.

Resume sections present: {sections_summary}

CRITICAL RULES:
- ONLY ask about sections that are EMPTY or very thin (listed as "empty" or "thin" above).
- Do NOT ask about sections that are already well-populated.
- Keep questions short and specific.
- Ask 2-4 questions maximum.

Return a JSON array:
[{{"section":"achievements","question":"Can you share 2-3 key professional accomplishments with measurable outcomes?"}},{{"section":"professional_summary","question":"How would you describe your professional identity in 2-3 sentences?"}}]

Valid sections: "skills", "projects", "experience", "achievements", "professional_summary"
Return ONLY the JSON array."""


JOB_PARSING_PROMPT = """You are a precise job description analysis system.
Extract ALL requirements, qualifications, and keywords from the following job posting.

Job Description:
---
{job_text}
---

Fill in this JSON template with the extracted information:
{{
    "job_title": "FILL_IN",
    "company": "FILL_IN",
    "required_skills": ["FILL_IN"],
    "preferred_skills": ["FILL_IN"],
    "experience_requirements": "FILL_IN",
    "education_requirements": "FILL_IN",
    "key_responsibilities": ["FILL_IN"],
    "keywords": ["FILL_IN"],
    "industry": "FILL_IN",
    "seniority_level": "FILL_IN"
}}

Completed JSON (replace all FILL_IN values with actual extracted data, use "" or [] for missing fields):"""


ATS_SCORING_PROMPT = """You are an ATS (Applicant Tracking System) compatibility analyzer.
Compare the following CV content against the job description keywords and provide a match analysis.

CV Content:
---
{cv_text}
---

Job Keywords and Requirements:
---
Required Skills: {required_skills}
Preferred Skills: {preferred_skills}
Key Responsibilities: {responsibilities}
All Keywords: {keywords}
---

Analyze keyword overlap and provide a compatibility score.
Fill in this JSON template with your analysis:
{{
    "ats_score": 0,
    "matched_keywords": ["FILL_IN"],
    "missing_keywords": ["FILL_IN"],
    "suggestions": ["FILL_IN"]
}}

Note: ats_score must be an integer from 0 to 100 reflecting keyword coverage percentage.

Completed JSON (replace all FILL_IN values with actual analysis results):"""


# =============================================================================
# LLAMA 3.2 PROMPTS (Generation & Rewriting - moderate temperature, natural prose)
# =============================================================================

PROFESSIONAL_SUMMARY_PROMPT = """Write a compelling professional summary for a CV.

Candidate Background:
- Name: {name}
- Most Recent Role: {recent_role}
- Years of Experience: {years_experience}
- Key Skills: {key_skills}
- Education: {education}

{job_context}

CRITICAL RULES:
- Do NOT fabricate, invent, or assume any experience, skills, or achievements not listed above.
- If Years of Experience is "0" or Most Recent Role is "N/A", this is a fresher/recent graduate.
  Write an objective statement focused on education, eagerness to learn, and career goals instead.
- ONLY mention skills, education, and experience that are explicitly listed above.
- If Key Skills is "N/A", do NOT invent skills.

Write a 3-4 sentence professional summary that:
1. Opens with professional identity (use years of experience ONLY if greater than 0)
2. Highlights 2-3 most relevant skills or achievements from the data above
3. Ends with value proposition or career objective
4. Uses strong action-oriented language

Return ONLY the summary paragraph. No titles, labels, introductory text, or closing notes."""


CV_GENERATION_PROMPT = """You are an expert CV writer specializing in ATS-optimized resumes.
Rewrite the following CV section to be more impactful and keyword-rich.

Section: {section_name}
Original Content:
---
{original_content}
---

{job_context}

CRITICAL RULES:
- Do NOT fabricate, invent, or add any information not present in the Original Content above.
- Do NOT create new job titles, companies, achievements, or experiences.
- ONLY improve the wording and presentation of what is already there.

Guidelines:
- Use strong action verbs (Led, Developed, Implemented, Optimized, etc.)
- Quantify achievements where possible (%, $, numbers)
- Naturally incorporate relevant keywords without keyword stuffing
- Keep bullet points concise (1-2 lines each)
- Maintain truthfulness - enhance presentation, do NOT fabricate facts

Return ONLY the rewritten section content, properly formatted with bullet points where appropriate.
Do NOT include any introductory text, disclaimers, notes, or explanations."""


SECTION_REWRITE_PROMPT = """Rewrite this CV section based on user feedback.

Section: {section_name}
Current Content:
---
{current_content}
---

User Feedback: {user_feedback}

Rewrite the section incorporating the user's feedback while maintaining:
- Professional tone
- ATS-friendly formatting
- Concise, impactful language

CRITICAL: Do NOT fabricate or add information not present in the current content.
Only restructure, rephrase, or adjust emphasis as the user requests.

Return ONLY the rewritten section content. No introductory text, no notes, no disclaimers."""


REVISION_PROMPT = """Enhance this CV content to naturally incorporate these missing keywords
without keyword stuffing or fabricating experience.

Current Content:
---
{current_content}
---

Missing Keywords to Incorporate: {missing_keywords}

CRITICAL RULES:
- ONLY add keywords where they naturally fit the candidate's actual experience
- Do NOT fabricate, invent, or assume any new experiences, skills, or achievements
- Do NOT change job titles, company names, dates, or degree names
- Do NOT add technologies or tools the candidate did not mention
- If a keyword cannot be naturally incorporated into existing content, SKIP IT
- Maintain professional tone and readability
- The output must contain the SAME number of roles/entries as the input

Return ONLY the enhanced content. No introductory text, no notes, no disclaimers."""


EXPERIENCE_TAILORING_PROMPT = """You are an expert CV writer. Rewrite the following work experience
entries to better align with the target job requirements.

CRITICAL RULES — VIOLATIONS MAKE THE OUTPUT USELESS:
- Do NOT fabricate, invent, or add ANY jobs, companies, achievements, or experiences
  that are not present in the original data below.
- Do NOT change job titles, company names, or dates.
- Do NOT add technologies, tools, or skills the candidate did not mention.
- ONLY improve the wording and presentation of what already exists.
- The output MUST contain the SAME number of roles as the input (no additions, no removals).
- If a bullet point cannot be honestly reframed for the target role, keep it as-is.

Work Experience:
---
{experience_text}
---

Target Job Requirements (for emphasis guidance only — do NOT add these if not in the resume):
- Required Skills: {required_skills}
- Key Responsibilities: {responsibilities}
- Industry Keywords: {keywords}

Guidelines:
- Apply this bullet formula: [Strong Action Verb] + [Task/Project] + [Technologies/Methods] + [Impact/Scope]
- Wrap key technical skills, tools, and platforms in **double asterisks** (e.g. **Python**, **Docker**, **AWS**)
- Quantify achievements wherever possible (numbers, percentages, dollar amounts)
- Emphasize experiences that match the target job requirements
- Naturally weave in relevant keywords from the job description
- Do NOT fabricate or exaggerate - only enhance the presentation of real experience
- Keep each bullet point to 1-2 lines maximum
- Maintain chronological order and accurate company/date information

IMPORTANT: Return ONLY the formatted experience entries below. Do NOT include:
- Any introductory text ("Here is...", "Rewritten entries:", etc.)
- Any explanatory notes or disclaimers at the end
- Any "Please note..." or "Note:" commentary
- Any sign-off ("Best regards", "Let me know", etc.)

Return ONLY the experience entries in this exact format (include EVERY role — each with at least 2 bullets):
TITLE | COMPANY | START_DATE - END_DATE
- Achievement/responsibility bullet point
- Achievement/responsibility bullet point

(blank line between roles)"""


SKILLS_OPTIMIZATION_PROMPT = """Reorder skills by job relevance. Do NOT add new skills. Return ONLY JSON.

Skills: technical=[{technical_skills}], soft=[{soft_skills}], tools=[{tools}], languages=[{languages}]
Job requires: {required_skills}
Job prefers: {preferred_skills}

Return JSON: {{"technical":[],"soft":[],"tools":[],"languages":[]}}"""


COMBINED_CV_GENERATION_PROMPT = """You are a CV writer. Output ONLY CV content. Do NOT write any commentary, greetings, sign-offs, confirmations, or questions. Do NOT say "Best regards", "Please confirm", "Let me know", or anything similar. Start directly with ===PROFESSIONAL_SUMMARY===.

Generate 3 CV sections using ONLY the candidate data below.

RULES:
- Do NOT fabricate or invent ANY information not present below.
- Do NOT mention the target job title in the summary.
- Do NOT add commentary like "No impact mentioned".
- Include ALL experience entries listed below — EVERY role MUST appear with at least 2 bullets each.
- Include ALL projects listed below.
- Write "NONE" if a section has no data.
- Do NOT echo back the candidate data or input fields.

BULLET FORMULA (apply to EVERY experience bullet):
[Strong Action Verb] + [Task/Project] + [Technologies/Methods] + [Impact/Scope]
Keep each bullet to ONE line (max 20 words). Be concise.

KEYWORD BOLDING: Wrap key technical skills, tools, and platforms in **double asterisks**. Example: **Python**, **Google Analytics**, **SEO**. Do NOT bold generic words.

ELABORATION: If a bullet is vague or under 8 words, expand it professionally while staying truthful. Start every bullet with a strong action verb. Where no metric exists, describe the scope.

Name: {name} | Role: {recent_role} | Experience: {years_experience} yrs | Skills: {key_skills} | Education: {education}

Work Experience:
{experience_text}

Projects:
{projects_text}

{job_context}

Output EXACTLY this format and NOTHING else:

===PROFESSIONAL_SUMMARY===
3-4 sentences. No **bold** in summary.

===EXPERIENCE===
(Include EVERY role below. Each role MUST have 2-4 bullets. Do NOT skip any role.)
TITLE | COMPANY | START - END
- Bullet with action verb, task, **technologies**, and impact/scope
(Write "NONE" if no experience)

===PROJECTS===
PROJECT_NAME: 2-3 sentences with **key technologies** bolded. (Technologies: list)
(Write "NONE" if no projects)"""
