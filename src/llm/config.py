"""
Configuration for LLM settings and prompt templates.
"""

LLM_CONFIG = {
    "model": "gemini-1.5-pro",
    "temperature": 0.2,
    "max_output_tokens": 1024,
}

PROMPT_TEMPLATES = {
    "extract_requirements": """
You are an expert HR professional analyzing a job description.
Extract the key elements from the following job description into structured categories.

Job Description:
{job_description}

Please analyze and output ONLY a JSON with the following categories:
- technical_skills: List of technical skills required
- soft_skills: List of soft skills/interpersonal skills required
- experience_level: General experience level (entry, mid, senior)
- key_responsibilities: Primary duties of the role
- required_competencies: Essential competencies needed to succeed

Return your analysis in valid JSON format only.
""",

    "rerank_assessments": """
You're an assessment selection expert tasked with recommending the best assessments for a job.

Job Requirements:
{job_requirements}

Available Assessments:
{assessments_text}

Instructions:
1. Analyze how well each assessment matches the specific job requirements
2. Assign a relevance score to each assessment (0-100)
3. Provide a brief explanation of why each assessment is relevant
4. List specific job requirements that each assessment addresses

Return your analysis in the following JSON format ONLY:
```json
[
  {{
    "assessment_index": 1,
    "relevance_score": 85,
    "explanation": "This assessment directly measures critical thinking skills required for data analysis roles.",
    "matched_requirements": ["analytical skills", "problem solving", "data interpretation"]
  }},
  ...
]
```

IMPORTANT: Only include the JSON in your response, no additional text.
"""
}