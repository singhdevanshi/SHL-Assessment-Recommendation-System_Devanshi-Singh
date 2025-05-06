"""
Configuration settings for the LLM components.
"""
# LLM API configuration
LLM_CONFIG = {
    "model": "gemini-1.5-pro",  # Model to use (removed trailing space)
    "temperature": 0.2,  # Lower temperature for more deterministic outputs
    "max_output_tokens": 1024,  # Maximum tokens for response
}

# Prompt templates
PROMPT_TEMPLATES = {
    "extract_requirements": """
    Extract key job requirements and skills from the following job description.
    Format the output as JSON with these categories:
    - technical_skills: List of technical skills required
    - soft_skills: List of soft skills mentioned
    - experience_level: Entry/Mid/Senior
    - key_responsibilities: List of main job responsibilities
    - required_competencies: List of competencies that could be assessed
   
    Job Description:
    {job_description}
    """,
   
    "rerank_assessments": """
    Given these job requirements:
    {job_requirements}
   
    And these candidate assessments:
    {assessments_text}
   
    Rank the assessments by how well they match the job requirements.
    For each assessment, provide:
    1. A relevance score from 0-100
    2. A brief explanation of why it matches or doesn't match
    3. Which job requirements it addresses
   
    Format your response as a JSON array with objects containing:
    - assessment_id: The ID or name of the assessment
    - relevance_score: Numerical score 0-100
    - explanation: Brief explanation
    - matched_requirements: List of matched requirements
   
    Sort the assessments by relevance_score in descending order.
    """,
   
    "generate_explanation": """
    Given this job description:
    {job_description}
   
    And this assessment:
    Name: {assessment_name}
    Description: {assessment_description}
    Type: {assessment_type}
    Duration: {assessment_duration}
   
    Explain in 2-3 sentences why this assessment would be appropriate for evaluating
    candidates for this position. Focus on how the assessment measures skills and
    competencies required for the job.
    """
}