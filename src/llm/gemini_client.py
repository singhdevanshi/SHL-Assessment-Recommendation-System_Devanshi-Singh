"""
Improved Gemini API client for the SHL Assessment Recommender.
This version includes better error handling and JSON parsing.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Union
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define default prompt templates in case config.py is missing them
DEFAULT_PROMPT_TEMPLATES = {
    "extract_requirements": """
Please analyze the following job description and extract the key requirements.
Format your response as a JSON object with the following structure:
{
  "technical_skills": ["skill1", "skill2", ...],
  "soft_skills": ["skill1", "skill2", ...],
  "experience_level": "entry/mid/senior/executive",
  "key_responsibilities": ["responsibility1", "responsibility2", ...],
  "required_competencies": ["competency1", "competency2", ...]
}

Job Description:
{job_description}

JSON Response:
""",

    "rerank_assessments": """
I need to rank which SHL assessments would be most relevant for a job with these requirements:

{job_requirements}

Here are the candidate assessments:
{assessments_text}

For each assessment, provide a relevance score (0-100) and explanation of why it matches the job.
Return your response as a JSON array with this structure:
[
  {{
    "assessment_index": 1,
    "relevance_score": 95,
    "explanation": "This assessment tests...",
    "matched_requirements": ["requirement1", "requirement2"]
  }},
  ...
]

Focus on providing accurate relevance scores based on how well each assessment matches the job requirements.
JSON Response:
""",

    "generate_explanation": """
I need to explain why this assessment is a good match for a job.

Job Description:
{job_description}

Assessment:
Name: {assessment_name}
Description: {assessment_description}
Type: {assessment_type}
Duration: {assessment_duration} minutes

Write a concise explanation (2-3 sentences) of why this assessment is relevant for this job position.
"""
}

# Default LLM configuration
DEFAULT_LLM_CONFIG = {
    "model": "gemini-1.5-pro",
    "temperature": 0.2,
    "max_output_tokens": 1024
}

class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, 
                 config: Optional[Dict[str, Any]] = None,
                 prompt_templates: Optional[Dict[str, str]] = None):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Gemini API key. If None, will try to load from environment variable.
            config: LLM configuration. If None, will use default.
            prompt_templates: Prompt templates. If None, will try to import from config or use default.
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("Gemini API key not provided and not found in environment variables (GEMINI_API_KEY).")
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Try to import config if not provided
        if config is None:
            try:
                from .config import LLM_CONFIG
                config = LLM_CONFIG
            except ImportError:
                print("Warning: Could not import LLM_CONFIG, using default configuration.")
                config = DEFAULT_LLM_CONFIG
        
        # Try to import prompt templates if not provided
        if prompt_templates is None:
            try:
                from .config import PROMPT_TEMPLATES
                prompt_templates = PROMPT_TEMPLATES
            except ImportError:
                print("Warning: Could not import PROMPT_TEMPLATES, using default templates.")
                prompt_templates = DEFAULT_PROMPT_TEMPLATES
        
        # Store configuration
        self.config = config
        self.prompt_templates = prompt_templates
        
        # Default model and configuration
        self.model_name = self.config.get("model", "gemini-1.5-pro")
        self.temperature = self.config.get("temperature", 0.2)
        self.max_output_tokens = self.config.get("max_output_tokens", 1024)
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel(
                self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                }
            )
            print(f"Successfully initialized Gemini model: {self.model_name}")
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            raise
    
    def _parse_json_from_response(self, text: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Parse JSON from the LLM response text, handling various formats.
        
        Args:
            text: The text response from the LLM
            
        Returns:
            Parsed JSON as dict or list
        """
        # Strip markdown code blocks if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Try to parse the JSON directly first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON-like object in text
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON array in text
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        # If all parsing attempts failed, print a verbose error
        print(f"Could not parse JSON from response. Response was: {text[:200]}...")
        
        # As a last resort, create a basic structured response
        if "technical_skills" in text.lower() or "soft_skills" in text.lower():
            # This looks like an extract_requirements response
            return {
                "technical_skills": ["Error parsing response"],
                "soft_skills": [],
                "experience_level": "Not determined",
                "key_responsibilities": [],
                "required_competencies": []
            }
        elif "assessment" in text.lower() and "relevance" in text.lower():
            # This looks like a rerank_assessments response
            return [{"assessment_index": 1, "relevance_score": 50, "explanation": "Error parsing response"}]
        else:
            # Generic fallback
            return {"error": "Could not parse response", "raw_text": text[:500]}
    
    def extract_job_requirements(self, job_description: str) -> Dict[str, Any]:
        """
        Extract structured job requirements from a job description.
        
        Args:
            job_description: The job description text.
            
        Returns:
            A dictionary containing structured job requirements.
        """
        prompt = self.prompt_templates.get("extract_requirements", DEFAULT_PROMPT_TEMPLATES["extract_requirements"]).format(
            job_description=job_description
        )
        
        try:
            print("Sending job requirements extraction request to Gemini...")
            response = self.model.generate_content(prompt)
            print("Received response from Gemini for job requirements")
            
            parsed_response = self._parse_json_from_response(response.text)
            
            # Ensure the response has the expected structure
            expected_keys = [
                "technical_skills", "soft_skills", "experience_level",
                "key_responsibilities", "required_competencies"
            ]
            
            for key in expected_keys:
                if key not in parsed_response:
                    parsed_response[key] = []
            
            return parsed_response
        except Exception as e:
            print(f"Error extracting job requirements: {e}")
            # Return a basic structure if there's an error
            return {
                "technical_skills": [],
                "soft_skills": [],
                "experience_level": "Not determined",
                "key_responsibilities": [],
                "required_competencies": []
            }
    
    def rerank_assessments(self, job_requirements: Dict[str, Any], 
                         candidate_assessments: List[Dict[str, Any]],
                         top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Rerank candidate assessments based on how well they match job requirements.
        
        Args:
            job_requirements: Structured job requirements
            candidate_assessments: List of candidate assessments from vector search
            top_k: Number of top assessments to return
            
        Returns:
            List of reranked assessments with relevance scores
        """
        # Format job requirements for the prompt
        job_req_text = json.dumps(job_requirements, indent=2)
        
        # Format assessments for the prompt
        assessments_text = ""
        for i, assessment in enumerate(candidate_assessments):
            assessments_text += f"""
Assessment {i+1}:
Name: {assessment.get('name', '')}
Description: {assessment.get('description', 'No description available')}
Type: {assessment.get('test_types', '')}
Duration: {assessment.get('duration', '')} minutes
Remote Testing: {assessment.get('remote_testing', '')}
Adaptive Testing: {assessment.get('adaptive_support', '')}

"""
        
        # Use the template from config
        prompt = self.prompt_templates.get("rerank_assessments", DEFAULT_PROMPT_TEMPLATES["rerank_assessments"]).format(
            job_requirements=job_req_text,
            assessments_text=assessments_text
        )
        
        try:
            print(f"Sending reranking request to Gemini for {len(candidate_assessments)} assessments...")
            response = self.model.generate_content(prompt)
            print("Received response from Gemini for reranking")
            
            parsed_response = self._parse_json_from_response(response.text)
            
            # If parsing returns a dict instead of a list, check for results key
            if isinstance(parsed_response, dict):
                parsed_response = parsed_response.get("results", [])
            
            # If we still don't have a list, return an empty list
            if not isinstance(parsed_response, list):
                print(f"Expected list but got {type(parsed_response)} from reranking")
                parsed_response = []
            
            # Merge the original assessment data with the reranking information
            reranked_assessments = []
            for item in parsed_response:
                if not isinstance(item, dict):
                    continue
                    
                assessment_idx = item.get("assessment_index", 0)
                # Adjust for 1-based indexing in the prompt
                if assessment_idx > 0 and assessment_idx <= len(candidate_assessments):
                    assessment = candidate_assessments[assessment_idx - 1].copy()
                    assessment.update({
                        "relevance_score": item.get("relevance_score", 0),
                        "explanation": item.get("explanation", "No explanation provided."),
                        "matched_requirements": item.get("matched_requirements", [])
                    })
                    reranked_assessments.append(assessment)
            
            # Sort and return top_k
            reranked_assessments.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            return reranked_assessments[:top_k]
        
        except Exception as e:
            print(f"Error reranking assessments: {e}")
            # Fallback to returning original assessments in original order
            fallback_assessments = []
            for i, assessment in enumerate(candidate_assessments[:top_k]):
                assessment_copy = assessment.copy()
                assessment_copy.update({
                    "relevance_score": 100 - (i * 10),  # Arbitrary scoring
                    "explanation": "This assessment may be relevant based on the job requirements.",
                    "matched_requirements": []
                })
                fallback_assessments.append(assessment_copy)
            return fallback_assessments
    
    def generate_explanation(self, job_description: str, assessment: Dict[str, Any]) -> str:
        """
        Generate a natural language explanation of why an assessment is recommended.
        
        Args:
            job_description: The original job description
            assessment: The assessment that was recommended
            
        Returns:
            A natural language explanation
        """
        # Validate inputs first
        if not job_description or job_description.strip() == "":
            print("Warning: Empty job description provided to generate_explanation")
            job_description = "No job description provided."
        
        # Make sure we have enough assessment information
        assessment_name = assessment.get("name", "")
        assessment_description = assessment.get("description", "")
        
        if not assessment_name and not assessment_description:
            return f"This assessment may be relevant based on the job requirements."
        
        # Use the relevance score and matched requirements if available
        relevance_info = ""
        if "relevance_score" in assessment and assessment["relevance_score"] > 0:
            relevance_info = f" with a relevance score of {assessment['relevance_score']}/100"
        
        matched_req = assessment.get("matched_requirements", [])
        if matched_req and len(matched_req) > 0:
            matched_str = ", ".join(matched_req[:3])  # Limit to first 3 for brevity
            if len(matched_req) > 3:
                matched_str += ", and other requirements"
            relevance_info += f" matching: {matched_str}"
        
        # If we already have an explanation from reranking, use it
        if "explanation" in assessment and assessment["explanation"] and assessment["explanation"] != "No explanation provided.":
            return assessment["explanation"]
            
        # Otherwise generate a new explanation
        prompt = self.prompt_templates.get("generate_explanation", DEFAULT_PROMPT_TEMPLATES["generate_explanation"]).format(
            job_description=job_description,
            assessment_name=assessment.get("name", "Unknown"),
            assessment_description=assessment.get("description", "No description"),
            assessment_type=assessment.get("test_types", "N/A"),
            assessment_duration=assessment.get("duration", "N/A")
        )
        
        try:
            print("Sending explanation request to Gemini...")
            response = self.model.generate_content(prompt)
            explanation = response.text.strip()
            
            # Validate the response isn't empty or an error message
            if not explanation or "error" in explanation.lower() or len(explanation) < 10:
                # Fallback explanation based on assessment details
                if assessment_description:
                    return f"This assessment tests {assessment_name} skills{relevance_info}. {assessment_description[:100]}..."
                else:
                    return f"This assessment evaluates {assessment_name} competencies{relevance_info}."
            
            return explanation
        except Exception as e:
            print(f"Error generating explanation: {e}")
            # Fallback with more context than the generic message
            if assessment_description:
                return f"This assessment focuses on {assessment_name}{relevance_info}."
            else:
                return f"This assessment may be relevant based on the job requirements{relevance_info}."