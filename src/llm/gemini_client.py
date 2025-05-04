"""
Gemini API client for the SHL Assessment Recommender.
This module handles all interactions with the Gemini API.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Union
import google.generativeai as genai
from dotenv import load_dotenv
from .config import LLM_CONFIG, PROMPT_TEMPLATES

# Load environment variables
load_dotenv()

class GeminiClient:
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Gemini API key. If None, will try to load from environment variable.
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("Gemini API key not provided and not found in environment variables.")
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Default model and configuration
        self.model_name = LLM_CONFIG.get("model", "gemini-1.5-pro")
        self.temperature = LLM_CONFIG.get("temperature", 0.2)
        self.max_output_tokens = LLM_CONFIG.get("max_output_tokens", 1024)
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            self.model_name,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            }
        )
    
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
        text = re.sub(r'```', '', text)
        
        # Try to parse the JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # Try to find JSON-like object in text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            
            # Try to find JSON array in text
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            
            # Return empty dict as fallback
            print(f"Could not parse JSON from response: {text[:100]}...")
            return {}
    
    def extract_job_requirements(self, job_description: str) -> Dict[str, Any]:
        """
        Extract structured job requirements from a job description.
        
        Args:
            job_description: The job description text.
            
        Returns:
            A dictionary containing structured job requirements.
        """
        prompt = PROMPT_TEMPLATES["extract_requirements"].format(
            job_description=job_description
        )
        
        try:
            response = self.model.generate_content(prompt)
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
        prompt = PROMPT_TEMPLATES["rerank_assessments"].format(
            job_requirements=job_req_text,
            assessments_text=assessments_text
        )
        
        try:
            response = self.model.generate_content(prompt)
            parsed_response = self._parse_json_from_response(response.text)
            
            # If parsing returns a dict instead of a list, check for results key
            if isinstance(parsed_response, dict):
                parsed_response = parsed_response.get("results", [])
            
            # If we still don't have a list, return an empty list
            if not isinstance(parsed_response, list):
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
                        "explanation": item.get("explanation", ""),
                        "matched_requirements": item.get("matched_requirements", [])
                    })
                    reranked_assessments.append(assessment)
            
            # Sort by relevance score and take top_k
            reranked_assessments.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            return reranked_assessments[:top_k]
        
        except Exception as e:
            print(f"Error reranking assessments: {e}")
            # Fallback to returning original assessments in original order
            return candidate_assessments[:top_k]
    
    def generate_explanation(self, job_description: str, assessment: Dict[str, Any]) -> str:
        """
        Generate a natural language explanation of why an assessment is recommended.
        
        Args:
            job_description: The original job description
            assessment: The assessment that was recommended
            
        Returns:
            A natural language explanation
        """
        # Use the template from config
        prompt = PROMPT_TEMPLATES["generate_explanation"].format(
            job_description=job_description,
            assessment_name=assessment.get("name", ""),
            assessment_description=assessment.get("description", "No description available"),
            assessment_type=assessment.get("test_types", ""),
            assessment_duration=assessment.get("duration", "")
        )
        
        try:
            response = self.model.generate_content(prompt)
            # No need to parse JSON for this one - we want the raw text
            explanation = response.text.strip()
            return explanation
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return "This assessment matches key requirements in the job description."