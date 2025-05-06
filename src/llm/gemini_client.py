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
        Parse JSON from the LLM response text, with improved error handling.
        
        Args:
            text: The text response from the LLM
            
        Returns:
            Parsed JSON as dict or list
        """
        # For debugging, log the raw response
        print(f"Raw response from Gemini: {text[:500]}...")
        
        # Strip markdown code blocks if present (more comprehensive pattern)
        text = re.sub(r'```(?:json|python)?\s*', '', text)
        text = re.sub(r'```', '', text)
        text = text.strip()
        
        # Try to parse the cleaned text directly
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"Initial JSON parsing failed: {e}")
            
            # More robust JSON object extraction
            json_pattern = r'(\{(?:[^{}]|(?R))*\})'
            matches = re.findall(json_pattern, text, re.DOTALL)
            
            if matches:
                for potential_json in matches:
                    try:
                        result = json.loads(potential_json)
                        print(f"Successfully extracted JSON object")
                        return result
                    except json.JSONDecodeError:
                        continue
            
            # Try for JSON arrays
            array_pattern = r'(\[(?:[^\[\]]|(?R))*\])'
            matches = re.findall(array_pattern, text, re.DOTALL)
            
            if matches:
                for potential_array in matches:
                    try:
                        result = json.loads(potential_array)
                        print(f"Successfully extracted JSON array")
                        return result
                    except json.JSONDecodeError:
                        continue
            
            # Last resort: try to fix common JSON syntax errors
            try:
                # Replace single quotes with double quotes
                fixed_text = text.replace("'", '"')
                return json.loads(fixed_text)
            except json.JSONDecodeError:
                pass
                
            print(f"Could not parse JSON from response. Returning empty result.")
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
        With improved response handling and debugging.
        
        Args:
            job_requirements: Structured job requirements
            candidate_assessments: List of candidate assessments from vector search
            top_k: Number of top assessments to return
            
        Returns:
            List of reranked assessments with relevance scores
        """
        # Format job requirements for the prompt
        job_req_text = json.dumps(job_requirements, indent=2)
        
        # Format assessments for the prompt with consistent keys
        assessments_text = ""
        for i, assessment in enumerate(candidate_assessments):
            # Ensure we have all the necessary keys with defaults
            name = assessment.get('name', '')
            desc = assessment.get('description', 'No description available')
            test_type = assessment.get('test_types', assessment.get('test_type', []))
            duration = assessment.get('duration', 0)
            remote = assessment.get('remote_testing', assessment.get('remote_support', 'No'))
            adaptive = assessment.get('adaptive_support', 'No')
            
            assessments_text += f"""
    Assessment {i+1}:
    Name: {name}
    Description: {desc}
    Type: {', '.join(test_type) if isinstance(test_type, list) else test_type}
    Duration: {duration} minutes
    Remote Testing: {remote}
    Adaptive Testing: {adaptive}
    """
        
        # Explicit prompt for reranking with specific output format
        prompt = f"""
    Given a job description and list of assessments, rerank the assessments based on relevance to the job.

    Job Requirements:
    {job_req_text}

    Available Assessments:
    {assessments_text}

    Instructions:
    1. Analyze how well each assessment matches the job requirements.
    2. Assign a relevance score (0-100) to each assessment.
    3. Provide a brief explanation of why each assessment is relevant.
    4. List specific job requirements that each assessment matches.

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

    Only include the JSON in your response, no additional text.
    """
        
        print(f"Sending reranking prompt:\n{prompt[:500]}...")
        
        try:
            response = self.model.generate_content(prompt)
            print(f"Raw reranking response: {response.text[:500]}...")
            
            parsed_response = self._parse_json_from_response(response.text)
            print(f"Parsed response: {parsed_response}")
            
            # Handle the case where parsing returns a dict instead of a list
            if isinstance(parsed_response, dict):
                if "results" in parsed_response:
                    parsed_response = parsed_response.get("results", [])
                # If it's a dict with assessment_index, wrap it in a list
                elif "assessment_index" in parsed_response:
                    parsed_response = [parsed_response]
            
            # Ensure we have a list
            if not isinstance(parsed_response, list):
                print("Warning: Expected list from reranking but got another type. Returning empty list.")
                parsed_response = []
            
            # Ensure each item has the expected fields
            reranked_assessments = []
            for item in parsed_response:
                if not isinstance(item, dict):
                    continue
                    
                assessment_idx = item.get("assessment_index", 0)
                # Adjust for 1-based indexing in the prompt
                if assessment_idx > 0 and assessment_idx <= len(candidate_assessments):
                    # Create a new dictionary to avoid reference issues
                    assessment = {}
                    # Copy the original assessment data
                    source_assessment = candidate_assessments[assessment_idx - 1]
                    for key, value in source_assessment.items():
                        assessment[key] = value
                    
                    # Add the reranking information
                    assessment.update({
                        "relevance_score": item.get("relevance_score", 0),
                        "explanation": item.get("explanation", "No explanation provided"),
                        "matched_requirements": item.get("matched_requirements", [])
                    })
                    
                    print(f"Reranked assessment {assessment_idx}: Score={assessment['relevance_score']}")
                    reranked_assessments.append(assessment)
            
            # Sort by relevance score and take top_k
            reranked_assessments.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            return reranked_assessments[:top_k]
        
        except Exception as e:
            print(f"Error reranking assessments: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to returning original assessments in original order
            print("Falling back to original ordering")
            return candidate_assessments[:top_k]
    
    def generate_explanation(self, job_description: str, assessment: Dict[str, Any], job_requirements: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a natural language explanation of why an assessment is recommended.
        Args:
            job_description: The original job description
            assessment: The assessment that was recommended
            job_requirements: Optional structured requirements extracted from the job description
        Returns:
            A natural language explanation
        """
        # Use job requirements if provided, otherwise fallback to job description
        job_info = json.dumps(job_requirements, indent=2) if job_requirements else job_description
        matched_reqs = assessment.get("matched_requirements", [])
        matched_reqs_text = "\n- " + "\n- ".join(matched_reqs) if matched_reqs else "N/A"
        
        # Enhanced prompt with more detailed instructions and examples
        prompt = f"""
You are an expert recruitment consultant specializing in assessment selection.
Your task is to explain precisely how a specific assessment tool aligns with job requirements.

### Job Requirements:
{job_info}

### Assessment Details:
Name: {assessment.get("name", "")}
Type: {assessment.get("test_types", "")}
Duration: {assessment.get("duration", "")} minutes
Description: {assessment.get("description", "No description available")}

### Matched Requirements:
{matched_reqs_text}

### Instruction:
Write a detailed, concrete explanation (4-6 sentences) of why this specific assessment is an excellent match for this role. 
Your explanation must:
1. Name 2-3 specific skills measured by this assessment that directly relate to job requirements
2. Explain how these skills connect to actual job responsibilities
3. Mention a business benefit of using this assessment (e.g., reduced turnover, better performance)
4. Use HR/recruitment professional language but avoid generic statements

Avoid vague phrases like "this assessment evaluates candidate abilities" or "measures important skills."
Instead, be specific about WHICH abilities and HOW they relate to the job.

### Explanation:
"""
    
        # Increase temperature slightly for more varied responses
        generation_config = {
            "temperature": 0.4,  # Higher than default to avoid generic responses
            "max_output_tokens": 1024
        }
        
        try:
            # Use custom generation config just for this call
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            explanation = response.text.strip()
            return explanation
        except Exception as e:
            print(f"Error generating explanation: {e}")
            return "This assessment matches key requirements in the job description."