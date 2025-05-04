"""
Gemini API client for the SHL Assessment Recommender.
This module handles all interactions with the Gemini API.
"""

import os
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv

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
        
        # Default model
        self.model = genai.GenerativeModel('gemini-1.5-pro')
    
    def extract_job_requirements(self, job_description: str) -> Dict[str, Any]:
        """
        Extract structured job requirements from a job description.
        
        Args:
            job_description: The job description text.
            
        Returns:
            A dictionary containing structured job requirements.
        """
        prompt = f"""
        Extract key job requirements and skills from the following job description.
        Format the output as JSON with these categories:
        - technical_skills: List of technical skills required
        - soft_skills: List of soft skills mentioned
        - experience_level: Entry/Mid/Senior
        - key_responsibilities: List of main job responsibilities
        - required_competencies: List of competencies that could be assessed
        
        Job Description:
        {job_description}
        """
        
        response = self.model.generate_content(prompt)
        
        # Parse the response - assuming it returns JSON-formatted text
        # In a production environment, add more robust parsing and error handling
        try:
            # This is a simplification - in reality you'll need to parse the text response
            # The actual implementation will depend on Gemini's response format
            return response.text
        except Exception as e:
            print(f"Error parsing Gemini response: {e}")
            return {"error": str(e)}
    
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
        # Prepare a prompt that asks Gemini to evaluate and rank assessments
        assessments_text = "\n\n".join([
            f"Assessment {i+1}:\nName: {assessment.get('name')}\n"
            f"Description: {assessment.get('description')}\n"
            f"Type: {assessment.get('type')}\n"
            f"Duration: {assessment.get('duration')}"
            for i, assessment in enumerate(candidate_assessments)
        ])
        
        prompt = f"""
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
        - assessment_index: The index of the assessment (1-based)
        - relevance_score: Numerical score 0-100
        - explanation: Brief explanation
        - matched_requirements: List of matched requirements
        
        Sort the assessments by relevance_score in descending order.
        """
        
        response = self.model.generate_content(prompt)
        
        # In a production environment, add more robust parsing and error handling
        try:
            # This is a simplification - you'll need to parse the JSON response
            parsed_response = response.text
            
            # Process the response to return the reranked assessments
            # This is a placeholder - actual implementation will depend on response format
            reranked_assessments = []
            
            # Actual implementation would parse the JSON and rerank based on scores
            # For now, we'll return a basic structure as an example
            
            return reranked_assessments[:top_k]
        except Exception as e:
            print(f"Error reranking assessments: {e}")
            return candidate_assessments[:top_k]  # Fallback to original ranking
    
    def generate_explanation(self, job_description: str, assessment: Dict[str, Any]) -> str:
        """
        Generate a natural language explanation of why an assessment is recommended.
        
        Args:
            job_description: The original job description
            assessment: The assessment that was recommended
            
        Returns:
            A natural language explanation
        """
        prompt = f"""
        Given this job description:
        {job_description}
        
        And this assessment:
        Name: {assessment.get('name')}
        Description: {assessment.get('description')}
        Type: {assessment.get('type')}
        Duration: {assessment.get('duration')}
        
        Explain in 2-3 sentences why this assessment would be appropriate for evaluating 
        candidates for this position. Focus on how the assessment measures skills and 
        competencies required for the job.
        """
        
        response = self.model.generate_content(prompt)
        return response.text