import os
import json
from typing import List, Dict, Any
import numpy as np
from src.llm.gemini_client import GeminiClient

class LLMRecommender:
    """
    A class that recommends assessments based on job descriptions using Google's Gemini 1.5 Pro.
    """
    
    def __init__(self, api_key: str, assessments_path: str = None):
        """
        Initialize the LLMRecommender with a Google API key.
        
        Args:
            api_key: Google API key for Gemini models (required)
            assessments_path: Path to the assessments JSON file (optional)
        """
        if not api_key:
            raise ValueError("API key is required for Gemini models")
            
        self.api_key = api_key
        try:
            # Initialize with Gemini 1.5 Pro
            self.llm_client = GeminiClient(api_key=api_key)
            
            # Load assessments if path is provided
            self.assessments = []
            if assessments_path:
                self._load_assessments(assessments_path)
            else:
                # Try to find assessments in default locations
                base_path = os.getenv("BASE_PATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                default_paths = [
                    os.path.join(base_path, "data", "assessments.json"),
                    os.path.join(base_path, "data", "assessments", "assessments.json"),
                ]
                for path in default_paths:
                    if os.path.exists(path):
                        self._load_assessments(path)
                        break
                        
            if not self.assessments:
                print("[WARNING] No assessments loaded. Please provide valid assessments_path or add assessments manually.")
                
        except Exception as e:
            print(f"Failed to initialize LLMRecommender: {e}")
            raise
            
    def _load_assessments(self, file_path: str) -> None:
        """
        Load assessments from a JSON file.
        
        Args:
            file_path: Path to the JSON file containing assessments
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Handle different possible formats
            if isinstance(data, list):
                self.assessments = data
            elif isinstance(data, dict) and "assessments" in data:
                self.assessments = data["assessments"]
            else:
                self.assessments = list(data.values()) if isinstance(data, dict) else []
                
            print(f"Loaded {len(self.assessments)} assessments from {file_path}")
        except Exception as e:
            print(f"Error loading assessments from {file_path}: {e}")
            self.assessments = []
            
    def add_assessment(self, assessment: Dict[str, Any]) -> None:
        """
        Add a single assessment to the available assessments.
        
        Args:
            assessment: Dictionary containing assessment details
        """
        if isinstance(assessment, dict) and "name" in assessment:
            self.assessments.append(assessment)
        else:
            print("[ERROR] Assessment must be a dictionary with at least a 'name' field")
            
    def add_assessments(self, assessments: List[Dict[str, Any]]) -> None:
        """
        Add multiple assessments to the available assessments.
        
        Args:
            assessments: List of assessment dictionaries
        """
        for assessment in assessments:
            self.add_assessment(assessment)
            
    def extract_job_requirements(self, job_description: str) -> Dict[str, Any]:
        """
        Extract job requirements from a job description.
        
        Args:
            job_description: String containing the job description
            
        Returns:
            Dictionary with keys 'skills', 'experience', and 'technologies'
        """
        try:
            return self.llm_client.extract_job_requirements(job_description)
        except Exception as e:
            print(f"Error in extract_job_requirements: {e}")
            return {"skills": [], "experience": [], "technologies": []}
            
    def rerank_assessments(self,
                          job_requirements: Dict[str, Any],
                          candidate_assessments: List[Dict[str, Any]],
                          top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Rerank candidate assessments based on job requirements.
        
        Args:
            job_requirements: Dictionary with job requirements
            candidate_assessments: List of assessment dictionaries
            top_k: Number of top assessments to return
            
        Returns:
            List of dictionaries with top assessments
        """
        try:
            return self.llm_client.rerank_assessments(job_requirements, candidate_assessments, top_k)
        except Exception as e:
            print(f"Error in rerank_assessments: {e}")
            return candidate_assessments[:top_k] if candidate_assessments else []
            
    def generate_explanation(self, job_description: str, assessment: Dict[str, Any]) -> str:
        """
        Generate explanation for why an assessment matches a job description.
        
        Args:
            job_description: String containing the job description
            assessment: Dictionary with assessment details
            
        Returns:
            String explanation of the match
        """
        try:
            return self.llm_client.generate_explanation(job_description, assessment)
        except Exception as e:
            print(f"Error in generate_explanation: {e}")
            return "Unable to generate explanation due to an error."
            
    def filter_candidate_assessments(self, job_requirements: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Filter assessments to find candidates matching job requirements.
        This is a simple keyword-based matching as a fallback when no embedding index is available.
        
        Args:
            job_requirements: Dictionary with job requirements
            top_k: Number of candidates to return
            
        Returns:
            List of candidate assessments
        """
        if not self.assessments:
            print("[ERROR] No assessments available for filtering")
            return []
            
        # Collect all terms for matching
        search_terms = []
        for key in ["skills", "experience", "technologies"]:
            if key in job_requirements and isinstance(job_requirements[key], list):
                search_terms.extend(job_requirements[key])
                
        # Count matches for each assessment
        scored_assessments = []
        for assessment in self.assessments:
            # Create a text blob from assessment
            text = " ".join([
                str(assessment.get("name", "")),
                str(assessment.get("description", "")),
                " ".join(str(tag) for tag in assessment.get("tags", []))
            ]).lower()
            
            # Count matches
            matches = sum(1 for term in search_terms if term.lower() in text)
            scored_assessments.append((matches, assessment))
            
        # Sort by match count and return top_k
        scored_assessments.sort(reverse=True, key=lambda x: x[0])
        return [assessment for _, assessment in scored_assessments[:top_k]]

    def recommend(self,
                  job_description: str,
                  top_k: int = 10,
                  rerank: bool = True,
                  final_results: int = 3) -> List[Dict[str, Any]]:
        """
        Recommend assessments based on job description.

        Args:
            job_description: The full job description text.
            top_k: Number of initial candidates to consider.
            rerank: Whether to rerank using LLM.
            final_results: Number of final recommendations to return.

        Returns:
            A list of recommended assessment dicts.
        """
        try:
            # Check if we have assessments
            if not self.assessments:
                print("[ERROR] No assessments available for recommendation")
                return []
                
            # Step 1: Extract job requirements
            job_requirements = self.extract_job_requirements(job_description)
            print(f"[INFO] Extracted job requirements: {job_requirements}")
            
            # Step 2: Get candidate assessments
            candidate_assessments = self.filter_candidate_assessments(job_requirements, top_k=top_k)
            print(f"[INFO] Found {len(candidate_assessments)} candidate assessments")
            
            if not candidate_assessments:
                print("[WARNING] No candidate assessments found")
                return []

            # Step 3: Optional LLM reranking
            if rerank:
                ranked_assessments = self.rerank_assessments(
                    job_requirements, candidate_assessments, top_k=final_results
                )
            else:
                ranked_assessments = candidate_assessments[:final_results]
                
            print(f"[INFO] Ranked assessments: {[a.get('name', 'Unnamed') for a in ranked_assessments]}")

            # Step 4: Add explanations
            results = []
            for assessment in ranked_assessments:
                # Create a copy to avoid modifying the original
                result = dict(assessment)
                result["explanation"] = self.generate_explanation(job_description, assessment)
                results.append(result)

            return results

        except Exception as e:
            print(f"Error in recommend: {e}")
            return []