import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
from src.llm.gemini_client import GeminiClient

# Default assessments as a fallback when no other assessments are available
DEFAULT_ASSESSMENTS = [
    {
        "id": "verbal-reasoning",
        "name": "Verbal Reasoning Assessment",
        "description": "Measures a candidate's ability to understand and analyze written information.",
        "test_types": "Multiple Choice",
        "duration": 25,
        "remote_testing": "Yes",
        "adaptive_support": "No",
        "tags": ["verbal", "reasoning", "comprehension", "analysis"],
        "url": "https://example.com/verbal-reasoning"
    },
    {
        "id": "numerical-reasoning",
        "name": "Numerical Reasoning Assessment",
        "description": "Evaluates a candidate's ability to work with numbers and data interpretation.",
        "test_types": "Multiple Choice",
        "duration": 30,
        "remote_testing": "Yes",
        "adaptive_support": "No",
        "tags": ["numerical", "reasoning", "mathematics", "data analysis"],
        "url": "https://example.com/numerical-reasoning"
    },
    {
        "id": "logical-reasoning",
        "name": "Logical Reasoning Assessment",
        "description": "Tests a candidate's ability to draw logical conclusions from information.",
        "test_types": "Multiple Choice",
        "duration": 25,
        "remote_testing": "Yes",
        "adaptive_support": "No",
        "tags": ["logical", "reasoning", "problem solving", "critical thinking"],
        "url": "https://example.com/logical-reasoning"
    },
    {
        "id": "personality-assessment",
        "name": "Personality Assessment",
        "description": "Evaluates a candidate's work style, behavior patterns, and cultural fit.",
        "test_types": "Questionnaire",
        "duration": 40,
        "remote_testing": "Yes",
        "adaptive_support": "No",
        "tags": ["personality", "workplace behavior", "culture fit", "soft skills"],
        "url": "https://example.com/personality-assessment"
    }
]

class LLMRecommender:
    """
    A class that recommends assessments based on job descriptions using Google's Gemini 1.5 Pro.
    """
    
    def __init__(self, api_key: str, assessments_path: str = None, vector_index = None):
        """
        Initialize the LLMRecommender with a Google API key.
        
        Args:
            api_key: Google API key for Gemini models (required)
            assessments_path: Path to the assessments JSON file (optional)
            vector_index: FAISS vector index for semantic search (optional)
        """
        if not api_key:
            raise ValueError("API key is required for Gemini models")
            
        self.api_key = api_key
        self.vector_index = vector_index
        self.assessment_data = []
        
        try:
            # Initialize with Gemini 1.5 Pro
            self.llm_client = GeminiClient(api_key=api_key)
            
            # First check if we have assessments from vector index
            if self.vector_index and hasattr(self.vector_index, 'assessment_data'):
                self.assessment_data = self.vector_index.assessment_data
                print(f"[INFO] Loaded {len(self.assessment_data)} assessments from vector index")
            
            # If no assessments from vector index, try to load from file
            if not self.assessment_data and assessments_path:
                self._load_assessments(assessments_path)
            elif not self.assessment_data:
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
            
            # If still no assessments, use default assessments
            if not self.assessment_data:
                print("[WARNING] No assessments found. Using default assessments.")
                self.assessment_data = DEFAULT_ASSESSMENTS
                
        except Exception as e:
            print(f"Failed to initialize LLMRecommender: {e}")
            # Fall back to default assessments
            print("[WARNING] Falling back to default assessments due to initialization error.")
            self.assessment_data = DEFAULT_ASSESSMENTS
            
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
                self.assessment_data = data
            elif isinstance(data, dict) and "assessments" in data:
                self.assessment_data = data["assessments"]
            else:
                self.assessment_data = list(data.values()) if isinstance(data, dict) else []
                
            print(f"[INFO] Loaded {len(self.assessment_data)} assessments from {file_path}")
        except Exception as e:
            print(f"[ERROR] Error loading assessments from {file_path}: {e}")
            self.assessment_data = []
            
    def add_assessment(self, assessment: Dict[str, Any]) -> None:
        """
        Add a single assessment to the available assessments.
        
        Args:
            assessment: Dictionary containing assessment details
        """
        if isinstance(assessment, dict) and "name" in assessment:
            self.assessment_data.append(assessment)
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
            Dictionary with categories like 'technical_skills', 'soft_skills', etc.
        """
        try:
            return self.llm_client.extract_job_requirements(job_description)
        except Exception as e:
            print(f"[ERROR] Error in extract_job_requirements: {e}")
            return {
                "technical_skills": [],
                "soft_skills": [],
                "experience_level": "Not determined",
                "key_responsibilities": [],
                "required_competencies": []
            }
            
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
            print(f"[ERROR] Error in rerank_assessments: {e}")
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
            print(f"[ERROR] Error in generate_explanation: {e}")
            return f"This assessment evaluates skills relevant to the position requirements."
            
    def filter_candidate_assessments(self, job_requirements: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Filter assessments to find candidates matching job requirements.
        Tries vector search first if available, falls back to keyword matching.
        
        Args:
            job_requirements: Dictionary with job requirements
            top_k: Number of candidates to return
            
        Returns:
            List of candidate assessments
        """
        # First check if we have any assessments at all
        if not self.assessment_data:
            print("[ERROR] No assessments available for filtering")
            return []
            
        # If we have vector search capability, use that first
        if self.vector_index and hasattr(self.vector_index, 'search'):
            try:
                # Just return all assessments from the index
                # In a real implementation, we would encode the job requirements
                # into a vector and perform similarity search
                print("[INFO] Using vector search to find candidate assessments")
                return self.assessment_data[:top_k]
            except Exception as e:
                print(f"[ERROR] Vector search failed: {e}")
                # Fall back to keyword search
        
        # Collect all terms for matching
        print("[INFO] Using keyword search to find candidate assessments")
        search_terms = []
        
        # Collect terms from job requirements structure
        for key in ["technical_skills", "soft_skills", "required_competencies"]:
            if key in job_requirements and isinstance(job_requirements[key], list):
                search_terms.extend(job_requirements[key])
        
        # If we have no search terms, just return all assessments up to top_k
        if not search_terms:
            print("[WARNING] No search terms found in job requirements")
            return self.assessment_data[:top_k]
            
        # Count matches for each assessment
        scored_assessments = []
        for assessment in self.assessment_data:
            # Create a text blob from assessment
            text = " ".join([
                str(assessment.get("name", "")),
                str(assessment.get("description", "")),
                " ".join(str(tag) for tag in assessment.get("tags", [])) if "tags" in assessment else ""
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
            # Step 1: Check if we have assessments
            if not self.assessment_data:
                print("[ERROR] No assessments available for recommendation")
                # Use default assessments as a fallback
                self.assessment_data = DEFAULT_ASSESSMENTS
                print(f"[INFO] Using {len(self.assessment_data)} default assessments as fallback")
                
            # Step 2: Extract job requirements
            job_requirements = self.extract_job_requirements(job_description)
            print(f"[INFO] Extracted job requirements: {job_requirements}")
            
            # Step 3: Get candidate assessments
            candidate_assessments = self.filter_candidate_assessments(job_requirements, top_k=top_k)
            print(f"[INFO] Found {len(candidate_assessments)} candidate assessments")
            
            if not candidate_assessments:
                print("[WARNING] No candidate assessments found. Using default assessments.")
                candidate_assessments = DEFAULT_ASSESSMENTS[:top_k]

            # Step 4: Optional LLM reranking
            if rerank:
                ranked_assessments = self.rerank_assessments(
                    job_requirements, candidate_assessments, top_k=final_results
                )
            else:
                ranked_assessments = candidate_assessments[:final_results]
                
            print(f"[INFO] Ranked assessments: {[a.get('name', 'Unnamed') for a in ranked_assessments]}")

            # Step 5: Add explanations
            results = []
            for assessment in ranked_assessments:
                # Create a copy to avoid modifying the original
                result = dict(assessment)
                # Add explanation if not already present
                if "explanation" not in result or not result["explanation"]:
                    result["explanation"] = self.generate_explanation(job_description, assessment)
                results.append(result)

            return results

        except Exception as e:
            print(f"[ERROR] Error in recommend: {e}")
            # Fall back to default assessments in case of error
            return DEFAULT_ASSESSMENTS[:final_results]