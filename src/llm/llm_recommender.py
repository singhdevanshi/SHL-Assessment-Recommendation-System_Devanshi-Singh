"""
LLM-enhanced recommender system for SHL assessments.
Combines vector search results with LLM reranking and explanation generation.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .gemini_client import GeminiClient
from ..embeddings.faiss_wrapper import FaissIndex  # Import your vector search class


class LLMRecommender:
    """
    LLM-enhanced recommender system that combines vector search with LLM reranking.
    """

    def __init__(self, 
                 vector_index: FaissIndex,
                 api_key: Optional[str] = None):
        """
        Initialize the LLM recommender.
        
        Args:
            vector_index: Initialized vector index for semantic search
            api_key: Optional Gemini API key
        """
        self.assessment_data = vector_index.get_assessment_data()
        self.vector_index = vector_index
        self.llm_client = GeminiClient(api_key)

    def recommend(self, 
                  job_description: str, 
                  top_k: int = 10, 
                  rerank: bool = True,
                  final_results: int = 3) -> List[Dict[str, Any]]:
        """
        Generate assessment recommendations based on a job description.
        """
        # Step 0: Validate input
        if not job_description or not job_description.strip():
            raise ValueError("Job description is empty. Please provide a valid job description.")

        # Step 1: Extract job requirements using LLM
        try:
            job_requirements = self.llm_client.extract_job_requirements(job_description)
            if not isinstance(job_requirements, dict):
                raise ValueError(f"LLM returned non-dict response: {job_requirements}")
        except Exception as e:
            print("[ERROR] Failed to extract job requirements from Gemini:", e)
            raise RuntimeError("Gemini failed to extract job requirements.")

        # Step 2: Perform vector search to get initial candidates
        vector_results = self.vector_index.search(job_description, k=top_k)

        # Step 2.5: Convert vector search indices to full assessment entries
        candidate_assessments = []
        for idx in vector_results.indices:
            if 0 <= idx < len(self.assessment_data):
                assessment = self.assessment_data[idx].copy()
                assessment["vector_similarity"] = float(vector_results.scores[idx])
                candidate_assessments.append(assessment)

        if not candidate_assessments:
            return []

        # Step 3: Add basic description if missing
        for assessment in candidate_assessments:
            if "description" not in assessment:
                assessment["description"] = f"Assessment for {assessment.get('name', '')}"

        # Step 4: Rerank using LLM (if enabled)
        if rerank:
            try:
                reranked_assessments = self.llm_client.rerank_assessments(
                    job_requirements=job_requirements,
                    candidate_assessments=[
                        self.format_assessment_for_prompt(a) for a in candidate_assessments
                    ],
                    top_k=final_results
                )
                # Map reranked data back to full original data
                name_to_full_data = {a["name"]: a for a in candidate_assessments}
                reranked_assessments = [
                    name_to_full_data[a["name"]] for a in reranked_assessments if a["name"] in name_to_full_data
                ]
            except Exception as e:
                print("[WARNING] Gemini reranking failed, falling back to vector-based sorting:", e)
                reranked_assessments = sorted(
                    candidate_assessments, 
                    key=lambda x: x.get("vector_similarity", 0), 
                    reverse=True
                )[:final_results]
        else:
            # Use vector-based sort fallback
            reranked_assessments = sorted(
                candidate_assessments, 
                key=lambda x: x.get("vector_similarity", 0), 
                reverse=True
            )[:final_results]

        # Step 5: Generate explanation for each recommended assessment
        recommendations = []
        for assessment in reranked_assessments:
            if "explanation" not in assessment:
                try:
                    formatted = self.format_assessment_for_prompt(assessment)
                    assessment["explanation"] = self.llm_client.generate_explanation(
                        job_description=job_description,
                        assessment=formatted
                    )
                except Exception as e:
                    print(f"[WARNING] Failed to generate explanation for {assessment.get('name')}: {e}")
                    assessment["explanation"] = "Explanation not available."
            recommendations.append(assessment)

        return recommendations

    def batch_recommend(self, 
                        job_descriptions: List[str], 
                        top_k: int = 10, 
                        rerank: bool = True,
                        final_results: int = 3) -> List[List[Dict[str, Any]]]:
        """
        Generate recommendations for multiple job descriptions.
        """
        results = []
        for job_description in job_descriptions:
            try:
                recommendations = self.recommend(
                    job_description,
                    top_k=top_k,
                    rerank=rerank,
                    final_results=final_results
                )
            except Exception as e:
                print(f"[ERROR] Recommendation failed for job description: {job_description[:60]}... -> {e}")
                recommendations = []
            results.append(recommendations)
        return results

    @staticmethod
    def format_assessment_for_prompt(assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert raw assessment dict to a structured format for LLM prompts.
        """
        return {
            "name": assessment.get("name", "Unknown Assessment"),
            "description": assessment.get("description", ""),
            "skills": assessment.get("skills", "Not specified"),
            "test_type": assessment.get("test_type", "Unknown"),
            "duration": f"{assessment.get('duration', 'Unknown')} mins",
            "remote": "Yes" if assessment.get("remote_testing_support") == "Yes" else "No",
            "adaptive": "Yes" if assessment.get("adaptive_testing_support") == "Yes" else "No",
        }
