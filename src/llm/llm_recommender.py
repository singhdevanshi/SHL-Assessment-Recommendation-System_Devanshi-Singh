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
                 assessment_data: List[Dict[str, Any]],
                 vector_index: FaissIndex,
                 api_key: Optional[str] = None):
        """
        Initialize the LLM recommender.
        
        Args:
            assessment_data: List of assessment data dictionaries
            vector_index: Initialized vector index for semantic search
            api_key: Optional Gemini API key
        """
        self.assessment_data = assessment_data
        self.vector_index = vector_index
        self.llm_client = GeminiClient(api_key)
    
    def recommend(self, 
                  job_description: str, 
                  top_k: int = 10, 
                  rerank: bool = True,
                  final_results: int = 3) -> List[Dict[str, Any]]:
        """
        Generate assessment recommendations based on a job description.
        
        Args:
            job_description: The job description text
            top_k: Number of initial candidates to retrieve from vector search
            rerank: Whether to apply LLM reranking
            final_results: Number of final results to return
            
        Returns:
            List of recommended assessments with explanations
        """
        # Step 1: Extract job requirements using LLM
        job_requirements = self.llm_client.extract_job_requirements(job_description)
        
        # Step 2: Perform vector search to get initial candidates
        vector_results = self.vector_index.search(job_description, k=top_k)
        
        # Get the actual assessment data for the retrieved indices
        candidate_assessments = [self.assessment_data[idx] for idx in vector_results.indices]
        
        # Step 3: Apply LLM reranking if requested
        if rerank:
            reranked_assessments = self.llm_client.rerank_assessments(
                job_requirements, 
                candidate_assessments,
                top_k=final_results
            )
        else:
            # Use vector search ranking
            reranked_assessments = candidate_assessments[:final_results]
        
        # Step 4: Generate explanations for each recommendation
        recommendations = []
        for assessment in reranked_assessments:
            explanation = self.llm_client.generate_explanation(job_description, assessment)
            
            recommendations.append({
                **assessment,  # Include all assessment details
                "explanation": explanation
            })
        
        return recommendations
    
    def batch_recommend(self, 
                        job_descriptions: List[str], 
                        top_k: int = 10, 
                        rerank: bool = True,
                        final_results: int = 3) -> List[List[Dict[str, Any]]]:
        """
        Generate recommendations for multiple job descriptions.
        Useful for evaluation purposes.
        
        Args:
            job_descriptions: List of job description texts
            top_k: Number of initial candidates to retrieve from vector search
            rerank: Whether to apply LLM reranking
            final_results: Number of final results to return
            
        Returns:
            List of recommendation lists, one for each job description
        """
        results = []
        for job_description in job_descriptions:
            recommendations = self.recommend(
                job_description,
                top_k=top_k,
                rerank=rerank,
                final_results=final_results
            )
            results.append(recommendations)
        
        return results