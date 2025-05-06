from typing import List, Dict, Any, Optional, Union
import numpy as np
import logging
from .gemini_client import GeminiClient
from embeddings.faiss_wrapper import FaissIndex  # Import your vector search class

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

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
        # Get assessment data directly from the vector index
        self.assessment_data = vector_index.get_assessment_data()
        self.vector_index = vector_index
        self.llm_client = GeminiClient(api_key)

        logging.debug("LLMRecommender initialized.")

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
        logging.debug(f"Starting recommendation for job description: {job_description[:100]}...")

        # Step 1: Extract job requirements using LLM
        try:
            job_requirements = self.llm_client.extract_job_requirements(job_description)
            logging.debug(f"Job requirements extracted: {job_requirements}")
        except Exception as e:
            logging.error(f"Error extracting job requirements: {e}")
            return []

        # Step 2: Perform vector search to get initial candidates
        try:
            vector_results = self.vector_index.search(job_description, k=top_k)
            logging.debug(f"Vector search returned {len(vector_results.indices)} results.")
        except Exception as e:
            logging.error(f"Error during vector search: {e}")
            return []

        # Get the actual assessment data for the retrieved indices
        candidate_assessments = []
        for idx in vector_results.indices:
            if 0 <= idx < len(self.assessment_data):  # Check index bounds
                candidate_assessments.append(self.assessment_data[idx])
            else:
                logging.warning(f"Index {idx} out of bounds for assessment data.")

        # Early return if we couldn't find any candidates
        if not candidate_assessments:
            logging.warning("No candidates found after vector search.")
            return []

        # Add vector similarity scores to candidate assessments
        for i, assessment in enumerate(candidate_assessments):
            if i < len(vector_results.scores):
                assessment["vector_similarity"] = float(vector_results.scores[i])
                logging.debug(f"Added vector similarity score: {assessment['vector_similarity']}")

        # Step 3: Apply LLM reranking if requested
        if rerank:
            logging.debug("Applying LLM reranking.")
            # Add description if missing (required for reranking)
            for assessment in candidate_assessments:
                if "description" not in assessment:
                    assessment["description"] = f"Assessment for {assessment.get('name', '')}"

            try:
                reranked_assessments = self.llm_client.rerank_assessments(
                    job_requirements, 
                    candidate_assessments,
                    top_k=final_results
                )
                logging.debug("Reranking complete.")
            except Exception as e:
                logging.error(f"Error during reranking: {e}")
                reranked_assessments = sorted(
                    candidate_assessments, 
                    key=lambda x: x.get("vector_similarity", 0), 
                    reverse=True
                )[:final_results]  # Fallback to vector search ranking
        else:
            logging.debug("Skipping LLM reranking, using vector search ranking.")
            reranked_assessments = sorted(
                candidate_assessments, 
                key=lambda x: x.get("vector_similarity", 0), 
                reverse=True
            )[:final_results]

        # Step 4: Generate explanations for each recommendation
        recommendations = []
        for assessment in reranked_assessments:
            # Only generate explanations if not already present
            if "explanation" not in assessment:
                try:
                    explanation = self.llm_client.generate_explanation(job_description, assessment)
                    assessment["explanation"] = explanation
                    logging.debug(f"Generated explanation for assessment: {assessment.get('name')}")
                except Exception as e:
                    logging.error(f"Error generating explanation for assessment {assessment.get('name')}: {e}")
                    assessment["explanation"] = "No explanation available."
            
            recommendations.append(assessment)

        logging.debug(f"Returning {len(recommendations)} recommendations.")
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
        logging.debug(f"Starting batch recommendation for {len(job_descriptions)} job descriptions.")
        results = []
        for job_description in job_descriptions:
            recommendations = self.recommend(
                job_description,
                top_k=top_k,
                rerank=rerank,
                final_results=final_results
            )
            results.append(recommendations)
        
        logging.debug("Batch recommendation complete.")
        return results
