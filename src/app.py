import os
import sys
from typing import List, Dict, Any, Optional
import logging

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import components
from embeddings.faiss_wrapper import FaissIndex
from llm.llm_recommender import LLMRecommender

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the FastAPI application"""
    
    # Load environment variables
    load_dotenv()
    
    # Define paths
    BASE_PATH = os.getenv("BASE_PATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INDEX_PATH = os.path.join(BASE_PATH, "data", "embeddings", "faiss_index.faiss")
    
    logger.info(f"Looking for FAISS index at: {INDEX_PATH}")
    logger.info(f"Index path exists: {os.path.exists(INDEX_PATH)}")
    
    # Initialize components
    index_loaded = False
    vector_index = FaissIndex()
    try:
        vector_index.load(INDEX_PATH)
        logger.info("Successfully loaded FAISS index")
        index_loaded = True
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {str(e)}")
        logger.info(f"Please generate and save a FAISS index to: {INDEX_PATH}")
        logger.info("The system will use fallback keyword search instead of vector search.")
    
    # Initialize LLM recommender with whatever index is available
    recommender = LLMRecommender(vector_index=vector_index if index_loaded else None)
    
    # Define request and response models
    class JobDescriptionRequest(BaseModel):
        query: str
        top_k: Optional[int] = 10
        rerank: Optional[bool] = True
        final_results: Optional[int] = 3
    
    class AssessmentResponse(BaseModel):
        url: str
        adaptive_support: str
        description: str
        duration: int
        remote_support: str
        test_type: List[str]
    
    class RecommendationResponse(BaseModel):
        recommended_assessments: List[AssessmentResponse]
    
    # Create FastAPI app
    app = FastAPI(
        title="SHL Assessment Recommender API",
        description="API for recommending SHL assessments based on job descriptions",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )
    
    @app.get("/")
    async def root():
        """Health check endpoint."""
        return {"status": "OK", "message": "SHL Assessment Recommender API is running"}
    
    @app.post("/recommend")
    async def recommend_assessments(request: JobDescriptionRequest):
        """
        Recommend SHL assessments based on a job description or query.
        
        Args:
            query: The job description or query text
            top_k: Number of initial candidates to retrieve from vector search
            rerank: Whether to apply LLM reranking
            final_results: Number of final results to return
        
        Returns:
            List of recommended assessments
        """
        try:
            # Extract job requirements first
            job_requirements = recommender.llm_client.extract_job_requirements(
                request.query
            )
            
            # Get recommendations
            recommendations = recommender.recommend(
                job_description=request.query,
                top_k=request.top_k,
                rerank=request.rerank,
                final_results=request.final_results
            )
            
            # Prepare response in the exact required format
            response = RecommendationResponse(
                recommended_assessments=[
                    AssessmentResponse(
                        url=rec.get("url", ""),
                        adaptive_support=rec.get("adaptive_support", "No"),
                        description=rec.get("description", ""),
                        duration=rec.get("duration", 0),
                        remote_support=rec.get("remote_support", "No"),
                        test_type=rec.get("test_type", [])
                    )
                    for rec in recommendations
                ]
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
    
    @app.post("/extract-requirements")
    async def extract_requirements(job_description: str = Body(..., embed=True)):
        """
        Extract structured job requirements from a job description.
        
        Args:
            job_description: The job description text
        
        Returns:
            Structured job requirements
        """
        try:
            job_requirements = recommender.llm_client.extract_job_requirements(job_description)
            return job_requirements
        
        except Exception as e:
            logger.error(f"Error extracting requirements: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error extracting requirements: {str(e)}")
    
    @app.get("/assessments")
    async def get_assessments(limit: int = 100, offset: int = 0):
        """
        Get a list of available assessments.
        
        Args:
            limit: Maximum number of assessments to return
            offset: Number of assessments to skip
        
        Returns:
            List of assessments
        """
        try:
            all_assessments = recommender.assessment_data
            paginated = all_assessments[offset:offset+limit]
            
            return {
                "total": len(all_assessments),
                "limit": limit,
                "offset": offset,
                "assessments": paginated
            }
        
        except Exception as e:
            logger.error(f"Error retrieving assessments: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error retrieving assessments: {str(e)}")
    
    return app
app = create_app()
