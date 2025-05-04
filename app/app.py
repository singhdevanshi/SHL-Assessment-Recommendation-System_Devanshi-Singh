"""
FastAPI application for the SHL Assessment Recommendation System.
"""

import os
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Import our components
from src.embeddings.faiss_wrapper import FaissIndex
from src.llm.llm_recommender import LLMRecommender

# Load environment variables
load_dotenv()

# Define paths
BASE_PATH = os.getenv("BASE_PATH", "C:/Users/devanshi/SHL-Assessment-Recommendation-System_Devanshi-Singh")
INDEX_PATH = f"{BASE_PATH}/data/embeddings/faiss_index"

# Initialize components
vector_index = FaissIndex()
vector_index.load(INDEX_PATH)
recommender = LLMRecommender(vector_index=vector_index)

# Define request and response models
class JobDescriptionRequest(BaseModel):
    job_description: str
    top_k: Optional[int] = 10
    rerank: Optional[bool] = True
    final_results: Optional[int] = 3

class AssessmentResponse(BaseModel):
    name: str
    url: str
    duration: Optional[float] = None
    remote_testing: Optional[str] = None
    adaptive_support: Optional[str] = None
    test_types: Optional[str] = None
    explanation: Optional[str] = None
    relevance_score: Optional[float] = None
    matched_requirements: Optional[List[str]] = None

class RecommendationResponse(BaseModel):
    recommendations: List[AssessmentResponse]
    job_requirements: Dict[str, Any]

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

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(request: JobDescriptionRequest):
    """
    Recommend SHL assessments based on a job description.
    
    Args:
        job_description: The job description text
        top_k: Number of initial candidates to retrieve from vector search
        rerank: Whether to apply LLM reranking
        final_results: Number of final results to return
    
    Returns:
        List of recommended assessments with explanations
    """
    try:
        # Extract job requirements first
        job_requirements = recommender.llm_client.extract_job_requirements(
            request.job_description
        )
        
        # Get recommendations
        recommendations = recommender.recommend(
            job_description=request.job_description,
            top_k=request.top_k,
            rerank=request.rerank,
            final_results=request.final_results
        )
        
        # Convert to response model
        response = RecommendationResponse(
            recommendations=[
                AssessmentResponse(
                    name=rec.get("name", ""),
                    url=rec.get("url", ""),
                    duration=rec.get("duration"),
                    remote_testing=rec.get("remote_testing"),
                    adaptive_support=rec.get("adaptive_support"),
                    test_types=rec.get("test_types"),
                    explanation=rec.get("explanation", ""),
                    relevance_score=rec.get("relevance_score"),
                    matched_requirements=rec.get("matched_requirements", [])
                )
                for rec in recommendations
            ],
            job_requirements=job_requirements
        )
        
        return response
    
    except Exception as e:
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
        raise HTTPException(status_code=500, detail=f"Error retrieving assessments: {str(e)}")

if __name__ == "__main__":
    # Start the API server
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)