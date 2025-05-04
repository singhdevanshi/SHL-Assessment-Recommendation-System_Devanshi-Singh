"""
FastAPI endpoint for the SHL Assessment Recommender System.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os

# Import the recommender system
from src.embeddings.faiss_wrapper import FaissIndex
from src.llm.llm_recommender import LLMRecommender

# Define base path
BASE_PATH = 'C:/Users/devanshi/SHL-Assessment-Recommendation-System_Devanshi-Singh'
INDEX_PATH = f'{BASE_PATH}/data/embeddings/faiss_index'

# Initialize FAISS index
index = FaissIndex()
if not index.load(INDEX_PATH):
    raise RuntimeError(f"Failed to load FAISS index from {INDEX_PATH}")

# Initialize LLM recommender
recommender = LLMRecommender(index)

# Create FastAPI app
app = FastAPI(
    title="SHL Assessment Recommender API",
    description="API for recommending SHL assessments based on job descriptions",
    version="1.0.0"
)

# Define request model
class RecommendationRequest(BaseModel):
    job_description: str
    top_k: Optional[int] = 10
    rerank: Optional[bool] = True
    final_results: Optional[int] = 3

# Define response model
class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get assessment recommendations based on a job description.
    """
    try:
        recommendations = recommender.recommend(
            job_description=request.job_description,
            top_k=request.top_k,
            rerank=request.rerank,
            final_results=request.final_results
        )
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)