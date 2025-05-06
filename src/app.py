import os
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Import components
from embeddings.faiss_wrapper import FaissIndex
from src.llm.llm_recommender import LLMRecommender

def create_app():
    """Create and configure the FastAPI application."""

    # Load environment variables
    load_dotenv()

    # Get Gemini API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Gemini API key not found in environment (.env file).")

    # Define paths
    BASE_PATH = os.getenv("BASE_PATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INDEX_PATH = os.path.join(BASE_PATH, "data", "embeddings", "faiss_index.faiss")

    print(f"[INFO] Looking for FAISS index at: {INDEX_PATH}")
    print(f"[INFO] Path exists: {os.path.exists(INDEX_PATH)}")

    # Initialize components
    vector_index = FaissIndex()
    try:
        vector_index.load(INDEX_PATH)
        print("[INFO] Successfully loaded FAISS index")
    except Exception as e:
        print(f"[ERROR] Failed to load FAISS index: {str(e)}")
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
        print(f"[INFO] Please generate and save a FAISS index to: {INDEX_PATH}")

    # Initialize LLMRecommender with the Gemini API key
    recommender = LLMRecommender(api_key=api_key)

    # Pydantic models
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

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check
    @app.get("/")
    async def root():
        return {"status": "OK", "message": "SHL Assessment Recommender API is running"}

    # Recommendation endpoint
    @app.post("/recommend")
    async def recommend_assessments(request: JobDescriptionRequest):
        try:
            job_requirements = recommender.extract_job_requirements(request.job_description)
            recommendations = recommender.recommend(
                job_description=request.job_description,
                top_k=request.top_k,
                rerank=request.rerank,
                final_results=request.final_results
            )
            return RecommendationResponse(
                recommendations=[
                    AssessmentResponse(
                        name=r.get("name", ""),
                        url=r.get("url", ""),
                        duration=r.get("duration"),
                        remote_testing=r.get("remote_testing"),
                        adaptive_support=r.get("adaptive_support"),
                        test_types=r.get("test_types"),
                        explanation=r.get("explanation", ""),
                        relevance_score=r.get("relevance_score"),
                        matched_requirements=r.get("matched_requirements", [])
                    ) for r in recommendations
                ],
                job_requirements=job_requirements
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

    # Requirement extraction endpoint
    @app.post("/extract-requirements")
    async def extract_requirements(job_description: str = Body(..., embed=True)):
        try:
            return recommender.extract_job_requirements(job_description)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extracting requirements: {str(e)}")

    # All assessments (raw access)
    @app.get("/assessments")
    async def get_assessments(limit: int = 100, offset: int = 0):
        try:
            all_assessments = recommender.assessment_data
            paginated = all_assessments[offset:offset + limit]
            return {
                "total": len(all_assessments),
                "limit": limit,
                "offset": offset,
                "assessments": paginated
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error retrieving assessments: {str(e)}")

    return app
