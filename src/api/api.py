from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()

# Load pre-trained model for encoding queries
model = SentenceTransformer('all-MiniLM-L6-v2')

class Query(BaseModel):
    text: str  # The query text

# Define a POST endpoint for querying
@app.post("/query/")
async def get_results(query: Query):
    # Encode the query using the SentenceTransformer model
    query_embedding = model.encode(query.text)
    
    # Simulate a search or recommendation system
    # For now, just return the encoded query as a placeholder for actual results
    return {"query": query.text, "embedding": query_embedding.tolist()}

# Run the server with: uvicorn api:app --reload
