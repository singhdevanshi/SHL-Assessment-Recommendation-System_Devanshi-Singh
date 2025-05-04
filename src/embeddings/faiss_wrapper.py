"""
Wrapper for the FAISS module to provide a consistent interface for the LLM recommender.
This will adapt the existing faiss_utils.py functionality to the FaissIndex class expected
by the LLM recommender.
"""

import numpy as np
from typing import List, Dict, Any, Union, Optional
import importlib
from dataclasses import dataclass

# Import the existing faiss module functions
from . import faiss_utils

@dataclass
class SearchResult:
    """
    Standard search result structure to ensure consistent interface.
    """
    indices: List[int]
    scores: List[float]

class FaissIndexWrapper:
    """
    Wrapper class for FAISS functionality that conforms to the interface
    expected by the LLM recommender.
    """
    
    def __init__(self):
        """Initialize the wrapper."""
        # Initialize the faiss_utils module
        faiss_utils.init()
        
        # Store internal state
        self.is_loaded = True
    
    def load(self, path: str) -> bool:
        """
        Load the FAISS index from disk.
        
        Args:
            path: Path to the index file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            faiss_utils.load_index(path)
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            self.is_loaded = False
            return False
    
    def search(self, query: Union[str, np.ndarray], k: int = 5) -> SearchResult:
        """
        Search the index for similar items.
        
        Args:
            query: Text query or embedding vector
            k: Number of results to return
            
        Returns:
            SearchResult object with indices and scores
        """
        # Use the search function from faiss_utils
        result = faiss_utils.search(query, k)
        
        # Return in the expected format for LLMRecommender
        return SearchResult(
            indices=result["indices"].tolist(),  # Convert numpy array to list
            scores=result["scores"].tolist()
        )
    
    def add(self, vectors: np.ndarray, ids: Optional[List[int]] = None) -> bool:
        """
        Add vectors to the index.
        
        Args:
            vectors: Vectors to add
            ids: Optional IDs for the vectors
            
        Returns:
            True if successful
        """
        index = faiss_utils.get_index()
        if index is not None:
            if ids is not None:
                index.add_with_ids(vectors, np.array(ids))
            else:
                index.add(vectors)
            return True
        return False
    
    def get_assessment_data(self) -> List[Dict[str, Any]]:
        """
        Get assessment data in the format expected by LLMRecommender.
        
        Returns:
            List of assessment data dictionaries
        """
        df = faiss_utils.get_assessment_data()
        
        # Convert DataFrame to list of dictionaries with lowercase keys
        assessments = []
        for _, row in df.iterrows():
            assessment = {
                "name": row.get("Name", ""),
                "url": row.get("URL", ""),
                "duration": row.get("Duration (mins)", 0),
                "remote_testing": row.get("Remote Testing Support", ""),
                "adaptive_support": row.get("Adaptive/IRT Support", ""),
                "test_types": row.get("Test Types", "")
            }
            assessments.append(assessment)
        
        return assessments

# Create an alias for backward compatibility
FaissIndex = FaissIndexWrapper