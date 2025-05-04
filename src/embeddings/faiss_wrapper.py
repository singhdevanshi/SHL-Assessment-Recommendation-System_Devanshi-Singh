"""
Wrapper for the FAISS module to provide a consistent interface for the LLM recommender.
This will adapt the existing faiss.py functionality to the FaissIndex class expected
by the LLM recommender.
"""

import numpy as np
from typing import List, Dict, Any, Union, Optional
import importlib

# Import the existing faiss module functions
from . import faiss_utils

class FaissIndexWrapper:
    """
    Wrapper class for FAISS functionality that conforms to the interface
    expected by the LLM recommender.
    """
    
    def __init__(self):
        """Initialize the wrapper."""
        # If the original module has an initialization function, call it
        if hasattr(faiss_utils, 'init') and callable(faiss_utils.init):
            faiss_utils.init()
        
        # Store internal state if needed
        self.index = None
        self.is_loaded = False
    
    def load(self, path: str) -> bool:
        """
        Load the FAISS index from disk.
        
        Args:
            path: Path to the index file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        # If the original module has a load function, call it
        if hasattr(faiss_utils, 'load_index') and callable(faiss_utils.load_index):
            self.index = faiss_utils.load_index(path)
            self.is_loaded = True
        elif hasattr(faiss_utils, 'load') and callable(faiss_utils.load):
            self.index = faiss_utils.load(path)
            self.is_loaded = True
        
        return self.is_loaded
    
    def search(self, query: Union[str, np.ndarray], k: int = 5) -> Any:
        """
        Search the index for similar items.
        
        Args:
            query: Text query or embedding vector
            k: Number of results to return
            
        Returns:
            Search results in the format used by the original module
        """
        # If query is a string, convert to embedding if needed
        if isinstance(query, str) and hasattr(faiss_utils, 'text_to_embedding'):
            query_vector = faiss_utils.text_to_embedding(query)
        else:
            query_vector = query
        
        # Use the appropriate search function
        if self.index is not None and hasattr(self.index, 'search'):
            # If index object has a search method
            return self.index.search(query_vector, k)
        elif hasattr(faiss_utils, 'search') and callable(faiss_utils.search):
            # If module has a search function
            return faiss_utils.search(query_vector, k)
        else:
            raise NotImplementedError("Search functionality not found in the faiss module")
    
    # Add any other methods that might be expected by the LLM recommender
    def add(self, vectors: np.ndarray, ids: Optional[List[int]] = None) -> bool:
        """
        Add vectors to the index.
        
        Args:
            vectors: Vectors to add
            ids: Optional IDs for the vectors
            
        Returns:
            True if successful
        """
        if self.index is not None and hasattr(self.index, 'add'):
            if ids is not None:
                self.index.add_with_ids(vectors, ids)
            else:
                self.index.add(vectors)
            return True
        elif hasattr(faiss_utils, 'add') and callable(faiss_utils.add):
            return faiss_utils.add(vectors, ids)
        else:
            raise NotImplementedError("Add functionality not found in the faiss module")

# Create an alias for backward compatibility
FaissIndex = FaissIndexWrapper