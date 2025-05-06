"""
FAISS vector index wrapper for semantic similarity search.
"""
import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple

class FaissIndex:
    """Wrapper for FAISS index to perform semantic similarity search on assessment embeddings."""
    
    def __init__(self):
        """Initialize an empty FAISS index."""
        self.index = None
        self.dimension = None
        self.assessment_data = []
        self.is_loaded = False
    
    def load(self, index_path: str, metadata_path: Optional[str] = None) -> bool:
        """
        Load a FAISS index from disk along with assessment metadata.
        
        Args:
            index_path: Path to the FAISS index (.faiss file)
            metadata_path: Path to assessment metadata (.json file), if None will try to infer
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            self.dimension = self.index.d
            self.is_loaded = True
            
            # Try to load metadata if path is provided or can be inferred
            if metadata_path is None:
                # Try to infer metadata path from index path
                base_dir = os.path.dirname(index_path)
                possible_paths = [
                    os.path.join(base_dir, "assessment_metadata.json"),
                    os.path.join(base_dir, "assessments.json"),
                    os.path.join(os.path.dirname(base_dir), "assessments", "assessments.json"),
                    os.path.join(os.path.dirname(os.path.dirname(base_dir)), "data", "assessments.json")
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        metadata_path = path
                        break
            
            # Load assessment metadata if available
            if metadata_path and os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Handle different possible formats
                    if isinstance(data, list):
                        self.assessment_data = data
                    elif isinstance(data, dict) and "assessments" in data:
                        self.assessment_data = data["assessments"]
                    else:
                        self.assessment_data = list(data.values()) if isinstance(data, dict) else []
                        
                    print(f"Loaded {len(self.assessment_data)} assessments from metadata")
                    
                    # Validate that we have the right number of assessments
                    expected_count = self.index.ntotal
                    if len(self.assessment_data) != expected_count:
                        print(f"Warning: Number of assessments ({len(self.assessment_data)}) doesn't match index size ({expected_count})")
                        
                except Exception as e:
                    print(f"Error loading assessment metadata: {e}")
                    # Create empty metadata entries if loading fails
                    self.assessment_data = [{"id": str(i), "name": f"Assessment {i}"} for i in range(self.index.ntotal)]
            else:
                print(f"No assessment metadata found at {metadata_path}")
                # Create empty metadata entries
                self.assessment_data = [{"id": str(i), "name": f"Assessment {i}"} for i in range(self.index.ntotal)]
            
            print(f"Successfully loaded FAISS index with {self.index.ntotal} vectors of dimension {self.dimension}")
            return True
            
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            self.index = None
            self.is_loaded = False
            self.assessment_data = []
            return False
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search the index for the k nearest neighbors to the query vector.
        
        Args:
            query_vector: Query embedding of shape (dimension,)
            k: Number of nearest neighbors to return
            
        Returns:
            List of dictionaries with assessment data and distances
        """
        if not self.is_loaded or self.index is None:
            print("Error: FAISS index not loaded")
            return []
        
        # Ensure query vector has the right shape and type
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)  # Convert to 2D
        
        # Convert to float32 if needed
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)
        
        # Check dimension
        if query_vector.shape[1] != self.dimension:
            print(f"Error: Query vector dimension ({query_vector.shape[1]}) doesn't match index dimension ({self.dimension})")
            return []
        
        # Limit k to the number of items in the index
        k = min(k, self.index.ntotal)
        
        # Perform search
        try:
            distances, indices = self.index.search(query_vector, k)
            
            # Get the actual assessment data
            results = []
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.assessment_data):
                    # Create a copy of the assessment data
                    result = dict(self.assessment_data[idx])
                    # Add the distance score
                    result["vector_distance"] = float(distances[0][i])
                    results.append(result)
                else:
                    print(f"Warning: Index {idx} out of range for assessment data")
            
            return results
        except Exception as e:
            print(f"Error during FAISS search: {e}")
            return []
    
    def save(self, index_path: str, metadata_path: Optional[str] = None) -> bool:
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save the FAISS index
            metadata_path: Path to save assessment metadata, if None will use default
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.is_loaded or self.index is None:
            print("Error: No FAISS index to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save metadata if available
            if self.assessment_data and metadata_path:
                os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(self.assessment_data, f, indent=2)
                print(f"Saved {len(self.assessment_data)} assessments to metadata")
            
            print(f"Successfully saved FAISS index with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
            return False