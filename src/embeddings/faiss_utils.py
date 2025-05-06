import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

# Define paths consistently
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(BASE_PATH, 'data', 'processed', 'shl_product_catalog_ready_for_embedding.csv')
EMBEDDINGS_PATH = os.path.join(BASE_PATH, 'data', 'embeddings', 'shl_name_embeddings.npy')
INDEX_PATH = os.path.join(BASE_PATH, 'data', 'embeddings', 'faiss_index.faiss')

# Initialize variables
data = None
embeddings = None
index = None
model = None

def init():
    """
    Initialize the FAISS utils module, loading data, embeddings and model.
    """
    global data, embeddings, index, model
    
    # Load data
    data = pd.read_csv(DATA_PATH)
    
    # Load embeddings
    embeddings = np.load(EMBEDDINGS_PATH).astype('float32')
    
    # Normalize and initialize index
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Cosine similarity (after normalization)
    index.add(embeddings)
    
    # Load sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return True

def text_to_embedding(text: str) -> np.ndarray:
    """
    Converts a text query to a normalized embedding vector.
    """
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
    embedding = model.encode([text])
    faiss.normalize_L2(embedding)  # Normalize to match FAISS index
    return embedding.astype('float32')

def search(query, top_k=5) -> dict:
    """
    Search the FAISS index for similar items based on text or vector.
    
    Args:
        query (str or np.ndarray): Text or embedding
        top_k (int): Number of top results

    Returns:
        dict with 'scores' and 'indices' arrays
    """
    global index
    if index is None:
        load_index(INDEX_PATH)
    
    if isinstance(query, str):
        query = text_to_embedding(query)
    
    # Search the FAISS index
    scores, indices = index.search(query, top_k)
    
    # Debugging: Print scores and indices for inspection
    print(f"Query: {query}")
    print(f"Scores: {scores}")
    print(f"Indices: {indices}")
    
    return {
        "scores": scores[0],
        "indices": indices[0]
    }

def search_assessments(query, top_k=5) -> pd.DataFrame:
    """
    Return assessment rows with similarity scores for a given query.
    
    Args:
        query (str): Text query
        top_k (int): Top results to return
        
    Returns:
        pd.DataFrame with Name, URL, and similarity
    """
    global data
    if data is None:
        data = pd.read_csv(DATA_PATH)
        
    search_results = search(query, top_k)
    
    # Debugging: Print the results for inspection
    print(f"Search results: {search_results}")
    
    # Get the corresponding rows from data and add similarity scores
    results = data.iloc[search_results["indices"]].copy()
    results['similarity'] = search_results["scores"]
    
    # Debugging: Print the final DataFrame with similarities
    print(f"Results with similarity: {results[['Name', 'URL', 'similarity']]}")
    
    return results[['Name', 'URL', 'similarity']]

def save_index(path: str = INDEX_PATH):
    """
    Save the current FAISS index to disk.
    """
    global index
    if index is None:
        raise ValueError("Index not initialized. Call init() first.")
    
    faiss.write_index(index, path)
    return True

def load_index(path: str = INDEX_PATH):
    """
    Load a FAISS index from disk.
    
    Returns:
        The loaded index
    """
    global index
    try:
        index = faiss.read_index(path)
        return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None

def get_index():
    """
    Return the current in-memory FAISS index.
    """
    global index
    return index

def get_assessment_data():
    """
    Return the loaded assessment data.
    """
    global data
    if data is None:
        data = pd.read_csv(DATA_PATH)
    return data
