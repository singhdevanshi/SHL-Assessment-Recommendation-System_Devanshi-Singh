import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# === Load data and model ===
data = pd.read_csv(
    'C:/Users/devanshi/SHL-Assessment-Recommendation-System_Devanshi-Singh/data/processed/shl_product_catalog_ready_for_embedding.csv'
)
embeddings = np.load(
    'C:/Users/devanshi/SHL-Assessment-Recommendation-System_Devanshi-Singh/data/embeddings/shl_name_embeddings.npy'
).astype('float32')

# === Normalize and initialize index ===
faiss.normalize_L2(embeddings)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Cosine similarity (after normalization)
index.add(embeddings)

# === Load sentence transformer model ===
model = SentenceTransformer('all-MiniLM-L6-v2')


def init():
    """
    Optional init function to match expected interface.
    """
    pass


def text_to_embedding(text: str) -> np.ndarray:
    """
    Converts a text query to a normalized embedding vector.
    """
    embedding = model.encode([text])
    faiss.normalize_L2(embedding)
    return embedding.astype('float32')


def search(query, top_k=5) -> dict:
    """
    Search the FAISS index for similar items based on text or vector.
    
    Args:
        query (str or np.ndarray): Text or embedding
        top_k (int): Number of top results

    Returns:
        dict with 'scores' and 'indices'
    """
    if isinstance(query, str):
        query = text_to_embedding(query)
    
    scores, indices = index.search(query, top_k)
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
    query_embedding = text_to_embedding(query)
    scores, indices = index.search(query_embedding, top_k)
    results = data.iloc[indices[0]].copy()
    results['similarity'] = scores[0]
    return results[['Name', 'URL', 'similarity']]


def save(path: str):
    """
    Save the current FAISS index to disk.
    """
    faiss.write_index(index, path)


def load(path: str):
    """
    Load a FAISS index from disk.
    
    Returns:
        The loaded index
    """
    global index
    index = faiss.read_index(path)
    return index


def get_index():
    """
    Return the current in-memory FAISS index.
    """
    return index
