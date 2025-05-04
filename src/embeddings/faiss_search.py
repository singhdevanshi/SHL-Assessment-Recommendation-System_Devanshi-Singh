import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Define paths consistently
BASE_PATH = 'C:/Users/devanshi/SHL-Assessment-Recommendation-System_Devanshi-Singh'
DATA_PATH = f'{BASE_PATH}/data/processed/shl_product_catalog_ready_for_embedding.csv'
INDEX_PATH = f'{BASE_PATH}/data/embeddings/faiss_index'

# Load the FAISS index from the saved file
try:
    # Load the FAISS index
    index = faiss.read_index(INDEX_PATH)
    print("FAISS index loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS index: {e}")
    exit(1)

# Load your data (this is the assessment data you want to query against)
try:
    data = pd.read_csv(DATA_PATH)
    print("Assessment data loaded successfully.")
except Exception as e:
    print(f"Error loading assessment data: {e}")
    exit(1)

# Create a model for query encoding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to search the index with a query
def search_assessments(query, top_k=5):
    # Convert query to embedding
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)  # Normalize the query just like we did with the index

    # Perform the search
    scores, indices = index.search(query_embedding, top_k)

    # Get the results from the dataset
    results = data.iloc[indices[0]].copy()
    results['similarity'] = scores[0]
    
    # Format results for display
    return results[['Name', 'URL', 'Duration (mins)', 'Remote Testing Support', 
                  'Adaptive/IRT Support', 'Test Types', 'similarity']]

if __name__ == "__main__":
    # Example search query
    query = "entry level software engineering role"
    results = search_assessments(query, top_k=5)

    # Print the results
    print("\nSearch Results:")
    print(results.to_string(index=False))