import faiss
import numpy as np
import pandas as pd

# Load the FAISS index from the saved file
index_path = 'C:/Users/devanshi/SHL-Assessment-Recommendation-System_Devanshi-Singh/src/data/embeddings/faiss_index'

try:
    # Load the FAISS index
    index = faiss.read_index(index_path)
    print("FAISS index loaded successfully.")
except Exception as e:
    print(f"Error loading FAISS index: {e}")

# Load your data (this is the assessment data you want to query against)
data = pd.read_csv('C:/Users/devanshi/SHL-Assessment-Recommendation-System_Devanshi-Singh/data/processed/shl_product_catalog_ready_for_embedding.csv')

# Create a model for query encoding (you can use any model like SentenceTransformer)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to search the index with a query
def search_assessments(query, top_k=5):
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)  # Normalize the query just like we did with the index

    # Perform the search
    scores, indices = index.search(query_embedding, top_k)

    # Get the results from the dataset
    results = data.iloc[indices[0]].copy()
    results['similarity'] = scores[0]
    return results[['Name', 'URL', 'similarity']]

# Example search query
query = "entry level software engineering role"
results = search_assessments(query, top_k=5)

# Print the results
print("Search Results:")
print(results)
