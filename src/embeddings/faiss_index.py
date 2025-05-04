import os
import faiss
import numpy as np
import pandas as pd

# Create directory for saving the index if it doesn't exist
os.makedirs('C:/Users/devanshi/SHL-Assessment-Recommendation-System_Devanshi-Singh/data/embeddings', exist_ok=True)

# Load your embeddings (assumed to be in a numpy array)
embeddings = np.load('C:/Users/devanshi/SHL-Assessment-Recommendation-System_Devanshi-Singh/data/embeddings/shl_name_embeddings.npy')

# Normalize embeddings for better performance (optional but recommended)
embeddings = embeddings.astype('float32')
faiss.normalize_L2(embeddings)

# Create the FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Using inner product for cosine similarity after normalization
index.add(embeddings)

# Save the index to a .faiss file
faiss.write_index(index, '../data/embeddings/faiss_index')
print("FAISS index saved at '../data/embeddings/faiss_index'")

# Optionally, print the number of vectors added to the index
print(f"FAISS index built with {index.ntotal} vectors.")
