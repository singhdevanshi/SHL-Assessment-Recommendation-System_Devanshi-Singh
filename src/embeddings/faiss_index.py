import os
import faiss
import numpy as np
import pandas as pd

# Define paths consistently
BASE_PATH = 'C:/Users/devanshi/SHL-Assessment-Recommendation-System_Devanshi-Singh'
EMBEDDINGS_DIR = f'{BASE_PATH}/data/embeddings'

# Create directory for saving the index if it doesn't exist
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Load your embeddings (assumed to be in a numpy array)
embeddings = np.load(f'{EMBEDDINGS_DIR}/shl_name_embeddings.npy')

# Normalize embeddings for better performance (optional but recommended)
embeddings = embeddings.astype('float32')
faiss.normalize_L2(embeddings)

# Create the FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Using inner product for cosine similarity after normalization
index.add(embeddings)

# Save the index to a .faiss file
index_path = f'{EMBEDDINGS_DIR}/faiss_index'
faiss.write_index(index, index_path)
print(f"FAISS index saved at '{index_path}'")

# Optionally, print the number of vectors added to the index
print(f"FAISS index built with {index.ntotal} vectors.")