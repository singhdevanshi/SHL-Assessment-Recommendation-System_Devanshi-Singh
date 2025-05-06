import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from sentence_transformers import SentenceTransformer
import json
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src", "embeddings")))
from embeddings.faiss_wrapper import FaissIndexWrapper

# Function to evaluate the recommender system
def evaluate_recommender_system(test_data_path: str, faiss_index_path: str, k_values: list = [1, 3, 5, 10], ground_truth_path: str = "ground_truth.json"):
    """
    Evaluate the LLM recommender system using the provided test data and FAISS index.

    Args:
        test_data_path: Path to the CSV file containing test queries and assessments
        faiss_index_path: Path to the FAISS index file
        k_values: List of k values for Recall@K evaluation
        ground_truth_path: Path to the ground truth JSON file

    Returns:
        dict: Evaluation metrics (Recall@K and MAP@K)
    """

    # Load the test data
    test_data = pd.read_csv(test_data_path)

    # Load ground truth data
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    # Initialize the SentenceTransformer model for encoding queries
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize the FAISS index wrapper and load the FAISS index
    faiss_index = FaissIndexWrapper()
    faiss_index.load(faiss_index_path)

    # Initialize evaluation metrics
    metrics = {
        "recall": {k: [] for k in k_values},
        "map": {k: [] for k in k_values}
    }

    # Iterate through test queries and evaluate
    for index, row in test_data.iterrows():
        query = row['Name']
        true_assessments = row['Test Types'].split()  # Assuming multiple test types are space-separated
        ground_truth_urls = next((item['ground_truth_urls'] for item in ground_truth if item['query'] == query), None)

        if not ground_truth_urls:
            continue  # Skip queries with no ground truth data

        # Encode the query using the model
        query_embedding = model.encode(query)

        # DEBUG: Print the shape of the query embedding
        print(f"Query embedding shape before reshape: {query_embedding.shape}")
        query_embedding = np.reshape(query_embedding, (1, -1))  # Reshape if necessary (1, 384)
        print(f"Query embedding shape after reshape: {query_embedding.shape}")

        # Perform search using the FAISS index
        search_result = faiss_index.search(query_embedding, k=max(k_values))

        # DEBUG: Print details of the FAISS search result
        print(f"Search result: {search_result}")
        print(f"Search result indices length: {len(search_result.indices)}")
        print(f"Search result scores length: {len(search_result.scores)}")

        # Extract the top-k results
        top_k_indices = search_result.indices
        top_k_scores = search_result.scores

        # Convert top-k indices to the actual URLs from the FAISS index
        assessment_data = faiss_index.get_assessment_data()
        retrieved_urls = [assessment_data[i]['url'] for i in top_k_indices]

        # DEBUG: Print the retrieved URLs and their corresponding scores
        print(f"Retrieved URLs: {retrieved_urls}")
        print(f"Top-k Scores: {top_k_scores}")

        # Compute Recall@K and MAP@K
        for k in k_values:
            retrieved_at_k = retrieved_urls[:k]
            relevant_at_k = [1 if url in ground_truth_urls else 0 for url in retrieved_at_k]
            
            recall_at_k = np.mean(relevant_at_k)
            metrics['recall'][k].append(recall_at_k)

            # Compute MAP@K if there are any relevant URLs
            average_precision = average_precision_score([1 if url in ground_truth_urls else 0 for url in retrieved_urls[:k]], top_k_scores[:k])
            metrics['map'][k].append(average_precision)

    # Calculate the final average of each metric
    final_metrics = {
        "recall": {k: np.mean(metrics['recall'][k]) for k in k_values},
        "map": {k: np.mean(metrics['map'][k]) for k in k_values}
    }

    return final_metrics


# Main evaluation logic
if __name__ == "__main__":
    # Define paths
    test_data_path = "C:/Users/devanshi/SHL-Assessment-Recommendation-System_Devanshi-Singh/src/evaluation/test.csv"  # Path to your test CSV file
    faiss_index_path = os.path.join(os.getcwd(), "data", "embeddings", "faiss_index.faiss")  # Path to FAISS index
    ground_truth_path = "C:/Users/devanshi/SHL-Assessment-Recommendation-System_Devanshi-Singh/src/evaluation/ground_truth.json"  # Path to your ground truth JSON file

    # Run evaluation
    evaluation_results = evaluate_recommender_system(test_data_path, faiss_index_path, ground_truth_path=ground_truth_path)

    # Print evaluation results
    print("Evaluation Results:")
    for k in [1, 3, 5, 10]:
        print(f"Recall@{k}: {evaluation_results['recall'][k]:.4f}")
        print(f"MAP@{k}: {evaluation_results['map'][k]:.4f}")

    # Save evaluation results to a file
    with open("evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)

    print("Evaluation results saved to 'evaluation_results.json'.")