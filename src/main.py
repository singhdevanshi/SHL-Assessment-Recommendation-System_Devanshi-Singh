"""
Main entry point for the SHL Assessment Recommendation System.
"""

import os
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_api():
    """Run the FastAPI server."""
    from src.api.app import app
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.getenv("API_PORT", 8000))
    
    print(f"Starting API server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

def run_ui():
    """Run the Streamlit UI."""
    import subprocess
    import sys
    
    # Get port from environment variable or use default
    port = int(os.getenv("UI_PORT", 8501))
    
    print(f"Starting Streamlit UI on port {port}...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "src/ui/streamlit_app.py",
        "--server.port", str(port)
    ])

def run_evaluation(sample_size=None):
    """Run the evaluation pipeline."""
    from src.embeddings.faiss_wrapper import FaissIndex
    from src.llm.llm_recommender import LLMRecommender
    from src.evaluation.evaluator import RecommenderEvaluator
    
    # Define paths
    base_path = os.getenv("BASE_PATH", "C:/Users/devanshi/SHL-Assessment-Recommendation-System_Devanshi-Singh")
    index_path = f"{base_path}/data/embeddings/faiss_index"
    eval_data_path = os.getenv("EVAL_DATA_PATH", f"{base_path}/data/evaluation/eval_data.csv")
    
    print("Initializing components...")
    vector_index = FaissIndex()
    vector_index.load(index_path)
    recommender = LLMRecommender(vector_index=vector_index)
    
    # Initialize evaluator
    evaluator = RecommenderEvaluator(
        recommender=recommender,
        eval_data_path=eval_data_path
    )
    
    # Run evaluation
    print("Running evaluation...")
    if sample_size:
        sample_size = int(sample_size)
    evaluator.evaluate(sample_size=sample_size)

def main():
    """Parse arguments and run the appropriate component."""
    parser = argparse.ArgumentParser(description="SHL Assessment Recommendation System")
    parser.add_argument("--component", choices=["api", "ui", "eval", "all"], 
                      default="api", help="Component to run")
    parser.add_argument("--sample-size", type=int, default=None, 
                      help="Sample size for evaluation")
    
    args = parser.parse_args()
    
    if args.component == "api" or args.component == "all":
        run_api()
    
    if args.component == "ui" or args.component == "all":
        run_ui()
    
    if args.component == "eval":
        run_evaluation(args.sample_size)
    
    if args.component == "all":
        run_evaluation(args.sample_size)

if __name__ == "__main__":
    main()