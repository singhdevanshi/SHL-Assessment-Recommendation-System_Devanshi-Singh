"""
Main entry point for the SHL Assessment Recommendation System.
"""

import os
import sys
import argparse
import uvicorn
import subprocess

# Add the project root to the Python path to make imports work properly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Handle command line arguments
parser = argparse.ArgumentParser(description='Run components of the SHL Assessment Recommendation System')
parser.add_argument('--component', type=str, choices=['api', 'ui'], default='api',
                    help='Component to run (api or ui)')
args = parser.parse_args()

# Import the app factory function
from src.app import create_app

# Create the FastAPI application
app = create_app()

if __name__ == "__main__":
    if args.component == 'api':
        print("Starting API server...")
        # Running with uvicorn directly
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif args.component == 'ui':
        print("Launching Streamlit UI...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "C:/Users/devanshi/SHL-Assessment-Recommendation-System_Devanshi-Singh/ui/streamlit_app.py"])