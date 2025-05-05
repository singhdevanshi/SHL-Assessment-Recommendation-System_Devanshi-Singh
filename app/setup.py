"""
Setup script for the SHL Assessment Recommendation System.
"""

from setuptools import setup, find_packages

setup(
    name="shl-assessment-recommender",
    version="0.1.0",
    description="A system for recommending SHL assessments based on job descriptions",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "python-dotenv",
        "faiss-cpu",
        "streamlit",
        "requests",
        "pandas",
        "numpy",
        "matplotlib",
        "tqdm",
        "beautifulsoup4",  # If using BeautifulSoup for scraping
        "sentence-transformers",  # If using SentenceTransformers for embeddings
    ],
    python_requires=">=3.8",
)