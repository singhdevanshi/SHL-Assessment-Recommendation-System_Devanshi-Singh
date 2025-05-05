"""
Streamlit frontend for the SHL Assessment Recommendation System.
"""

import os
import sys
import streamlit as st
import requests
import json
import traceback
from dotenv import load_dotenv

# Add the project root to the Python path to make imports work properly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

# Define the API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="🎯",
    layout="wide"
)

st.title("SHL Assessment Recommender")
st.markdown("""
Enter a job description to get personalized SHL assessment recommendations.
""")

# Create input area
job_description = st.text_area("Job Description", height=200, 
                              placeholder="Paste a job description here...")

# Check if the API is running
api_status = ""
try:
    response = requests.get(f"{API_URL}/")
    if response.status_code == 200:
        api_status = "✅ API is running"
    else:
        api_status = f"⚠️ API returned status code {response.status_code}"
except Exception as e:
    api_status = f"❌ API not accessible: {str(e)}"

st.sidebar.write(api_status)

# Sidebar options
with st.sidebar:
    st.header("Options")
    top_k = st.slider("Initial candidates to retrieve", 5, 30, 10)
    rerank = st.checkbox("Apply LLM reranking", True)
    final_results = st.slider("Number of final results", 1, 10, 3)

# Submit button
if st.button("Get Recommendations"):
    if not job_description:
        st.error("Please enter a job description.")
    else:
        try:
            with st.spinner("Analyzing job description and finding the best assessments..."):
                # Prepare the request data
                data = {
                    "job_description": job_description,
                    "top_k": top_k,
                    "rerank": rerank,
                    "final_results": final_results
                }
                
                # Log the request details for debugging
                st.sidebar.write("Request details:")
                st.sidebar.json(data)
                
                # Make the API request
                response = requests.post(f"{API_URL}/recommend", json=data)
                
                # Log the response status for debugging
                st.sidebar.write(f"Response status: {response.status_code}")
                
                # Check if the request was successful
                if response.status_code == 200:
                    results = response.json()
                    
                    # Display job requirements
                    st.subheader("Extracted Job Requirements")
                    requirements = results["job_requirements"]
                    
                    # Format requirements for display
                    col1, col2 = st.columns(2)
        
        
                    
                    # Display recommendations
                    st.subheader("Recommended Assessments")
                    
                    for i, rec in enumerate(results["recommendations"]):
                        with st.expander(f"{i+1}. {rec['name']}", expanded=True):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"**Why this assessment:** {rec['explanation']}")
                                
                                if rec.get('matched_requirements'):
                                    st.markdown("**Matched Requirements:**")
                                    for req in rec['matched_requirements']:
                                        st.markdown(f"- {req}")
                            
                            with col2:
                                st.markdown(f"**Relevance Score:** {rec.get('relevance_score', 'N/A')}")
                                st.markdown(f"**Duration:** {rec.get('duration', 'N/A')}")
                                st.markdown(f"**Remote Testing:** {rec.get('remote_testing', 'N/A')}")
                                st.markdown(f"**Adaptive Support:** {rec.get('adaptive_support', 'N/A')}")
                                st.markdown(f"**Test Types:** {rec.get('test_types', 'N/A')}")
                                st.markdown(f"[View Assessment]({rec['url']})")
                else:
                    st.error(f"API Error: {response.status_code} {response.reason}")
                    try:
                        st.json(response.json())
                    except:
                        st.write(response.text)
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error("Make sure the API server is running at " + API_URL)
            st.error("Exception details:")
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("SHL Assessment Recommendation System")