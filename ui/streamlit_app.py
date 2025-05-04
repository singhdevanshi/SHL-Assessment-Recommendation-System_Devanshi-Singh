"""
Streamlit UI for the SHL Assessment Recommendation System.
"""

import streamlit as st
import pandas as pd
import requests
import json
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Define constants
API_URL = os.getenv("API_URL", "http://localhost:8000")
SAMPLE_JOB_DESCRIPTIONS = [
    {
        "title": "Data Scientist",
        "description": """
        We are looking for a skilled Data Scientist to join our team. The ideal candidate should have:
        - Strong background in machine learning and statistical analysis
        - Experience with Python, R, or similar programming languages
        - Ability to work with large datasets and derive insights
        - Good communication skills to present findings to stakeholders
        - Bachelor's degree in Computer Science, Statistics, or related field
        - 2+ years of experience in data analysis or machine learning
        """
    },
    {
        "title": "Sales Manager",
        "description": """
        We are seeking an experienced Sales Manager to lead our sales team. Responsibilities include:
        - Managing a team of sales representatives
        - Developing sales strategies and targets
        - Building relationships with key clients
        - Forecasting and analyzing sales metrics
        - Bachelor's degree in Business Administration or related field
        - 5+ years of sales experience, with at least 2 years in a management role
        - Strong communication and leadership skills
        """
    },
    {
        "title": "Software Engineer",
        "description": """
        We're hiring a Software Engineer to join our development team. Requirements:
        - Strong knowledge of Java, Python, or JavaScript
        - Experience with web development frameworks
        - Understanding of databases and data structures
        - Ability to work in an agile environment
        - Bachelor's degree in Computer Science or related field
        - 3+ years of software development experience
        - Good problem-solving skills and attention to detail
        """
    }
]

def make_api_request(endpoint, data=None, method="POST"):
    """
    Make a request to the API.
    
    Args:
        endpoint: API endpoint
        data: Request data
        method: HTTP method
    
    Returns:
        API response
    """
    url = f"{API_URL}/{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, params=data)
        else:
            response = requests.post(url, json=data)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def display_recommendations(recommendations, job_requirements):
    """
    Display recommendations and job requirements.
    
    Args:
        recommendations: List of recommended assessments
        job_requirements: Extracted job requirements
    """
    # Display job requirements
    with st.expander("Extracted Job Requirements", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Technical Skills:**")
            if job_requirements.get("technical_skills"):
                for skill in job_requirements["technical_skills"]:
                    st.markdown(f"- {skill}")
            else:
                st.write("None identified")
                
            st.write("**Soft Skills:**")
            if job_requirements.get("soft_skills"):
                for skill in job_requirements["soft_skills"]:
                    st.markdown(f"- {skill}")
            else:
                st.write("None identified")
        
        with col2:
            st.write(f"**Experience Level:** {job_requirements.get('experience_level', 'Not specified')}")
            
            st.write("**Key Responsibilities:**")
            if job_requirements.get("key_responsibilities"):
                for resp in job_requirements["key_responsibilities"]:
                    st.markdown(f"- {resp}")
            else:
                st.write("None identified")
            
            st.write("**Required Competencies:**")
            if job_requirements.get("required_competencies"):
                for comp in job_requirements["required_competencies"]:
                    st.markdown(f"- {comp}")
            else:
                st.write("None identified")
    
    # Display recommendations
    st.subheader("Recommended Assessments")
    
    if not recommendations:
        st.warning("No recommendations found")
        return
    
    # Create tabs for each recommendation
    tabs = st.tabs([f"{i+1}. {rec['name']}" for i, rec in enumerate(recommendations)])
    
    for i, (tab, rec) in enumerate(zip(tabs, recommendations)):
        with tab:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"### {rec['name']}")
                st.markdown(f"**Explanation:** {rec['explanation']}")
                
                if rec.get('matched_requirements'):
                    st.markdown("**Matched Requirements:**")
                    for req in rec['matched_requirements']:
                        st.markdown(f"- {req}")
            
            with col2:
                if rec.get('relevance_score') is not None:
                    st.metric("Relevance Score", f"{rec['relevance_score']:.0f}%")
                
                st.markdown(f"**Duration:** {rec.get('duration', 'N/A')} minutes")
                st.markdown(f"**Remote Testing:** {rec.get('remote_testing', 'N/A')}")
                st.markdown(f"**Adaptive Testing:** {rec.get('adaptive_support', 'N/A')}")
                st.markdown(f"**Test Types:** {rec.get('test_types', 'N/A')}")
            
            st.markdown(f"[View Assessment Details]({rec['url']})")
            
            # Add divider between tabs content
            if i < len(recommendations) - 1:
                st.divider()

def main():
    """Main function for the Streamlit app."""
    st.set_page_config(
        page_title="SHL Assessment Recommender",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("SHL Assessment Recommender")
    st.markdown("""
    This tool recommends SHL assessments based on job descriptions.
    Enter a job description or select a sample to get started.
    """)
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Configuration options
    top_k = st.sidebar.slider("Initial candidates", min_value=5, max_value=30, value=10, step=5,
                            help="Number of initial candidates to retrieve from vector search")
    
    rerank = st.sidebar.checkbox("Apply LLM reranking", value=True,
                              help="Whether to apply LLM reranking to improve results")
    
    final_results = st.sidebar.slider("Final results", min_value=1, max_value=10, value=3, step=1,
                                   help="Number of final results to return")
    
    # Sample job descriptions
    st.sidebar.header("Sample Job Descriptions")
    selected_sample = st.sidebar.selectbox(
        "Select a sample job description",
        [""] + [sample["title"] for sample in SAMPLE_JOB_DESCRIPTIONS]
    )
    
    # Input tab and Results tab
    tab1, tab2 = st.tabs(["Input", "Results"])
    
    with tab1:
        # Job description input
        if selected_sample:
            sample = next((s for s in SAMPLE_JOB_DESCRIPTIONS if s["title"] == selected_sample), None)
            if sample:
                job_description = st.text_area("Job Description", sample["description"], height=300)
            else:
                job_description = st.text_area("Job Description", "", height=300)
        else:
            job_description = st.text_area("Job Description", "", height=300)
        
        # Submit button
        if st.button("Get Recommendations", type="primary"):
            if not job_description:
                st.error("Please enter a job description")
            else:
                with st.spinner("Generating recommendations..."):
                    # Make API request
                    response = make_api_request("recommend", {
                        "job_description": job_description,
                        "top_k": top_k,
                        "rerank": rerank,
                        "final_results": final_results
                    })
                    
                    if response:
                        # Store results in session state for the Results tab
                        st.session_state.recommendations = response["recommendations"]
                        st.session_state.job_requirements = response["job_requirements"]
                        st.session_state.has_results = True
                        
                        # Switch to Results tab
                        st.experimental_rerun()
    
    with tab2:
        if st.session_state.get("has_results", False):
            display_recommendations(
                st.session_state.recommendations,
                st.session_state.job_requirements
            )
        else:
            st.info("Enter a job description and click 'Get Recommendations' to see results")

if __name__ == "__main__":
    main()