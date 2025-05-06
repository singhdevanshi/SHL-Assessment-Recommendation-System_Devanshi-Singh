# **AI-based Interview Question & Assessment Recommendation System**

## **Overview**
This project implements an intelligent recommendation system designed to match job role queries with the most relevant assessments from SHL’s product catalog. The system leverages advanced NLP techniques, semantic embeddings, and large language models (LLMs) to provide precise and explainable recommendations.

## **Key Features**
- **Accurate Assessment Matching**: Matches job role or skill queries to SHL assessments using semantic similarity and relevance-based criteria.
- **Explainable Recommendations**: Provides detailed, context-aware explanations for each assessment recommendation, focusing on skill-level alignment.
- **Optimized Query Processing**: Efficient handling of natural language queries with high confidence and precision.
- **Robust Evaluation**: Evaluated using standard metrics (Mean Recall@K, MAP@K) on a provided test set, achieving strong performance metrics.

## **Technologies Used**
- **Python**: Core language for the backend.
- **Streamlit**: Used for creating the interactive demo.
- **Selenium & BeautifulSoup**: Web scraping for extracting assessment metadata from SHL's product catalog.
- **scikit-learn**: For similarity and distance calculations.
- **FAISS**: Used for efficient retrieval of embeddings and assessment matching.
- **Hugging Face’s `all-MiniLM-L6-v2`**: Used for embedding-based semantic similarity.
- **Google Gemini 1.5 Pro**: Used for generating human-readable, context-aware explanations.

## **Setup Instructions**

### **Prerequisites**
- Python 3.8+
- Docker (optional, for containerization)
- Required libraries (listed below)

### **Installation**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/singhdevanshi/SHL-Assessment-Recommendation-System_Devanshi-Singh.git
   cd shl-assessment-recommendation

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Run the application:**
   ```bash
   streamlit run app.py

This will launch the interactive demo of the recommendation system in your browser.

## **Usage**
1. **Interactive Demo:** The system is designed to recommend assessments based on a user-provided query (e.g., a job role or skill set).
2. **Explainable Results:** For each recommendation, the system generates an explanation detailing which skills and requirements the assessment addresses.
3. **Evaluation:** The system’s performance is evaluated based on Mean Recall@K and MAP@K scores. These scores can be accessed via the evaluate.py script.

## **Optimizations & Enhancements**
1. **Query Processing:** Enhanced the extraction of high-confidence information from queries, reducing speculative or irrelevant results.
2. **Relevance Matching:** Applied stricter thresholds for similarity, ensuring more precise and accurate matches.
3. **Explanations:** Improved the explanation generation logic to focus on specific, skill-level matches and avoid generic responses.
4. **Code Optimization:** Streamlined the codebase by modularizing functions and eliminating redundancy, ensuring better maintainability and clarity.

## **Evaluation Metrics**
1. **MAP@K (Mean Average Precision at K)**
2. **Recall@K**

Achieved MAP@5: 0.95
Achieved Recall@5: 0.20

## **Contact**
For any questions or feedback, please contact me at devanshi075@gmail.com




