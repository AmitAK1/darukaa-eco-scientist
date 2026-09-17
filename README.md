
# 🌍 Darukaa AI Environmental Scientist

A scientifically grounded, multi-metric RAG system built for the Darukaa.Earth Biodiversity Intelligence Challenge. 

Unlike generic LLM wrappers that rely on naive vector similarity, this system acts as a deterministic environmental state machine. It uses Reciprocal Rank Fusion (RRF) Hybrid Search to retrieve precise scientific evidence from authoritative FAO and IPCC datasets, ensuring all recommendations are backed by explicit causal reasoning.

## 🏗️ Architecture Overview

The system is decoupled into four highly specialized engines:

1. **State Tracker (Memory Engine):** Uses Gemini 3.6 Flash and strict Pydantic schemas to extract and persist environmental variables (Soil pH, Rainfall, Land Use) across multi-turn conversations.
2. **Clarification Engine:** A deterministic Python logic block that halts the LLM and asks targeted questions if fewer than 3 environmental variables are present, preventing hallucinations.
3. **Hybrid Retriever (Knowledge Engine):** Combines Dense Vector Search (`all-MiniLM-L6-v2`) with Full-Text Keyword Search (BM25 via Tantivy) using Reciprocal Rank Fusion. This guarantees the retrieval of highly specific biological terms (e.g., "mycorrhizal fungi") that dense vectors often ignore.
4. **Causal Reasoner (Reasoning Engine):** Forces the LLM to output a structured JSON "Reasoning Chain," proving it connected at least 3 variables before generating an intervention and a verifiable FAO/IPCC citation.

## 🗄️ Database & Schema

*   **Database:** `LanceDB` (Embedded). Chosen for zero-setup local deployment, allowing instant verification by judges without Docker or PostgreSQL port conflicts.
*   **Vector Space:** 384-dimensional embeddings stored alongside metadata and full-text indexes.
*   **Knowledge Schema:** 
    *   `chunk_id` (Unique identifier)
    *   `document_title` (e.g., *FAO Recarbonizing Global Soils*)
    *   `domains` (Array: Soil Health, Climate, Biodiversity, etc.)
    *   `pdf_page` & `printed_page` (Ensures exact citation traceability)
    *   `text` (600-character scientific chunks)

## 🚀 Local Setup

This project uses an embedded database and requires zero external infrastructure other than an API key.

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd darukaa-eco-scientist

2. **Create a virtual environment:**

    `Bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate

3. **Install dependencies:**
    Bash
    pip install -r requirements.txt
    Add your API Key:
    Create a .env file in the root directory and add a free Gemini API key:
    Plaintext
    GEMINI_API_KEY="your_api_key_here"
    
`
4. **Run the Application:**

    Bash
    streamlit run app.py


**🔄 CI/CD Notes**

For production deployment, this architecture is designed to be fully serverless. The LanceDB data folder (.lancedb) can be stored in an AWS S3 bucket or Google Cloud Storage. The FastAPI/Streamlit frontend can be deployed via GitHub Actions to Google Cloud Run or HuggingFace Spaces, triggering a fresh container build on every push to the main branch.


---
