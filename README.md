# 📚 AI Research Paper Assistant

An intelligent, full-stack Academic Research Assistant powered by Retrieval-Augmented Generation (RAG). This platform allows users to upload research papers (PDFs), automatically extracts and refines sections, generates targeted summaries, and provides an interactive Q&A chat interface with source citations. 

The project features a premium glassmorphic frontend, a robust Flask backend handling Groq API key rotation for rate limits, and an extensive suite of single and multi-document evaluation pipelines measuring LLM and RAG performance metrics (BERTScore, ROUGE, LLM-as-a-judge).

## ✨ Features

- **📄 Dynamic PDF Parsing:** Upload research PDFs; the system extracts content and semantically divides it into logical sections (Abstract, Methodology, Results, etc.) using LLMs.
- **🎯 Section-Specific Summaries:** Click on any extracted section to instantly generate a detailed, easy-to-understand summary.
- **💬 RAG-Powered Q&A Chat:** Ask questions about the paper. The system retrieves relevant context, provides accurate answers, and cites the exact snippets it used.
- **🔄 Robust API Handling:** Built-in rate limit handling with automatic API key rotation to ensure continuous availability.
- **📊 Comprehensive Evaluation Suite:** Includes scripts for evaluating single-doc and multi-doc performance, generating comparative tables, and plotting conference-ready visualizations (bar charts, trend lines) of system accuracy.
- **🎨 Premium UI/UX:** A stunning, responsive frontend utilizing modern Emerald & Slate color themes with glassmorphism design.

## 🛠️ Tech Stack

- **Frontend:** HTML, CSS, Vanilla JavaScript (Premium Glassmorphism UI)
- **Backend:** Python, Flask
- **LLM Engine:** ChatGroq (Extremely fast inference)
- **Embeddings:** HuggingFaceEmbeddings (`all-MiniLM-L6-v2` or similar)
- **RAG & Vector Database:** LangChain
- **Evaluation:** BERTScore, ROUGE, and custom LLM-as-a-judge metrics

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- [Git](https://git-scm.com/)
- A Groq API Key (system supports two keys for rate-limit rotation)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd AI-Research-Paper-Assistant
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your Groq credentials and model preferences:
   ```env
   GROQ_API_KEY=your_primary_groq_api_key_here
   GROQ_API_KEY_2=your_secondary_groq_api_key_here
   LLM_MODEL=llama3-8b-8192  # Or your preferred model
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   ```

5. **Run the Application:**
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000` to interact with the assistant.

## 📈 Evaluation & Visualizations

The project comes with a rigorous evaluation pipeline located in the root directory:
- `evaluate_pipeline.py` & `evaluate_single_doc_final.py`: Scripts to test the RAG accuracy against ground truth.
- `evaluate_multi_doc.py` & `generate_multi_doc_tables.py`: Generate professional tables evaluating multiple documents simultaneously.
- `generate_visualizations.py`: Generates professional charts (found in `Final_Evaluation_Results`).

To run evaluations:
```bash
python evaluate_pipeline.py
```



