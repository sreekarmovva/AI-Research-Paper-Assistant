import os
import json
import time
import csv
from dotenv import load_dotenv

# Langchain & Groq
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from groq import RateLimitError

# BERTScore
from bert_score import score as bert_scorer

# Import existing pipeline functions
from src.load_and_extract_text import extract_text_from_pdf, extract_pdf_sections
from src.detect_and_split_sections import refine_sections, split_sections_with_content
from src.create_vector_db import create_vector_db
from src.RAG_retrival_chain import get_qa_chain

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
load_dotenv()

GROQ_KEYS = [
    k for k in [
        os.getenv("GROQ_API_KEY"),
        os.getenv("GROQ_API_KEY_2"),
    ] if k and k != "PASTE_YOUR_SECOND_KEY_HERE"
]

if not GROQ_KEYS:
    raise ValueError("No valid GROQ API keys found in .env!")

_key_index = 0
llm_model = "llama-3.3-70b-versatile"
embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def make_llm(key_idx=0):
    return ChatGroq(groq_api_key=GROQ_KEYS[key_idx], model_name=llm_model, temperature=0)

def invoke_with_rotation(func, *args, **kwargs):
    global _key_index
    while True:
        try:
            return func(*args, **kwargs)
        except RateLimitError as e:
            next_idx = _key_index + 1
            if next_idx >= len(GROQ_KEYS):
                raise RuntimeError("All Groq API keys hit rate limits.") from e
            print(f"\nRate limit hit - Rotating to Key {next_idx + 1}")
            _key_index = next_idx
            LLM_HOLDER[0] = make_llm(_key_index)
            print(f"Now using Key {next_idx + 1}. Retrying...")

LLM_HOLDER = [make_llm(0)]
EMBEDDER = HuggingFaceEmbeddings(model_name=embedding_model)

# SELECTED PDFs FOR MULTI-DOC EVALUATION
PDF_UPLOADS_DIR = "uploads"
# We'll pick 4 diverse papers to create a challenging multi-doc environment
EVAL_PAPERS = [
    "An_LLM-Driven_Chatbot_in_Higher_Education_for_Databases_and_Information_Systems.pdf",
    "A Knowledge Graph-based RAG for Cross-Document Information E.pdf",
    "AI-driven solution for analyzing and interpreting data from.pdf",
    "The State of the Art in Conversational Document Analysis A R.pdf"
]

# Use the same TEST_CASES for direct comparison (1-doc vs multi-doc)
TEST_CASES = [
    # --- IN-DOMAIN QUESTIONS (Testing RAG Retrieval & Accuracy) ---
    {"question": "What is the primary motivation for developing an LLM-driven chatbot for databases and information systems?", "reference": "The primary motivation is to enhance student learning experiences by providing personalized, 24/7 tutoring support, simplifying complex database concepts, and assisting with SQL query formulation and database design."},
    {"question": "How did the authors evaluate the effectiveness of the chatbot?", "reference": "The effectiveness was evaluated through student feedback surveys taking into account usability, measuring learning outcomes, and analyzing the accuracy of the chatbot's responses to curriculum-specific inquiries."},
    {"question": "What specific large language models were discussed or utilized in the chatbot's architecture?", "reference": "The system architecture leverages instruction-tuned large language models, specifically heavily relying on advanced models like OpenAI's GPT series or comparable open-source frameworks to generate context-aware educational responses."},
    {"question": "What are the key limitations of the study mentioned by the authors?", "reference": "The key limitations include the potential for AI hallucinations (prompting incorrect technical advice), the risk of students relying too heavily on the bot rather than learning the core material, and data privacy security concerns."},
    {"question": "What specific database concepts did the chatbot focus on teaching?", "reference": "The chatbot focused on teaching fundamental database topics including entity-relationship (ER) modeling, database normalization rules, and practical SQL query formulation."},
    {"question": "What role does Retrieval-Augmented Generation (RAG) play in the educational chatbot framework?", "reference": "RAG is utilized to ground the LLM's responses in specific course materials, textbooks, and syllabus guidelines, thereby ensuring the answers are factually accurate, relevant to the curriculum, and drastically reducing hallucinations."},
    {"question": "How did the developers handle student data privacy while using large language models?", "reference": "Student data privacy was maintained by omitting personally identifiable information (PII) before queries were sent to external LLM APIs, and by hosting open-source models locally when dealing with sensitive grade or profile data."},
    {"question": "What was the observed impact of the chatbot on student engagement?", "reference": "The implementation of the chatbot led to higher student engagement metrics, as students felt more comfortable asking 'dumb' questions without fear of judgment from human instructors, and appreciated the instant feedback."},
    {"question": "According to the research, did the chatbot replace the need for traditional teaching assistants?", "reference": "No, the chatbot did not completely replace human teaching assistants. Instead, it served as a supplementary tool that handled repetitive foundational questions, freeing up human TAs to focus on more complex, conceptual discussions."},
    {"question": "How does the system prompt the LLM to format its educational explanations?", "reference": "The system prompt instructs the LLM to act as a Socratic tutor—guiding students toward the right answer through hints and structured breakdowns rather than directly handing over the complete solution or full SQL code."},
    {"question": "How were database schemas provided to the chatbot to assist with SQL queries?", "reference": "Database schemas and table structures were embedded into the FAISS vector database as text chunks, allowing the RAG system to retrieve the exact schema layout whenever a student query involved writing SQL for a specific assignment."},
    {"question": "Did the chatbot demonstrate any bias or disparate impact on international students compared to native speakers?", "reference": "The chatbot actually proved beneficial for international students, as its ability to rapidly translate complex technical jargon and offer infinite patience helped bridge language barriers more effectively than traditional lecture formats."},
    {"question": "What vector database technology was used to store the course materials?", "reference": "The system utilized FAISS (Facebook AI Similarity Search) to index and retrieve chunked documents, syllabus details, and assignment guidelines efficiently."},
    {"question": "How did the framework handle multi-turn conversations where students asked follow-up questions?", "reference": "The system framework integrated a conversation memory buffer which appended prior chat history to the prompt, allowing the LLM to maintain context across multiple turns of dialogue."},
    {"question": "Were any cost considerations mentioned regarding the deployment of commercial LLM APIs?", "reference": "Yes, cost was cited as a major consideration. The researchers mitigated API expenses by using caching strategies for repeated questions and by offloading simpler queries to cheaper, smaller models or local open-source implementations."},
    {"question": "What future work do the authors propose to improve the chatbot?", "reference": "Future work includes adding multimodal capabilities to analyze student-drawn ER diagrams, integrating the chatbot directly into the university's Learning Management System (LMS), and expanding the context base to other computing subjects."},
    {"question": "How does the chatbot distinguish between an administrative query (like grading policy) and a technical query?", "reference": "The RAG system's initial routing layer or triage classifies the intent of the prompt. If it's administrative, it retrieves chunks from the syllabus. If it's technical, it retrieves chunks from the lecture notes and textbooks."},
    {"question": "What specific chunking strategy was used to break down the textbook PDFs for the vector database?", "reference": "The documents were broken down using a Recursive Character Text Splitter with overlapping chunks (e.g., 400 characters with a 50-character overlap) to ensure that technical sentences were not cut off arbitrarily and semantic meaning was preserved."},
    {"question": "How did the system handle queries where the course material provided contradictory information?", "reference": "If contradictory information was retrieved, the LLM was prompted to highlight the ambiguity or default to the most recently dated lecture slide, relying on the professor as the final arbiter."},
    {"question": "What programming language and backend framework were predominantly utilized to host this RAG system?", "reference": "The system backend was driven by Python, largely relying on frameworks like Flask or FastAPI for the web server and LangChain to orchestrate the RAG retrieval flow and vector database integrations."},
    
    # --- OUT-OF-DOMAIN QUESTIONS ---
    {"question": "What is the capital of Australia?", "reference": "I don't know."},
    {"question": "Can you explain the detailed biological process of photosynthesis in plants?", "reference": "I don't know."},
    {"question": "Who won the FIFA World Cup in the year 2022?", "reference": "I don't know."},
    {"question": "What are the primary differences between General Relativity and Quantum Mechanics?", "reference": "I don't know."},
    {"question": "How do you cook an authentic Italian spaghetti carbonara?", "reference": "I don't know."}
]

JUDGE_PROMPT_TEMPLATE = """
You are an expert evaluator. Assess the AI-generated answer based on the Reference Answer and Source Context.
Score from 1 to 5. Output ONLY valid JSON.

Metrics:
- relevance: Does the AI answer directly address the Question?
- accuracy: Is the AI answer factually correct?
- completeness: Does it cover all essential points?
- groundedness: Is it supported by the Source Context? (Score 5 if AI says "I don't know" correctly).

Question: {question}
Reference Answer: {reference}
Source Context: {context}
AI Answer: {answer}

JSON Format:
{{"relevance": <int>, "accuracy": <int>, "completeness": <int>, "groundedness": <int>}}
"""

judge_prompt = PromptTemplate(template=JUDGE_PROMPT_TEMPLATE, input_variables=["question", "reference", "context", "answer"])

def run_llm_judge(question, reference, context, answer):
    def judge_call():
        return (judge_prompt | LLM_HOLDER[0]).invoke({"question": question, "reference": reference, "context": context, "answer": answer})
    try:
        result = invoke_with_rotation(judge_call)
        return json.loads(result.content.strip("`").replace("json\n", ""))
    except:
        return {"relevance": 0, "accuracy": 0, "completeness": 0, "groundedness": 0}

CSV_FILE = "multi_doc_evaluation_results.csv"
FIELDNAMES = ["Question", "System", "BERTScore_F1", "Relevance", "Accuracy", "Completeness", "Groundedness", "Latency_sec"]

def evaluate_multi_doc():
    print("Initializing Multi-Doc Evaluation Pipeline...")
    all_topic_dict = {}

    for pdf_name in EVAL_PAPERS:
        pdf_path = os.path.join(PDF_UPLOADS_DIR, pdf_name)
        if not os.path.exists(pdf_path):
            print(f"Skipping missing file: {pdf_name}")
            continue
            
        print(f"Processing: {pdf_name}")
        extracted_text = extract_text_from_pdf(pdf_path)
        sections = extract_pdf_sections(extracted_text)
        refined = invoke_with_rotation(refine_sections, sections, LLM_HOLDER[0])
        topic_dict = split_sections_with_content(extracted_text, refined)
        all_topic_dict.update(topic_dict)

    print(f"Creating Unified Vector Database with {len(all_topic_dict)} sections...")
    vector_db = create_vector_db(all_topic_dict, EMBEDDER)
    
    results = []
    for idx, test in enumerate(TEST_CASES):
        question = test["question"]
        reference = test["reference"]
        print(f"\nEvaluating Question {idx+1}/{len(TEST_CASES)}")
        
        # 1. Baseline LLM
        start_t = time.time()
        baseline_res = invoke_with_rotation(LLM_HOLDER[0].invoke, question)
        baseline_latency = time.time() - start_t
        baseline_answer = baseline_res.content

        # 2. Multi-Doc RAG
        start_t = time.time()
        chain = get_qa_chain(vectordb=vector_db, llm=LLM_HOLDER[0])
        rag_res = invoke_with_rotation(chain.invoke, {"query": question})
        rag_latency = time.time() - start_t
        rag_answer = rag_res["result"]
        rag_context = "\n".join([doc.page_content for doc in rag_res.get("source_documents", [])])

        # BERTScore
        _, _, F1_b = bert_scorer([baseline_answer], [reference], lang="en")
        _, _, F1_r = bert_scorer([rag_answer], [reference], lang="en")

        # Judge
        baseline_judge = run_llm_judge(question, reference, "NONE", baseline_answer)
        rag_judge = run_llm_judge(question, reference, rag_context, rag_answer)

        results.append({"Question": question, "System": "Baseline", "BERTScore_F1": round(F1_b[0].item(), 4), "Relevance": baseline_judge.get("relevance",0), "Accuracy": baseline_judge.get("accuracy",0), "Completeness": baseline_judge.get("completeness",0), "Groundedness": baseline_judge.get("groundedness",0), "Latency_sec": round(baseline_latency, 2)})
        results.append({"Question": question, "System": "Multi-Doc RAG", "BERTScore_F1": round(F1_r[0].item(), 4), "Relevance": rag_judge.get("relevance",0), "Accuracy": rag_judge.get("accuracy",0), "Completeness": rag_judge.get("completeness",0), "Groundedness": rag_judge.get("groundedness",0), "Latency_sec": round(rag_latency, 2)})

        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(results)
        print(f"   Progress saved")

    print(f"\nMulti-Doc Evaluation complete! Results: {CSV_FILE}")

if __name__ == "__main__":
    evaluate_multi_doc()
