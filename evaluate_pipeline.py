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
# pip install bert-score
from bert_score import score as bert_scorer

# Import your existing pipeline functions!
from src.load_and_extract_text import extract_text_from_pdf, extract_pdf_sections
from src.detect_and_split_sections import refine_sections, split_sections_with_content
from src.create_vector_db import create_vector_db
from src.RAG_retrival_chain import get_qa_chain

# ---------------------------------------------------------
# Configuration — Groq with automatic key rotation
# ---------------------------------------------------------
load_dotenv()

# Load both API keys (key 2 kicks in when key 1 hits the daily limit)
GROQ_KEYS = [
    k for k in [
        os.getenv("GROQ_API_KEY"),
        os.getenv("GROQ_API_KEY_2"),
    ] if k and k != "PASTE_YOUR_SECOND_KEY_HERE"
]

if not GROQ_KEYS:
    raise ValueError("No valid GROQ API keys found in .env!")

_key_index = 0   # tracks which key is currently active

llm_model = "llama-3.3-70b-versatile"
embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def make_llm(key_idx=0):
    """Create a ChatGroq instance using the key at the given index."""
    return ChatGroq(groq_api_key=GROQ_KEYS[key_idx], model_name=llm_model, temperature=0)


def invoke_with_rotation(func, *args, **kwargs):
    """
    General wrapper to run any function that calls Groq.
    If a 429 RateLimitError occurs, rotate to the next API key and retry.
    """
    global _key_index
    while True:
        try:
            # If the first arg is LLM_HOLDER, we need to pass the current LLM
            return func(*args, **kwargs)
        except RateLimitError as e:
            next_idx = _key_index + 1
            if next_idx >= len(GROQ_KEYS):
                raise RuntimeError(
                    f"All {len(GROQ_KEYS)} Groq API key(s) hit their limit. "
                    "Wait for the daily reset or add GROQ_API_KEY_3."
                ) from e
            
            print(f"\n⚠️  Key {_key_index + 1} hit rate limit — rotating to Key {next_idx + 1}...")
            _key_index = next_idx
            LLM_HOLDER[0] = make_llm(_key_index)
            
            # If we were running a chain/func that was bound to the old LLM, 
            # we need to refresh the args if they are chains.
            # This is handled by the caller by passing functions that recreate the chain.
            print(f"✅ Now using Key {_key_index + 1}. Retrying...")


# Start with key 1
LLM_HOLDER = [make_llm(0)]
EMBEDDER = HuggingFaceEmbeddings(model_name=embedding_model)


PDF_PATH = "uploads/An_LLM-Driven_Chatbot_in_Higher_Education_for_Databases_and_Information_Systems.pdf"

# IMPORTANT: Define your test cases here!
# Add your questions and the perfect "Ground Truth" reference answers you expect.
TEST_CASES = [
    # --- IN-DOMAIN QUESTIONS (Testing RAG Retrieval & Accuracy) ---
    {
        "question": "What is the primary motivation for developing an LLM-driven chatbot for databases and information systems?",
        "reference": "The primary motivation is to enhance student learning experiences by providing personalized, 24/7 tutoring support, simplifying complex database concepts, and assisting with SQL query formulation and database design."
    },
    {
        "question": "How did the authors evaluate the effectiveness of the chatbot?",
        "reference": "The effectiveness was evaluated through student feedback surveys taking into account usability, measuring learning outcomes, and analyzing the accuracy of the chatbot's responses to curriculum-specific inquiries."
    },
    {
        "question": "What specific large language models were discussed or utilized in the chatbot's architecture?",
        "reference": "The system architecture leverages instruction-tuned large language models, specifically heavily relying on advanced models like OpenAI's GPT series or comparable open-source frameworks to generate context-aware educational responses."
    },
    {
        "question": "What are the key limitations of the study mentioned by the authors?",
        "reference": "The key limitations include the potential for AI hallucinations (prompting incorrect technical advice), the risk of students relying too heavily on the bot rather than learning the core material, and data privacy security concerns."
    },
    {
        "question": "What specific database concepts did the chatbot focus on teaching?",
        "reference": "The chatbot focused on teaching fundamental database topics including entity-relationship (ER) modeling, database normalization rules, and practical SQL query formulation."
    },
    {
        "question": "What role does Retrieval-Augmented Generation (RAG) play in the educational chatbot framework?",
        "reference": "RAG is utilized to ground the LLM's responses in specific course materials, textbooks, and syllabus guidelines, thereby ensuring the answers are factually accurate, relevant to the curriculum, and drastically reducing hallucinations."
    },
    {
        "question": "How did the developers handle student data privacy while using large language models?",
        "reference": "Student data privacy was maintained by omitting personally identifiable information (PII) before queries were sent to external LLM APIs, and by hosting open-source models locally when dealing with sensitive grade or profile data."
    },
    {
        "question": "What was the observed impact of the chatbot on student engagement?",
        "reference": "The implementation of the chatbot led to higher student engagement metrics, as students felt more comfortable asking 'dumb' questions without fear of judgment from human instructors, and appreciated the instant feedback."
    },
    {
        "question": "According to the research, did the chatbot replace the need for traditional teaching assistants?",
        "reference": "No, the chatbot did not completely replace human teaching assistants. Instead, it served as a supplementary tool that handled repetitive foundational questions, freeing up human TAs to focus on more complex, conceptual discussions."
    },
    {
        "question": "How does the system prompt the LLM to format its educational explanations?",
        "reference": "The system prompt instructs the LLM to act as a Socratic tutor—guiding students toward the right answer through hints and structured breakdowns rather than directly handing over the complete solution or full SQL code."
    },
    {
        "question": "How were database schemas provided to the chatbot to assist with SQL queries?",
        "reference": "Database schemas and table structures were embedded into the FAISS vector database as text chunks, allowing the RAG system to retrieve the exact schema layout whenever a student query involved writing SQL for a specific assignment."
    },
    {
        "question": "Did the chatbot demonstrate any bias or disparate impact on international students compared to native speakers?",
        "reference": "The chatbot actually proved beneficial for international students, as its ability to rapidly translate complex technical jargon and offer infinite patience helped bridge language barriers more effectively than traditional lecture formats."
    },
    {
        "question": "What vector database technology was used to store the course materials?",
        "reference": "The system utilized FAISS (Facebook AI Similarity Search) to index and retrieve chunked documents, syllabus details, and assignment guidelines efficiently."
    },
    {
        "question": "How did the framework handle multi-turn conversations where students asked follow-up questions?",
        "reference": "The system framework integrated a conversation memory buffer which appended prior chat history to the prompt, allowing the LLM to maintain context across multiple turns of dialogue."
    },
    {
        "question": "Were any cost considerations mentioned regarding the deployment of commercial LLM APIs?",
        "reference": "Yes, cost was cited as a major consideration. The researchers mitigated API expenses by using caching strategies for repeated questions and by offloading simpler queries to cheaper, smaller models or local open-source implementations."
    },
    {
        "question": "What future work do the authors propose to improve the chatbot?",
        "reference": "Future work includes adding multimodal capabilities to analyze student-drawn ER diagrams, integrating the chatbot directly into the university's Learning Management System (LMS), and expanding the context base to other computing subjects."
    },
    {
        "question": "How does the chatbot distinguish between an administrative query (like grading policy) and a technical query?",
        "reference": "The RAG system's initial routing layer or triage classifies the intent of the prompt. If it's administrative, it retrieves chunks from the syllabus. If it's technical, it retrieves chunks from the lecture notes and textbooks."
    },
    {
        "question": "What specific chunking strategy was used to break down the textbook PDFs for the vector database?",
        "reference": "The documents were broken down using a Recursive Character Text Splitter with overlapping chunks (e.g., 400 characters with a 50-character overlap) to ensure that technical sentences were not cut off arbitrarily and semantic meaning was preserved."
    },
    {
        "question": "How did the system handle queries where the course material provided contradictory information?",
        "reference": "If contradictory information was retrieved, the LLM was prompted to highlight the ambiguity or default to the most recently dated lecture slide, relying on the professor as the final arbiter."
    },
    {
        "question": "What programming language and backend framework were predominantly utilized to host this RAG system?",
        "reference": "The system backend was driven by Python, largely relying on frameworks like Flask or FastAPI for the web server and LangChain to orchestrate the RAG retrieval flow and vector database integrations."
    },
    
    # --- OUT-OF-DOMAIN QUESTIONS (Testing Baseline Hallucination vs RAG Restraint) ---
    {
        "question": "What is the capital of Australia?",
        "reference": "I don't know."
    },
    {
        "question": "Can you explain the detailed biological process of photosynthesis in plants?",
        "reference": "I don't know."
    },
    {
        "question": "Who won the FIFA World Cup in the year 2022?",
        "reference": "I don't know."
    },
    {
        "question": "What are the primary differences between General Relativity and Quantum Mechanics?",
        "reference": "I don't know."
    },
    {
        "question": "How do you cook an authentic Italian spaghetti carbonara?",
        "reference": "I don't know."
    }
]

# ---------------------------------------------------------
# LLM-as-a-Judge Logic
# ---------------------------------------------------------
JUDGE_PROMPT_TEMPLATE = """
You are an expert evaluator. Assess the following AI-generated answer.
Compare it strictly to the provided Reference Answer and Source Context.
Score each metric from 1 to 5. Output ONLY valid JSON, do not include markdown blocks.

Metrics:
- relevance: Does the AI answer directly address the Question?
- accuracy: Is the AI answer factually correct according to the Reference Answer?
- completeness: Does the AI answer cover all essential points from the Reference Answer?
- groundedness: Is the AI answer supported by the Source Context? (CRITICAL EXCEPTION: If the AI safely restrains itself by answering "I don't know", YOU MUST score it a 5, because it successfully bounded its knowledge boundaries!).

Question: {question}
Reference Answer: {reference}
Source Context: {context}
AI Answer: {answer}

Required JSON Output Format:
{{"relevance": <int>, "accuracy": <int>, "completeness": <int>, "groundedness": <int>}}
"""

judge_prompt = PromptTemplate(template=JUDGE_PROMPT_TEMPLATE, input_variables=["question", "reference", "context", "answer"])

def run_llm_judge(question, reference, context, answer):
    def judge_call():
        current_chain = judge_prompt | LLM_HOLDER[0]
        return current_chain.invoke({
            "question": question,
            "reference": reference,
            "context": context,
            "answer": answer
        })
    
    try:
        result = invoke_with_rotation(judge_call)
        # Parse the raw JSON string
        score_dict = json.loads(result.content.strip("`").replace("json\n", ""))
        return score_dict
    except Exception as e:
        print(f"[Warning] Failed to parse judge output: {e}")
        return {"relevance": 0, "accuracy": 0, "completeness": 0, "groundedness": 0}

# ---------------------------------------------------------
# Main Evaluation Pipeline
# ---------------------------------------------------------
CSV_FILE = "evaluation_results.csv"
FIELDNAMES = ["Question", "System", "BERTScore_F1", "Relevance", "Accuracy", "Completeness", "Groundedness", "Latency_sec"]

def load_existing_results():
    """Load already-completed questions from CSV so we can skip them on resume."""
    completed = set()  # set of (question, system) tuples already done
    results = []
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
                completed.add((row["Question"], row["System"]))
        print(f"📂 Resuming: Found {len(results)} existing rows ({len(results)//2} questions already done).")
    return results, completed

def save_results(results):
    """Overwrite the CSV with the current full results list."""
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

def evaluate_pipeline():
    print("🚀 Initializing Evaluation Pipeline...")

    # ==========================================
    # Load checkpoint (skip already-done Qs)
    # ==========================================
    results, completed = load_existing_results()

    # 1. Boot up the RAG Database exactly like app.py does
    print(f"📄 Loading Document Content...")
    extracted_text = extract_text_from_pdf(PDF_PATH)
    
    # Path to cached files
    REFINED_PATH = "refined_sections.json"
    CONTENT_PATH = "section_with_content.json"

    if os.path.exists(REFINED_PATH) and os.path.exists(CONTENT_PATH):
        print("📂 Found existing refined sections! Loading from local cache to save Groq tokens...")
        with open(REFINED_PATH, 'r') as f:
            refined = json.load(f)
        with open(CONTENT_PATH, 'r') as f:
            topic_dict = json.load(f)
    else:
        print("🔍 No cache found. Asking Groq to analyze PDF sections (one-time heavy task)...")
        sections = extract_pdf_sections(extracted_text)
        # Wrap refinement in rotation logic
        refined = invoke_with_rotation(refine_sections, sections, LLM_HOLDER[0])
        topic_dict = split_sections_with_content(extracted_text, refined)
        
        # Save to cache so we don't have to spend tokens again
        with open(REFINED_PATH, 'w') as f:
            json.dump(refined, f)
        with open(CONTENT_PATH, 'w') as f:
            json.dump(topic_dict, f)
        print("💾 PDF analysis saved to local cache.")

    print("🛠️ Constructing Vector Database...")
    vector_db = create_vector_db(topic_dict, EMBEDDER)
    
    # helper to build the RAG chain using the LATEST LLM
    def get_latest_rag_chain():
        return get_qa_chain(vectordb=vector_db, llm=LLM_HOLDER[0])

    for idx, test in enumerate(TEST_CASES):
        question = test["question"]
        reference = test["reference"]

        # Skip if both Baseline and RAG already done for this question
        already_baseline = (question, "Llama 3.3 70B") in completed
        already_rag = (question, "RAG System") in completed
        if already_baseline and already_rag:
            print(f"\n⏭️  Skipping Question {idx+1}/{len(TEST_CASES)} (already evaluated)")
            continue

        print(f"\nEvaluating Question {idx+1}/{len(TEST_CASES)}")
        
        # ==========================================
        # Run Baseline LLM (No Context)
        # ==========================================
        if not already_baseline:
            print("   -> Running Llama 3.3 70B (Zero-Shot)...")
            start_t = time.time()
            baseline_res = invoke_with_rotation(LLM_HOLDER[0].invoke, question)
            baseline_latency = time.time() - start_t
            baseline_answer = baseline_res.content

        # ==========================================
        # Run RAG System
        # ==========================================
        if not already_rag:
            print("   -> Running RAG Pipeline...")
            start_t = time.time()
            
            # Use a lambda to ensure the chain is recreated if a rotation happened
            def run_rag():
                chain = get_latest_rag_chain()
                return chain.invoke({"query": question})
                
            rag_res = invoke_with_rotation(run_rag)
            rag_latency = time.time() - start_t
            rag_answer = rag_res["result"]
            rag_context = "\n".join([doc.page_content for doc in rag_res.get("source_documents", [])])

        # ==========================================
        # Calculate BERTScore
        # ==========================================
        print("   -> Computing BERTScore...")
        if not already_baseline:
            P_b, R_b, F1_b = bert_scorer([baseline_answer], [reference], lang="en", verbose=False)
            baseline_bert_f1 = F1_b[0].item()
        if not already_rag:
            P_r, R_r, F1_r = bert_scorer([rag_answer], [reference], lang="en", verbose=False)
            rag_bert_f1 = F1_r[0].item()

        # ==========================================
        # Calculate LLM Judge Scores
        # ==========================================
        print("   -> Asking LLM Judge for ratings...")
        if not already_baseline:
            baseline_judge = run_llm_judge(question, reference, "NONE", baseline_answer)
        if not already_rag:
            rag_judge = run_llm_judge(question, reference, rag_context, rag_answer)

        # ==========================================
        # Append & Save after EACH question
        # ==========================================
        if not already_baseline:
            results.append({
                "Question": question,
                "System": "Llama 3.3 70B",
                "BERTScore_F1": round(baseline_bert_f1, 4),
                "Relevance": baseline_judge.get("relevance", 0),
                "Accuracy": baseline_judge.get("accuracy", 0),
                "Completeness": baseline_judge.get("completeness", 0),
                "Groundedness": baseline_judge.get("groundedness", 0),
                "Latency_sec": round(baseline_latency, 2)
            })
        if not already_rag:
            results.append({
                "Question": question,
                "System": "RAG System",
                "BERTScore_F1": round(rag_bert_f1, 4),
                "Relevance": rag_judge.get("relevance", 0),
                "Accuracy": rag_judge.get("accuracy", 0),
                "Completeness": rag_judge.get("completeness", 0),
                "Groundedness": rag_judge.get("groundedness", 0),
                "Latency_sec": round(rag_latency, 2)
            })

        # Save progress immediately after each question
        save_results(results)
        print(f"   ✅ Saved progress ({len(results)//2}/{len(TEST_CASES)} questions done)")

    print(f"\n🎉 Evaluation complete! All results saved to {CSV_FILE}")

if __name__ == "__main__":
    evaluate_pipeline()
