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

# Metrics
from bert_score import score as bert_scorer
from rouge_score import rouge_scorer

# Import existing pipeline functions
from src.load_and_extract_text import extract_text_from_pdf, extract_pdf_sections
from src.detect_and_split_sections import refine_sections, split_sections_with_content
from src.create_vector_db import create_vector_db
from src.RAG_retrival_chain import get_qa_chain

load_dotenv()

# Configuration
GROQ_KEYS = [
    k for k in [os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY_2")]
    if k and k != "PASTE_YOUR_SECOND_KEY_HERE"
]
_key_index = 0
llm_model = "llama-3.3-70b-versatile"
embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def make_llm(key_idx=0):
    return ChatGroq(groq_api_key=GROQ_KEYS[key_idx], model_name=llm_model, temperature=0)

LLM_HOLDER = [make_llm(0)]
EMBEDDER = HuggingFaceEmbeddings(model_name=embedding_model)
PDF_PATH = "uploads/An_LLM-Driven_Chatbot_in_Higher_Education_for_Databases_and_Information_Systems.pdf"

def invoke_with_rotation(func, *args, **kwargs):
    global _key_index
    while True:
        try:
            return func(*args, **kwargs)
        except RateLimitError as e:
            next_idx = _key_index + 1
            if next_idx >= len(GROQ_KEYS): raise e
            print(f"Rotating to Key {next_idx + 1}...")
            _key_index = next_idx
            LLM_HOLDER[0] = make_llm(_key_index)
            return func(*args, **kwargs)

# Test cases from evaluate_pipeline.py
TEST_CASES = [
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
    {"question": "What is the capital of Australia?", "reference": "I don't know."},
    {"question": "Can you explain the detailed biological process of photosynthesis in plants?", "reference": "I don't know."},
    {"question": "Who won the FIFA World Cup in the year 2022?", "reference": "I don't know."},
    {"question": "What are the primary differences between General Relativity and Quantum Mechanics?", "reference": "I don't know."},
    {"question": "How do you cook an authentic Italian spaghetti carbonara?", "reference": "I don't know."}
]

JUDGE_PROMPT_TEMPLATE = """
You are an expert evaluator. Assess the following AI-generated answer.
Compare it strictly to the provided Reference Answer and Source Context.
Score each metric from 1 to 5. Output ONLY valid JSON.

Metrics:
- relevance: Does the AI answer directly address the Question?
- accuracy: Is the AI answer factually correct?
- completeness: Does it cover all points from the reference?
- groundedness: Supported by context? (Score 5 if it correctly says "I don't know" for out-of-domain).

Question: {question}
Reference Answer: {reference}
Source Context: {context}
AI Answer: {answer}

Required JSON: {{"relevance": <int>, "accuracy": <int>, "completeness": <int>, "groundedness": <int>}}
"""
judge_prompt = PromptTemplate(template=JUDGE_PROMPT_TEMPLATE, input_variables=["question", "reference", "context", "answer"])

def run_llm_judge(question, reference, context, answer):
    try:
        res = invoke_with_rotation((judge_prompt | LLM_HOLDER[0]).invoke, {"question": question, "reference": reference, "context": context, "answer": answer})
        return json.loads(res.content.strip("`").replace("json\n", ""))
    except: return {"relevance": 0, "accuracy": 0, "completeness": 0, "groundedness": 0}

def evaluate():
    print("🚀 Starting Final Single-Doc Evaluation (BERTScore + ROUGE + LLM Judge)...")
    
    # Setup RAG
    txt = extract_text_from_pdf(PDF_PATH)
    refined = invoke_with_rotation(refine_sections, extract_pdf_sections(txt), LLM_HOLDER[0])
    topic_dict = split_sections_with_content(txt, refined)
    vector_db = create_vector_db(topic_dict, EMBEDDER)
    rag_chain = get_qa_chain(vectordb=vector_db, llm=LLM_HOLDER[0])
    
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []
    
    for idx, test in enumerate(TEST_CASES):
        q, ref = test["question"], test["reference"]
        print(f"Question {idx+1}/{len(TEST_CASES)}")
        
        # Baseline
        base_ans = invoke_with_rotation(LLM_HOLDER[0].invoke, q).content
        # RAG
        rag_res = invoke_with_rotation(rag_chain.invoke, {"query": q})
        rag_ans = rag_res["result"]
        rag_ctx = "\n".join([d.page_content for d in rag_res.get("source_documents", [])])
        
        for sys_name, ans, ctx in [("Baseline", base_ans, "NONE"), ("RAG System", rag_ans, rag_ctx)]:
            # Metrics
            _, _, f1 = bert_scorer([ans], [ref], lang="en")
            rs = rouge.score(ref, ans)
            judge = run_llm_judge(q, ref, ctx, ans)
            
            results.append({
                "Question": q, "System": sys_name,
                "BERTScore": round(f1[0].item(), 4),
                "ROUGE_1": round(rs['rouge1'].fmeasure, 4),
                "ROUGE_2": round(rs['rouge2'].fmeasure, 4),
                "ROUGE_L": round(rs['rougeL'].fmeasure, 4),
                "Relevance": judge["relevance"], "Accuracy": judge["accuracy"],
                "Completeness": judge["completeness"], "Groundedness": judge["groundedness"]
            })

    with open("final_single_doc_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("✅ Done! Results saved to final_single_doc_results.csv")

if __name__ == "__main__":
    evaluate()
