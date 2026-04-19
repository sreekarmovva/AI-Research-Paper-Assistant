from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from flask import Flask, render_template, request, jsonify
from groq import RateLimitError

from src.load_and_extract_text import extract_text_from_pdf, extract_pdf_sections
from src.detect_and_split_sections import refine_sections, split_sections_with_content
from src.get_summary import generate_detailed_summary
from src.create_vector_db import create_vector_db
from src.RAG_retrival_chain import get_qa_chain

from dotenv import load_dotenv
import os, traceback

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Globals
Research_paper_topics = {}
vector_db = None
full_text = ""

# Load settings
llm_model = os.getenv("LLM_MODEL")
embedding_model = os.getenv("EMBEDDING_MODEL")

# Load keys for rotation
GROQ_KEYS = [
    k for k in [os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY_2")]
    if k and k != "PASTE_YOUR_SECOND_KEY_HERE"
]

def get_llm(key_idx=0):
    return ChatGroq(groq_api_key=GROQ_KEYS[key_idx], model_name=llm_model)

# Initial LLM and Embedder
llm = get_llm(0)
embedder = HuggingFaceEmbeddings(model_name=embedding_model)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_pdf():
    global full_text, Research_paper_topics, vector_db

    try:
        vector_db = None
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        extracted_text = extract_text_from_pdf(filename)
        full_text = extracted_text

        extracted_sections = extract_pdf_sections(full_text=extracted_text)
        refined_sections = refine_sections(extracted_sections, llm)
        section_with_content = split_sections_with_content(extracted_text, refined_sections)

        Research_paper_topics = section_with_content

        return jsonify({"topics": list(Research_paper_topics.keys())})

    except Exception as e:
        print("Error in /upload:", repr(e))
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500


@app.route('/check_state', methods=['GET'])
def check_state():
    # Returns true if a paper is currently loaded in memory
    return jsonify({"is_uploaded": bool(Research_paper_topics)})


@app.route('/summary', methods=['POST'])
def get_summary():
    global Research_paper_topics, llm
    try:
        payload = request.get_json(silent=True) or {}
        topic = payload.get('topic')

        if not topic:
            return jsonify({"error": "Missing topic"}), 400

        if not Research_paper_topics:
            return jsonify({"error": "Upload a PDF first"}), 400

        # Find the topic content case-insensitively
        topic_content = None
        target_topic = topic.strip().upper()
        
        for key, content in Research_paper_topics.items():
            if key.strip().upper() == target_topic:
                topic_content = content
                break
        
        if not topic_content:
            return jsonify({"summary": "No summary available for this specific section."})

        try:
            summary = generate_detailed_summary(topic_content, llm)
        except RateLimitError:
            if len(GROQ_KEYS) > 1:
                print("⚠️ App Key 1 hit rate limit, switching to Key 2...")
                llm = get_llm(1)
                summary = generate_detailed_summary(topic_content, llm)
            else:
                raise

        return jsonify({"summary": summary})

    except Exception as e:
        print("Error in /summary:", repr(e))
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500


@app.route('/chat', methods=['POST'])
def chat():
    global full_text, vector_db, Research_paper_topics, llm
    try:
        user_message = request.json.get('message')
        topic = request.json.get('topic')

        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        if not Research_paper_topics:
            return jsonify({"error": "Upload a PDF first"}), 400

        # Create vector DB once
        if not vector_db:
            vector_db = create_vector_db(
                section_dict=Research_paper_topics,
                embedder=embedder
            )

        def run_chain():
            chain = get_qa_chain(vectordb=vector_db, llm=llm)
            return chain.invoke({"query": user_message})

        try:
            result = run_chain()
        except RateLimitError:
            if len(GROQ_KEYS) > 1:
                print("⚠️ App Key 1 hit rate limit, switching to Key 2...")
                llm = get_llm(1)
                result = run_chain()
            else:
                raise

        answer = result["result"]
        sources = result.get("source_documents", [])

        # CLEAN ANSWER
        if "Supporting Context" in answer:
            answer = answer.split("Supporting Context")[0]

        if "🔍" in answer:
            answer = answer.split("🔍")[0]

        answer = answer.replace("**", "").strip()

        # limit verbosity
        answer = result["result"]
        sources = result.get("source_documents", [])

        # CLEAN ANSWER (remove bolding for cleaner UI if desired, or keep it)
        answer = answer.strip()

        # FORMAT SOURCES (Citations)
        source_texts = []
        seen_content = set()

        for doc in sources:
            content = doc.page_content.strip().replace("\n", " ")
            # Avoid showing the same text twice
            if content[:50] in seen_content:
                continue
            seen_content.add(content[:50])

            section = doc.metadata.get("section", "Context")
            # Show a meaningful snippet
            snippet = content[:200] + "..." if len(content) > 200 else content
            source_texts.append(f"[{len(source_texts)+1}] {section}: \"{snippet}\"")

        # Return separately so frontend can style them
        return jsonify({
            "response": answer, 
            "sources": "\n".join(source_texts) if source_texts else ""
        })

    except Exception as e:
        print("Error in /chat:", repr(e))
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True)