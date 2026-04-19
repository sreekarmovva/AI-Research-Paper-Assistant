from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


def get_qa_chain(vectordb, llm, section_filter=None):
    
    search_kwargs = {"k": 4}
    if section_filter:
        search_kwargs["filter"] = {"section": section_filter}

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
)
    
    # Custom prompt
    prompt_template = """
You are a strict research assistant. Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

Instructions:
- Summarize the answer in your own words. DO NOT exactly copy sentences.
- STRICT LENGTH LIMIT: Your total response MUST be under 350 characters.
- FORMAT: Start with a brief 1-sentence summary. Then provide 2-3 short bullet points.
- No fluff or extra jargon. Give only the core facts.

If the answer is not clearly present, say: "I don't know."
""" 

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return chain
    