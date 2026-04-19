from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


def create_vector_db(section_dict, embedder):
    """
    Creates a FAISS vector database using section-wise content.
    
    Args:
        section_dict (dict): {section_name: content}
        embedder: embedding model
        
    Returns:
        vectordb: FAISS vector database
    """

    documents = []

    # Split each section separately
    for section_name, content in section_dict.items():

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "]
        )

        chunks = splitter.split_text(content)

        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"section": section_name}
                )
            )

    # Create FAISS vector database
    vectordb = FAISS.from_documents(documents, embedding=embedder)

    # Save locally
    vectordb.save_local("research_paper_vector_db")

    return vectordb