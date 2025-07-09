# document_loader.py

import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_documents_from_folder(folder_path: str) -> List[Document]:
    """
    Load and combine all PDFs in the given folder into LangChain Documents.
    """
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Loading: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into chunks using LangChain's RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(documents)

def load_and_chunk_pdfs(folder_path: str = "data/rti_docs") -> List[Document]:
    """
    Combined function to load PDFs from a folder and split them into chunks.
    """
    raw_documents = load_documents_from_folder(folder_path)
    print(f"Loaded {len(raw_documents)} raw documents.")
    
    chunked_documents = chunk_documents(raw_documents)
    print(f"Chunked into {len(chunked_documents)} total chunks.")
    
    return chunked_documents

# For standalone testing
if __name__ == "__main__":
    chunks = load_and_chunk_pdfs()
    print(f"Sample Chunk:\n{chunks[0].page_content[:500]}...")
