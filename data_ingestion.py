# ingest.py

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Constants ---
DATA_PATH = "data"
DB_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # A good, fast, and local model

def main():
    """
    Main function to perform the ingestion process.
    1. Loads documents from the data directory.
    2. Splits documents into smaller chunks.
    3. Creates a Chroma vector store with embeddings.
    """
    print("--- Starting Ingestion Process ---")

    # 1. Load documents
    print(f"Loading documents from '{DATA_PATH}'...")
    # Using TextLoader for individual files to avoid metadata issues
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith('.txt'):
            file_path = os.path.join(DATA_PATH, file)
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
    
    if not documents:
        print("No .txt files found in the data directory. Exiting.")
        return
        
    print(f"Loaded {len(documents)} document(s).")

    # 2. Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    # 3. Create embeddings and store in ChromaDB
    print("Creating embeddings and storing in ChromaDB...")
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'} # Use CPU for broad compatibility
    )

    # Create the vector store. This will automatically handle embedding the chunks.
    # The `persist_directory` argument tells Chroma where to save the data on disk.
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    )

    print("--- Ingestion Complete ---")
    print(f"Vector store created at '{DB_PATH}' with {vector_store._collection.count()} entries.")


if __name__ == "__main__":
    main()