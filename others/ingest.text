
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Create vector database
def create_vector_db():
    print("🔍 Step 1: Loading PDF documents from:", DATA_PATH)
    loader = DirectoryLoader(
        DATA_PATH,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} document(s).")

    print("🔄 Step 2: Splitting documents into chunks (safe length)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,        # Reduced to avoid exceeding context window
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(texts)} text chunks.")

    # Optional: print first chunk preview
    print("📌 Preview of first chunk:")
    print(texts[0].page_content[:500], '...' if len(texts[0].page_content) > 500 else '')

    print("🧠 Step 3: Generating embeddings using HuggingFace model...")
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    print("✅ Embedding model loaded.")

    print("📦 Step 4: Creating and saving FAISS vector store...")
    db = FAISS.from_documents(texts, embeddings)
    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ Vector database saved to: {DB_FAISS_PATH}")

if __name__ == "__main__":
    print("🚀 Starting ingestion pipeline...")
    create_vector_db()
    print("🏁 Ingestion completed successfully.")
