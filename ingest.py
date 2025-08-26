import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

DB_FAISS_PATH = 'vectorstore/db_faiss'
EXCEL_FILE = "data/Trasnlate.xlsx"  # ملف البنود الجديد

# Create vector database
def create_vector_db():
    print(f"🔍 Step 1: Loading Excel file: {EXCEL_FILE}")
    df = pd.read_excel(EXCEL_FILE)

    print(f"✅ Loaded {len(df)} rows from Excel.")

    # حوّل كل صف إلى Document
    documents = []
    for i, row in df.iterrows():
        content = f"""
        Item No.: {row.get('Item No.', i+1)}
        Item Name: {row.get('Item Name', '')}
        Requirement No.: {row.get('Requirement No.', '')}
        Requirement Text: {row.get('Requirement Text', '')}
        Definition (Saudi Code): {row.get('Definition according to the Saudi Code', '')}
        Suggested Repair Method: {row.get('Suggested Repair Method', '')}
        Estimated Cost (SAR): {row.get('Estimated Cost (SAR)', '')}
        """

        doc = Document(
            page_content=content,
            metadata={
                "row_index": i,
                "item_name": row.get("Item Name", ""),
                "requirement_text": row.get("Requirement Text", ""),
                "definition": row.get("Definition according to the Saudi Code", "")
            }
        )
        documents.append(doc)

    print(f"📄 Converted {len(documents)} rows into documents.")

    print("🧠 Step 2: Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    print("✅ Embedding model loaded.")

    print("📦 Step 3: Creating and saving FAISS vector store...")
    db = FAISS.from_documents(documents, embeddings)
    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ Vector database saved to: {DB_FAISS_PATH}")

if __name__ == "__main__":
    print("🚀 Starting ingestion pipeline...")
    create_vector_db()
    print("🏁 Ingestion completed successfully.")
