import os
import base64
import requests
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from transformers import AutoTokenizer

# ==== Config ====
os.environ['ALLOW_DANGEROUS_DESERIALIZATION'] = "true"
DB_FAISS_PATH = 'vectorstore/db_faiss'

OPENROUTER_API_KEY = "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"
VISION_MODEL = "mistralai/mistral-small-3.2-24b-instruct:free"

# ==== Tokenizer for Truncation ====
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def truncate_text_to_tokens(text: str, max_tokens: int = 512):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)

# ==== Prompt Template ====
custom_prompt_template = """
You are a helpful assistant. Use the following extracted document information to answer the question.

If the answer is in multiple parts (e.g., general and technical requirements), include both parts clearly and completely.

Always extract only from the provided context. Do not make up or guess answers.

Context:
{context}

Question:
{question}

Answer in a clear, organized list format:
"""

# ==== Load Local LLM ====
def load_llm():
    print("🧠 Loading local LLM...")
    llm = CTransformers(
        model="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    print("✅ LLM loaded.")
    return llm

# ==== Load FAISS Vector DB ====
def load_vector_db():
    print("📚 Loading FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    print("✅ Vector DB loaded.")
    return db

# ==== Main RAG Answer Function ====
def get_answer(query: str):
    print(f"\n❓ Question: {query}")
    
    db = load_vector_db()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = load_llm()

    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

    print("🔎 Retrieving top document chunk(s)...")
    docs = retriever.get_relevant_documents(query)
    for i, doc in enumerate(docs):
        content = doc.page_content.strip()
        print(f"\n--- Chunk {i+1} ---\n{content[:500]}{'...' if len(content) > 500 else ''}")

    raw_context = "\n\n".join(doc.page_content for doc in docs)
    truncated_context = truncate_text_to_tokens(raw_context, max_tokens=512)

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"context": truncated_context, "question": query})

    print("\n✅ Answer:\n" + response['text'])
    return {
        "result": response['text'],
        "source_documents": docs
    }

# ==== Vision-to-RAG Function ====
def analyze_image_and_get_answer(image_path: str, user_instruction: str = "Analyze this image and describe what you see."):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    print("🖼️ Sending image to Vision API...")
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)

    if res.status_code != 200:
        raise Exception(f"❌ Vision API Error {res.status_code}: {res.text}")

    vision_response = res.json()
    analysis_text = vision_response["choices"][0]["message"]["content"]
    print("🧾 Vision Model Analysis:\n", analysis_text)

    return get_answer(analysis_text)
