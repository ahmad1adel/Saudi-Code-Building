import os
import base64
import requests
import re
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from transformers import AutoTokenizer

# ==== Config ====
os.environ['ALLOW_DANGEROUS_DESERIALIZATION'] = "true"
DB_FAISS_PATH = 'vectorstore/db_faiss'

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
VISION_MODEL = "mistralai/mistral-small-3.2-24b-instruct:free"

# ==== Tokenizer ====
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def truncate_text_to_tokens(text: str, max_tokens: int = 512):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    return tokenizer.decode(tokens, skip_special_tokens=True)

# ==== Prompt Template ====
custom_prompt_template = """
You are a helpful assistant. Use the following extracted document information to answer the question.

Answer clearly and in an organized list format.

Context:
{context}

Question:
{question}

Answer:
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
    print("✅ LLM Loaded.")
    return llm

# ==== Load FAISS ====
def load_vector_db():
    print("📚 Loading FAISS vector DB...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    print("✅ FAISS DB Loaded.")
    return db

# ==== Main RAG Answer Function ====
def get_answer(query: str):
    print(f"\n❓ [RAG] Query: {query}")
    db = load_vector_db()
    retriever = db.as_retriever(search_kwargs={"k": 2})
    llm = load_llm()

    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

    print("🔎 Retrieving relevant documents...")
    docs = retriever.get_relevant_documents(query)
    for i, doc in enumerate(docs):
        print(f"\n--- RAG Chunk {i+1} ---")
        print(doc.page_content[:500], "..." if len(doc.page_content) > 500 else "")

    raw_context = "\n\n".join(doc.page_content for doc in docs)
    truncated_context = truncate_text_to_tokens(raw_context, max_tokens=512)

    print("\n📦 Truncated Context (to 512 tokens):")
    print(truncated_context[:700], "..." if len(truncated_context) > 700 else "")

    chain = LLMChain(llm=llm, prompt=prompt)
    print("\n⚡ Sending to LLM...")
    response = chain.invoke({"context": truncated_context, "question": query})

    text = response['text']
    print("\n🧾 Raw LLM Answer:")
    print(text)

    sentences = re.split(r'(?<=[.!?]) +', text)
    shortened = " ".join(sentences[:4])

    print("\n✂️ Shortened Answer (first 3-4 sentences):")
    print(shortened)

    return {"result": shortened, "source_documents": docs}

# ==== Electrical Checklist (English) ====
electrical_checklist = """
Check the following items in the image and return each item with one of:
- "Passed"
- "Partial pass" + describe issue and possible fix
- "Not passed"
- "Unmeasurable"

Items:
1. Quality of external sockets
2. Load capacity of electrical sockets
3. Proper installation of socket wiring
4. Covering of external sockets
5. Finishing quality around sockets
6. Alignment of sockets horizontally
7. Safety of visible or external wiring
8. Power supply according to plan
9. Separation of lighting circuits from sockets
10. Cable size suitable for electrical load
11. Grounding connection
12. Load distribution on breakers
13. Labeling of electrical circuits
14. Protection against overload
15. Lighting distribution
16. Height of sockets
17. Consistency in socket distribution
18. Distribution of electrical switches
19. Safety of electrical installations
20. Distance between switches and doors
21. Organization of switches by function
22. Condition of electrical switches
23. Proper fixing of switches
24. Switch height above floor level
25. Distribution of switches inside rooms
26. Quality of switch cover installation
27. Consistency of switches with wall design
28. Compliance of switches with electrical plans
29. Safety of control switches
30. Compliance of switch covers with standards
31. Socket level from floor
32. Location of switches
33. Electrical load distribution
34. Electrical grounding system
"""

# ==== Vision-to-RAG Function ====
def analyze_image_and_get_answer(image_path: str):
    print(f"\n🖼️ Reading Image: {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    payload = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": electrical_checklist},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        }],
        "stream": False
    }

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    print("🚀 Sending image to Vision API...")
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers)

    print("\n📩 [Vision API Raw Response]:")
    print(res.text)

    if res.status_code != 200:
        raise Exception(f"❌ Vision API Error {res.status_code}: {res.text}")

    vision_response = res.json()
    if "choices" not in vision_response:
        raise Exception(f"❌ Unexpected API response format: {vision_response}")

    analysis_text = vision_response["choices"][0]["message"]["content"]
    print("\n🧾 [Vision Raw Output]:\n", analysis_text)

    structured = []

    for line in analysis_text.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue

        # ✅ Regex now handles **bold** and ":" or "-" separators
        match = re.match(
            r'^\d+\.\s*\*{0,2}(.+?)\*{0,2}\s*[:\-]\s*\*{0,2}(Passed|Partial pass|Not passed|Unmeasurable)\*{0,2}(.*)$',
            line,
            re.IGNORECASE
        )

        if match:
            item = match.group(1).strip()
            status = match.group(2).strip()
            notes = match.group(3).strip()

            structured.append({
                "item": item,
                "status": status,
                "notes": notes
            })
            print(f"✅ Parsed Item: {item} | Status: {status} | Notes: {notes}")

    # 🔎 Run RAG for Partial/Not passed
    for r in structured:
        if r["status"] in ["Partial pass", "Not passed"]:
            rag_result = get_answer(f"SBC requirement for: {r['item']}")
            r["notes"] += f"\n\nSBC Ref: {rag_result['result']}"

    print("\n✅ [Final Structured Results]:")
    for r in structured:
        print(r)

    return {"structured": structured}
