from flask import Flask, request, jsonify, render_template
import os
import time
import requests
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# ---------------- ENV ----------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN missing")

# ---------------- LOAD MODELS ----------------
print("Loading embeddings (remote HuggingFace endpoint)...")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEndpointEmbeddings(
    model=EMBED_MODEL,
    huggingfacehub_api_token=HF_TOKEN
)

print("Loading FAISS index...")
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever(search_kwargs={"k": 4})

# ---------------- PROMPT ----------------
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant for SR University.

Answer ONLY using the context below.
If the answer is not found, reply exactly:

"The context does not provide sufficient information. Please contact SR University support.
Phone: 0870-281-8333 / 8311
Email: info@sru.edu.in"

Context:
{context}

Question:
{question}

Answer:
"""
)

# Helper function to format the retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

print("All models loaded")

# ---------------- FLASK ----------------
app = Flask(__name__)

def get_answer(question: str) -> str:
    if not question:
        return "Please enter a valid question."

    # 1. Retrieve relevant documents
    docs = retriever.invoke(question)
    context_str = format_docs(docs)

    # 2. Format the prompt
    formatted_prompt = PROMPT.format(context=context_str, question=question)

    # 3. Call HuggingFace Inference API directly (avoids routing / task-mapping ValueErrors)
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "temperature": 0.3,
            "max_new_tokens": 1024,
            "return_full_text": False
        }
    }
    
    response = requests.post(
        "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"HuggingFace API error (Status {response.status_code}): {response.text}")
        
    result_json = response.json()
    if isinstance(result_json, list) and len(result_json) > 0:
        generated_text = result_json[0].get("generated_text", "")
    elif isinstance(result_json, dict):
        generated_text = result_json.get("generated_text", "")
    else:
        generated_text = str(result_json)
        
    return generated_text.strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json.get("message", "")
    if not msg.strip():
        return jsonify({"response": "Please type a message."})

    try:
        response = get_answer(msg)
        return jsonify({"response": response})
    except Exception as e:
        error_msg = str(e)
        print("LLM Error:", error_msg)
        
        # Categorize Hugging Face API errors
        if "rate limit" in error_msg.lower() or "429" in error_msg:
            status_code = 429
            friendly_msg = "Hugging Face API rate limit reached. Please wait a minute before trying again."
        elif "loading" in error_msg.lower() or "unavailable" in error_msg.lower() or "503" in error_msg or "estimated_time" in error_msg:
            status_code = 503
            friendly_msg = "The Hugging Face model is currently loading or warming up. Please try again in a few seconds."
        elif "authorization" in error_msg.lower() or "token" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
            status_code = 401
            friendly_msg = "Authorization failed. Please check if your HUGGINGFACEHUB_API_TOKEN is valid."
        else:
            status_code = 500
            friendly_msg = f"Sorry, an error occurred while processing your request: {error_msg}"
            
        return jsonify({"error": friendly_msg}), status_code

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
