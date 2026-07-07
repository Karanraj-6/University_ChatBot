from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from huggingface_hub import InferenceClient

# ---------------- ENV ----------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN missing")

CHAT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------- HF INFERENCE CLIENT ----------------
print("Loading Hugging Face Inference Client...")
client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)
print("Inference client loaded.")

# ---------------- EMBEDDINGS ADAPTER ----------------
class HFHubEmbeddings(Embeddings):
    """Wraps huggingface_hub.InferenceClient.feature_extraction to satisfy
    LangChain's Embeddings interface (embed_documents / embed_query)."""

    def __init__(self, client: InferenceClient, model_id: str):
        self.client = client
        self.model_id = model_id

    def _embed(self, text: str) -> list[float]:
        vec = self.client.feature_extraction(text, model=self.model_id)
        arr = np.array(vec, dtype=np.float32)
        # Some endpoints return token-level embeddings (2D) instead of a
        # pooled sentence embedding (1D) — mean-pool if needed.
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        return arr.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

embeddings = HFHubEmbeddings(client, EMBED_MODEL_ID)

# ---------------- LOAD FAISS ----------------
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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

print("All models loaded")

# ---------------- FLASK ----------------
app = Flask(__name__)

def get_answer(question: str) -> str:
    if not question.strip():
        return "Please enter a valid question."

    docs = retriever.invoke(question)
    context_str = format_docs(docs)
    formatted_prompt = PROMPT.format(context=context_str, question=question)

    completion = client.chat.completions.create(
        model=CHAT_MODEL_ID,
        messages=[{"role": "user", "content": formatted_prompt}],
        temperature=0.3,
        max_tokens=1024,
    )

    answer = completion.choices[0].message.content
    return answer.strip() if answer else ""

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

        if "rate limit" in error_msg.lower() or "429" in error_msg:
            status_code = 429
            friendly_msg = "Hugging Face API rate limit reached. Please wait a minute before trying again."
        elif "loading" in error_msg.lower() or "unavailable" in error_msg.lower() or "503" in error_msg:
            status_code = 503
            friendly_msg = "The Hugging Face model is currently loading or warming up. Please try again in a few seconds."
        elif "authorization" in error_msg.lower() or "token" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
            status_code = 401
            friendly_msg = "Authorization failed. Please check if your HUGGINGFACEHUB_API_TOKEN is valid."
        elif "not supported" in error_msg.lower() or "provider" in error_msg.lower():
            status_code = 400
            friendly_msg = f"Model/provider issue: {error_msg}"
        else:
            status_code = 500
            friendly_msg = f"Sorry, an error occurred while processing your request: {error_msg}"

        return jsonify({"error": friendly_msg}), status_code

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))