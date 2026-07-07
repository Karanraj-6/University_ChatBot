from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from huggingface_hub import InferenceClient

# ---------------- ENV ----------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN missing")

CHAT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"   # Change to 3B if you want faster responses

# ---------------- HF CLIENT ----------------
print("Initializing Hugging Face Inference Client...")
client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN,
)

# ---------------- CUSTOM EMBEDDINGS ----------------
class HFHubEmbeddings(Embeddings):
    def __init__(self, client: InferenceClient, model_id: str):
        self.client = client
        self.model_id = model_id

    def _embed(self, text: str):
        vec = self.client.feature_extraction(text, model=self.model_id)
        arr = np.array(vec, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        return arr.tolist()

    def embed_documents(self, texts: list[str]):
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str):
        return self._embed(text)

embeddings = HFHubEmbeddings(client, "sentence-transformers/all-MiniLM-L6-v2")

# ---------------- LOAD FAISS ----------------
print("Loading FAISS index...")
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever(search_kwargs={"k": 5})

print("System ready.")

# ---------------- FLASK ----------------
app = Flask(__name__)

def get_answer(question: str) -> str:
    if not question or not question.strip():
        return "Please enter a valid question."

    try:
        # Retrieve relevant documents
        docs = retriever.invoke(question)
        context_str = "\n\n".join(doc.page_content for doc in docs)

        # Call model with system prompt
        completion = client.chat.completions.create(
            model=CHAT_MODEL_ID,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant for SR University.\n"
                        "Answer ONLY using the provided context.\n"
                        "If the answer cannot be found in the context, reply exactly with this sentence:\n\n"
                        "The context does not provide sufficient information. "
                        "Please contact SR University support.\n"
                        "Phone: 0870-281-8333 / 8311\n"
                        "Email: info@sru.edu.in"
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_str}\n\nQuestion: {question}"
                }
            ],
            temperature=0.2,
            max_tokens=800,
            top_p=0.85,
        )

        answer = completion.choices[0].message.content.strip()

        # Fallback if model ignores instructions
        if len(answer) < 15 and "context does not provide" not in answer.lower():
            answer = "The context does not provide sufficient information. Please contact SR University support."

        return answer

    except Exception as e:
        print("LLM Error:", str(e))
        return "Sorry, an error occurred while processing your request. Please try again."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json.get("message", "").strip()
    if not msg:
        return jsonify({"response": "Please type a message."})

    response = get_answer(msg)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))