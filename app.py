from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from dotenv import load_dotenv
from collections import defaultdict

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from huggingface_hub import InferenceClient

# ---------------- ENV ----------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN missing")

CHAT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

client = InferenceClient(provider="auto", api_key=HF_TOKEN)

# ---------------- EMBEDDINGS ----------------
class HFHubEmbeddings(Embeddings):
    def __init__(self, client, model_id):
        self.client = client
        self.model_id = model_id

    def _embed(self, text):
        vec = self.client.feature_extraction(text, model=self.model_id)
        arr = np.array(vec, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        return arr.tolist()

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)

embeddings = HFHubEmbeddings(client, "sentence-transformers/all-MiniLM-L6-v2")

# ---------------- FAISS ----------------
print("Loading FAISS index...")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 6})

# ---------------- MEMORY (Per Session) ----------------
conversation_history = defaultdict(list)  # session_id -> list of messages

# ---------------- DETAILED SYSTEM PROMPT (300+ words) ----------------
SYSTEM_PROMPT = """
You are a professional, friendly, and highly knowledgeable AI assistant for SR University (SRU), Warangal, Telangana.

Your role is to help students, parents, faculty, and visitors with accurate information about SR University.

Core Rules:
- Answer ONLY using the information provided in the "Context" section.
- If the requested information is not present in the context, reply with this exact sentence:
  "The context does not provide sufficient information. Please contact SR University support."

- Be polite, clear, and professional in your tone.
- Use bullet points or numbered lists when explaining multiple points.
- Provide complete and helpful answers when the context allows it.
- Never make up information, invent facts, or guess.
- If the user asks about placement, CTC, rankings, admissions, facilities, or academic programs, be as detailed as the context allows.
- Maintain conversation context naturally. Remember previous questions in the same chat session.

You have access to the latest Student Handbook and official university information through the provided context.

Context:
{context}

Question: {question}

Answer:
"""

# ---------------- FLASK ----------------
app = Flask(__name__)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_answer(session_id: str, question: str) -> str:
    if not question.strip():
        return "Please enter a valid question."

    try:
        # Get last 8 messages for context
        history = conversation_history[session_id][-8:]

        # Retrieve fresh documents
        docs = retriever.invoke(question)
        context_str = format_docs(docs)

        # Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add history
        messages.extend(history)

        # Add current question with context
        messages.append({
            "role": "user",
            "content": f"Context:\n{context_str}\n\nQuestion: {question}"
        })

        # Generate response
        completion = client.chat.completions.create(
            model=CHAT_MODEL_ID,
            messages=messages,
            temperature=0.2,
            max_tokens=900,
            top_p=0.85,
        )

        answer = completion.choices[0].message.content.strip()

        # Save to history
        conversation_history[session_id].append({"role": "user", "content": question})
        conversation_history[session_id].append({"role": "assistant", "content": answer})

        # Optional: Limit history size
        if len(conversation_history[session_id]) > 20:
            conversation_history[session_id] = conversation_history[session_id][-20:]

        return answer

    except Exception as e:
        print("Error:", str(e))
        return "Sorry, an error occurred while processing your request. Please try again."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json.get("message", "").strip()
    session_id = request.json.get("session_id", "default")

    if not msg:
        return jsonify({"response": "Please type a message."})

    response = get_answer(session_id, msg)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))