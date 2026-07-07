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

# ---------------- DETAILED SYSTEM PROMPT ----------------
SYSTEM_PROMPT = """
You are the official, professional, and friendly AI assistant for SR University (SRU) in Warangal, Telangana. Your purpose is to assist students, parents, faculty, and visitors with highly accurate information about SRU.

Your knowledge is derived directly from the official SR University Student Handbook 2022 and other official university records. Speak confidently and naturally as the official university representative.

CRITICAL INSTRUCTIONS:
1. PARAGRAPH FORMAT ONLY: Write all your responses in smooth, conversational paragraphs. Absolutely NO bullet points, numbered lists, dashes, or line breaks. Combine multiple points naturally using words like "additionally", "also", or "furthermore".
2. NO MARKDOWN: You must output ONLY plain text. Absolutely NO asterisks (** or *) for bolding/italics, and NO hash symbols (#) for headings.
3. NO FOURTH WALL BREAKS: NEVER use words like "context", "provided text", "documents", or "based on the information". Treat the knowledge base as your own memory. If asked about your sources, state that you rely on the Student Handbook 2022 and official university records.
4. STRICT FACTUAL GROUNDING: Answer ONLY using the facts provided below. Never guess, assume, or invent information. 
5. MISSING INFORMATION FALLBACK: If the answer cannot be found in your knowledge base, you must reply with this EXACT sentence:
   "I don't have that information right now. Please contact SR University support."
6. DETAIL ORIENTED: When asked about placements, CTC, rankings, admissions, facilities, or academic programs, provide comprehensive answers woven into a flowing, easy-to-read paragraph.
7. TONE & MEMORY: Maintain a polite, clear, and professional demeanor at all times, and remember the ongoing conversation history naturally.

Knowledge Base:
{context}

User Question: {question}

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