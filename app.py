from flask import Flask, request, jsonify, render_template
import os, time
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEndpoint

# ---------------- ENV ----------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN missing")

# ---------------- LOAD MODELS ----------------
print("Loading embeddings...")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

print("Loading FAISS index...")
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever(search_kwargs={"k": 4})

print("Loading LLM...")
llm = HuggingFaceEndpoint(
    repo_id="Qwen/QwQ-32B-Preview",   
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.3,
    max_new_tokens=1024,
)

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

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
)

print("All models loaded")

# ---------------- FLASK ----------------
app = Flask(__name__)

def get_answer(question: str) -> str:
    if not question:
        return "Please enter a valid question."

    try:
        res = qa_chain.invoke({"query": question})
        return res.get("result", "").strip() or "No answer generated."
    except Exception as e:
        print("LLM error:", e)
        return "The assistant is temporarily unavailable. Please try again."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    msg = request.json.get("message", "")
    return jsonify({"response": get_answer(msg)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
