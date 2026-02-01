from flask import Flask, request, jsonify, render_template
import os
import time
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEndpoint  # new LLM wrapper

from sentence_transformers import SentenceTransformer

# ------------------ ENV ------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN not set in .env")

# ------------------ LOAD MODELS ONCE ------------------
print(" Loading embedding model...")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
sentence_model = SentenceTransformer(EMBED_MODEL_NAME)
embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL_NAME)

print(" Loading FAISS index...")
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever(search_kwargs={"k": 4})

print(" Loading LLM (HF Endpoint)...")

llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/Qwen/QwQ-32B-Preview",  # router endpoint for that model
    huggingfacehub_api_token=HF_TOKEN,
    task="text-generation",
    model_kwargs={
        "temperature": 0.3,
        "max_new_tokens": 512
    }
)


# ------------------ PROMPT ------------------
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant for SR University.

Answer the question using ONLY the provided context.
If the answer is not present, reply exactly with:

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
    return_source_documents=False
)

print(" All models loaded")

# ------------------ FLASK APP ------------------
app = Flask(__name__)

def get_answer(question: str) -> str:
    if not question:
        return "Please enter a valid question."

    retries = 2
    for attempt in range(retries):
        try:
            response = qa_chain.invoke({"query": question})
            answer = response.get("result", "").strip()
            return answer if answer else "No answer generated."
        except Exception as e:
            print(f" LLM error (attempt {attempt+1}): {e}")
            time.sleep(1)

    return "The assistant is temporarily unavailable. Please try again."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = data.get("message", "").strip()
    reply = get_answer(message)
    return jsonify({"response": reply})

# ------------------ ENTRYPOINT ------------------
if __name__ == "__main__":
    app.run(debug=True)
