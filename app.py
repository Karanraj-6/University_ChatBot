from flask import Flask, request, jsonify, render_template
import time
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import SentenceTransformerEmbeddings  
from sentence_transformers import SentenceTransformer

# ------------------ ENV ------------------
load_dotenv()
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ------------------ LOAD ONCE (IMPORTANT) ------------------
print("Loading embeddings model...")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
sentence_model = SentenceTransformer(model_name)
embeddings_model = SentenceTransformerEmbeddings(model_name=model_name)

print("Loading FAISS index...")
db = FAISS.load_local(
    "faiss_index",
    embeddings_model,
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever()

print("Loading LLM...")
llm = HuggingFaceHub(
    repo_id="Qwen/QwQ-32B-Preview",
    huggingfacehub_api_token=hf_api_token,
    model_kwargs={
        "temperature": 0.4,
        "max_length": 5000
    },
    task="text-generation"
)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant made for SR University. Use the provided context to answer the question.
Provide only the answer, not the context or instructions.

If the context does not contain the answer, respond with:
"The context does not provide sufficient information. I couldn't find enough information. Please check back later or contact the SR University support team for more details. Contact: 0(870) 281-8333/8311 MAIL : info@sru.edu.in"

Context: {context}
Question: {question}

Helpful Answer:
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=False,
    output_key="result"
)

print("All models loaded successfully")

# ------------------ APP ------------------
app = Flask(__name__)

def get_answer(user_question: str) -> str:
    """Return answer to user_question. Always returns a string."""
    if user_question.lower() in ["exit", "quit"]:
        return "Thank you for using the Campus Assistant. Bye for now!"

    try:
        response = qa_chain({"query": user_question})
        answer = response.get("result", "")
        if not answer:
            return "No answer generated."
        return str(answer)

    except Exception as e:
        # Safely convert exception to string
        return "The assistant is temporarily unavailable. Please try again."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please type a message."})

    # Always returns string, so no need for try/except here
    bot_response = get_answer(user_message)

    return jsonify({"response": bot_response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # use Render's PORT or fallback
    app.run(host="0.0.0.0", port=port)
