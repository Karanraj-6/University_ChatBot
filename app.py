from flask import Flask, request, jsonify, render_template
import time
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
import faiss
import pickle
import os
from langchain.embeddings import SentenceTransformerEmbeddings  
from sentence_transformers import SentenceTransformer
load_dotenv()
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def get_answer(user_question):
    # index_path = 'faiss_index'  # Path to your faiss_index folder
    # faiss_index = faiss.read_index(os.path.join(index_path, 'index.faiss'))

    # # Load the embeddings model stored in index.pkl
    # with open(os.path.join(index_path, 'index.pkl'), 'rb') as f:
    #     embeddings_model = pickle.load(f)

    # # Now you can create a retriever to query the FAISS index
    # retriever = FAISS(index=faiss_index, embeddings=embeddings_model)
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    embeddings_model = SentenceTransformerEmbeddings(model_name=model_name)
    new_db = FAISS.load_local("faiss_index", embeddings_model,allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever()
    # Set up the HuggingFace LLM
    llm = HuggingFaceHub(
    repo_id="Qwen/QwQ-32B-Preview",  
    huggingfacehub_api_token=hf_api_token,
    model_kwargs={"temperature": 0.4, "max_length": 5000},
    task="text-generation"  
  )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an AI assistant made for SR University. Use the provided context to answer the question.
        Provide only the answer, not the context or instructions, Your great at providing revelent and meaningful answer to user. search for answer more deeply and clearly and check one again for verification.and make sure that you dont stop sentences generation in middle while providing answer to user. If the context does not contain the answer, respond with: 
        "The context does not provide sufficient information. I couldn't find enough information. Please check back later or contact the SR University support team for more details. Contact: 0(870) 281-8333/8311 MAIL : info@sru.edu.in " Do not add any other information.

        Context: {context}
        Question: {question}

        Helpful Answer:
        """ 
    )

    # Set up the QA chain with memory
    qa_chain_with_memory = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff", 
        chain_type_kwargs={"prompt": prompt_template},  
        return_source_documents=False,  
        output_key="result"  
    )

    # Start the chatbot interaction
    print("Welcome to the Campus Assistant! Type 'exit' to end the conversation.\n")

    user_query = user_question

    # Exit the conversation if the user types 'exit' or 'quit'
    if user_query.lower() in ["exit", "quit"]:
        return ("Thank you for using the Campus Assistant. Bye for now 😊!")
        

    try:
        response = qa_chain_with_memory({"query": user_query})

        answer = response.get('result', 'No answer provided').strip()

        helpful_answer = answer.split("Helpful Answer:")[-1].strip()

        return (helpful_answer)

    except Exception as e:
        return ("An error occurred:", e)
app = Flask(__name__)


def query_model(user_question):
    time.sleep(1)   # Simulate "thinking" time
    answer = get_answer(user_question)  # Return the answer from the function here
    return answer
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "")
    if user_message.strip() == "":
        return jsonify({"response": "Please type a message."})
    
    # Process the message with the model
    bot_response = query_model(user_message)
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(debug=True)
