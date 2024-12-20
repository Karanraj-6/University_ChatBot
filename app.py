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
        repo_id="tiiuae/falcon-7b-instruct",  
        huggingfacehub_api_token=hf_api_token,
        model_kwargs={"temperature": 0.4, "max_length": 512},
        task="text-generation"  # Specify the task as text generation
    )

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an AI assistant made for SR University. Use the provided context to answer the question.
        Provide only the answer, not the context or instructions. If the context does not contain the answer, respond with: 
        "The context does not provide sufficient information.", and you are not permitted to provide information about anything other than SR university.

        Context: {context}
        Question: {question}

        Helpful Answer:
        """  # Ensure only the answer is returned
    )

    # Set up the QA chain with memory
    qa_chain_with_memory = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # Default option for combining multiple documents
        chain_type_kwargs={"prompt": prompt_template},  # Use the prompt template here
        return_source_documents=False,  # Set this to False if you do not want source documents
        output_key="result"  # Explicitly set output_key to 'result'
    )

    try:
        # Run the query through the QA chain
        response = qa_chain_with_memory.run({"query": user_question})

        # Extract answer from the response
        helpful_answer_index = response.find("Helpful Answer:")
        if helpful_answer_index != -1:
            answer = response[helpful_answer_index + len("Helpful Answer:"):].strip()
        else:
            answer = "Helpful answer not found"

        return answer  # Return the result as the function's output
    
    except Exception as e:
        return f"An error occurred: {e}"  # Return error message

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
