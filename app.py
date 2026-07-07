from flask import Flask, request, jsonify, render_template
import time
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
import os
from langchain_community.embeddings import HuggingFaceHubEmbeddings  

load_dotenv()
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize models globally for optimization (loaded only once on startup)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_model = HuggingFaceHubEmbeddings(
    repo_id=model_name,
    huggingfacehub_api_token=hf_api_token
)
new_db = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
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

# Set up the QA chain globally
qa_chain_with_memory = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff", 
    chain_type_kwargs={"prompt": prompt_template},  
    return_source_documents=False,  
    output_key="result"  
)

def get_answer(user_question):
    # Exit the conversation if the user types 'exit' or 'quit'
    if user_question.lower() in ["exit", "quit"]:
        return "Thank you for using the Campus Assistant. Bye for now 😊!"

    response = qa_chain_with_memory({"query": user_question})
    answer = response.get('result', 'No answer provided').strip()
    helpful_answer = answer.split("Helpful Answer:")[-1].strip()
    return helpful_answer

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
    
    try:
        # Process the message with the model
        bot_response = query_model(user_message)
        return jsonify({"response": bot_response})
    except Exception as e:
        error_msg = str(e)
        
        # Categorize the Hugging Face API errors
        if "rate limit" in error_msg.lower() or "429" in error_msg:
            status_code = 429
            friendly_msg = "Hugging Face API rate limit reached. Please wait a minute before trying again."
        elif "loading" in error_msg.lower() or "503" in error_msg or "estimated_time" in error_msg:
            status_code = 503
            friendly_msg = "The Hugging Face model is currently loading or warming up. Please try again in a few seconds."
        elif "authorization" in error_msg.lower() or "token" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
            status_code = 401
            friendly_msg = "Authorization failed. Please check if your HUGGINGFACEHUB_API_TOKEN is valid."
        else:
            status_code = 500
            friendly_msg = f"Sorry, an error occurred while processing your request: {error_msg}"
            
        return jsonify({"error": friendly_msg}), status_code

if __name__ == '__main__':
    app.run(debug=True)
