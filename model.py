from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
import faiss
import pickle
import os

load_dotenv()
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def get_answer(user_question):
    index_path = 'faiss_index'  # Path to your faiss_index folder
    faiss_index = faiss.read_index(os.path.join(index_path, 'index.faiss'))

    # Load the embeddings model stored in index.pkl
    with open(os.path.join(index_path, 'index.pkl'), 'rb') as f:
        embeddings_model = pickle.load(f)

    # Now you can create a retriever to query the FAISS index
    retriever = FAISS(index=faiss_index, embeddings_model=embeddings_model)

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

# Main interaction loop for the Flask app
if __name__ == "__main__":
    print("Welcome to the Campus Assistant! Type 'exit' to end the conversation.\n")
    
    while True:
        user_query = input("Your Question: ")
        
        # Exit the conversation if the user types 'exit' or 'quit'
        if user_query.lower() in ["exit", "quit"]:
            print("Thank you for using the Campus Assistant. Bye for now 😊!")
            break

        # Get the answer and print it
        answer = get_answer(user_query)
        print("Answer:", answer)
