from flask import Flask, request, jsonify, render_template
import time

app = Flask(__name__)

def query_model(user_input):
    time.sleep(1)  # Simulate "thinking" time
    return f"Bot: You said '{user_input}'"

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
