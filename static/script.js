// Get references to UI elements
const chatButton = document.getElementById("chatbot-button");
const chatWindow = document.getElementById("chat-window");
const closeButton = document.getElementById("close-btn");
const sendButton = document.getElementById("send-btn");
const chatInput = document.getElementById("chat-input");
const chatBody = document.getElementById("chat-body");

// Show chat window when button is clicked
chatButton.addEventListener("click", () => {
    chatWindow.style.display = "flex"; // Show the chat window
    chatButton.style.display = "none"; // Hide the floating button
    chatInput.focus(); // Focus on the input field when the chat window opens
});

// Close chat window
closeButton.addEventListener("click", () => {
    chatWindow.style.display = "none"; // Hide the chat window
    chatButton.style.display = "flex"; // Show the floating button
});

// Enable dragging for the floating button
chatButton.addEventListener("dragstart", (e) => {
    e.dataTransfer.setData("text/plain", null); // Required for dragstart to work in Firefox
});

chatButton.addEventListener("dragend", (e) => {
    const rect = chatButton.getBoundingClientRect();
    chatButton.style.position = "fixed";
    chatButton.style.top = `${e.clientY - rect.height / 2}px`;
    chatButton.style.left = `${e.clientX - rect.width / 2}px`;
});

// Function to send a message
async function sendMessage() {
    const userMessage = chatInput.value.trim();
    if (!userMessage) return;

    // Display user message
    const userMessageDiv = document.createElement("div");
    userMessageDiv.classList.add("user-message");
    userMessageDiv.textContent = `You: ${userMessage}`;
    chatBody.appendChild(userMessageDiv);
    chatInput.value = "";

    // Display "thinking..." indicator with animation
    const thinkingDiv = document.createElement("div");
    thinkingDiv.classList.add("thinking");
    thinkingDiv.innerHTML = `<div class="thinking-dots">...</div>`;
    chatBody.appendChild(thinkingDiv);
    chatBody.scrollTop = chatBody.scrollHeight;

    try {
        // Call Flask backend
        const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: userMessage }),
        });
        const data = await response.json();

        // Remove "thinking..." and display bot response with image
        chatBody.removeChild(thinkingDiv);
        const botMessageDiv = document.createElement("div");
        botMessageDiv.classList.add("bot-message");
        botMessageDiv.innerHTML = `
            <img src="/static/chatbot_icon.png" alt="Bot" class="bot-avatar">
            <span class="bot-text">${data.response}</span>
        `;
        chatBody.appendChild(botMessageDiv);
    } catch (error) {
        console.error("Error:", error);
        chatBody.removeChild(thinkingDiv);
        const errorDiv = document.createElement("div");
        errorDiv.classList.add("bot-message");
        errorDiv.textContent = "Bot: Sorry, something went wrong!";
        chatBody.appendChild(errorDiv);
    }

    // Scroll to the bottom
    chatBody.scrollTop = chatBody.scrollHeight;
}

// Add event listener for the "Send" button
sendButton.addEventListener("click", sendMessage);

// Add event listener for the "Enter" key
chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        event.preventDefault(); // Prevent the default form submission behavior
        sendMessage(); // Call the sendMessage function
    }
});
