document.addEventListener('DOMContentLoaded', () => {
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');
    
    // Backend API URL
    const apiUrl = 'http://localhost:8000/chat';
    
    // Function to add a message to the chat
    function addMessage(message, isUser = false) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        messageElement.classList.add(isUser ? 'user' : 'bot');
        
        const messageContent = document.createElement('div');
        messageContent.classList.add('message-content');
        messageContent.textContent = message;
        
        messageElement.appendChild(messageContent);
        chatMessages.appendChild(messageElement);
        
        // Scroll to the bottom of the chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to send a message to the backend
    async function sendMessage(message) {
        if (!message.trim()) return;
        
        // Add user message to the chat
        addMessage(message, true);
        
        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: message }),
            });
            
            if (!response.ok) {
                throw new Error('Failed to get response from server');
            }
            
            const data = await response.json();
            
            // Add bot response to the chat
            addMessage(data.response);
        } catch (error) {
            console.error('Error:', error);
            addMessage('Sorry, there was an error communicating with the server.', false);
        }
        
        // Clear the input field
        messageInput.value = '';
    }
    
    // Send message when the send button is clicked
    sendButton.addEventListener('click', () => {
        sendMessage(messageInput.value);
    });
    
    // Send message when Enter key is pressed
    messageInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage(messageInput.value);
        }
    });
}); 