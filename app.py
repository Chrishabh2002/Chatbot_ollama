import streamlit as st
import ollama
import numpy as np
import pdfplumber
from io import BytesIO

# Define the Reinforcement Learning agent with a training mechanism
class ReinforcementLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

# Initialize the reinforcement learning agent
state_size = 10  # Number of possible states
action_size = 5  # Number of possible actions
agent = ReinforcementLearningAgent(state_size, action_size)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ''
    with pdfplumber.open(BytesIO(file.read())) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text

# Function to get chatbot response from Ollama model
@st.cache_data
def get_chatbot_response(prompt, context):
    try:
        context = context[:2000]  # Limit context length
        client = ollama.Client()
        result = client.chat(
            model="llama3.1",
            messages=[
                {"role": 'system', "content": "You are a helpful assistant."},
                {"role": 'user', "content": f"{context}\n\nUser's question: {prompt}"}
            ]
        )
        return result['message']['content']
    except Exception as e:
        return f"Error initializing Ollama model: {e}"

# Streamlit user interface
st.title("Chatbot with Ollama and Reinforcement Learning")

# Initialize session state if not already present
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'state' not in st.session_state:
    st.session_state.state = 0  # Initial state

# PDF upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.session_state.context = pdf_text
    st.write("PDF content extracted successfully.")

# Input for the prompt
prompt = st.chat_input("Ask Anything")

if prompt:
    st.session_state.messages.append({"role": 'user', "content": prompt})

    # Processing
    with st.spinner("Thinking..."):
        context = st.session_state.context if 'context' in st.session_state else ""
        response = get_chatbot_response(prompt, context)
        st.session_state.messages.append({"role": 'assistant', "content": response})
        
        # Display thumbs-up/thumbs-down feedback options
        feedback_col1, feedback_col2 = st.columns(2)
        with feedback_col1:
            thumbs_up = st.button("Thumbs Up :)")
        with feedback_col2:
            thumbs_down = st.button("Thumbs Down :( ")
        
        # Process feedback
        if thumbs_up:
            user_feedback = 1  # Positive feedback
        elif thumbs_down:
            user_feedback = -1  # Negative feedback
        else:
            user_feedback = 0  # Neutral feedback
        
        # Choose an action based on the current state
        action = agent.choose_action(st.session_state.state)
        
        # Define next state based on context or conversation history
        next_state = (st.session_state.state + 1) % state_size  # Simple state transition logic
        
        # Update Q-table with feedback
        if user_feedback != 0:
            agent.update_q_table(st.session_state.state, action, user_feedback, next_state)
        
        # Update state in session
        st.session_state.state = next_state

# Display chat history
for message in st.session_state.messages:
    if message['role'] == 'user':
        with st.chat_message('user'):
            st.write(message['content'])
    else:
        with st.chat_message('assistant'):
            st.write(message['content'])

    