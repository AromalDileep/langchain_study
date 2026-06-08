from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.title("Groq Chatbot")

# 1. Initialize the model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=1.4,
)

# 2. Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are helpful ai assistant")
    ]

# 3. Display history (Required to keep bubbles on screen after rerun)
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# 4. Chat Input & Logic
if prompt := st.chat_input("What is up?"):
    # Add and show user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and show AI response
    with st.chat_message("assistant"):
        result = model.invoke(st.session_state.messages)
        st.markdown(result.content)
    
    # Save response to history
    st.session_state.messages.append(AIMessage(content=result.content))