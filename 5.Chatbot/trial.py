import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

st.title("Chatbot")

model = ChatGroq(
    model ="llama-3.1-8b-instant",
    temperature=1.4    
)

if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a helpful assistant")]

for conversation in st.session_state.messages:
    if isinstance(conversation, HumanMessage):
        with st.chat_message("user"):
            st.markdown(conversation.content)

    if isinstance(conversation, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(conversation.content)
    
if prompt := st.chat_input("What is up"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    
    result = model.invoke(st.session_state.messages)
    with st.chat_message("assistant"):
        st.markdown(result.content)
    
    st.session_state.messages.append(AIMessage(content=result.content))
