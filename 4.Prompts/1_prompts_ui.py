from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.load import load
from dotenv import load_dotenv
import streamlit as st
import json

load_dotenv()

st.header("Research Tool")

model = ChatGoogleGenerativeAI(
    model="models/gemma-3-1b-it"
)

paper_input = st.selectbox("Select Research Paper Name",["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

with open("template.json","r") as f:
    template_data = json.load(f)

template = load(template_data)

# prompt = template.invoke({
#     'paper_input':paper_input,
#     'style_input':style_input,
#     'length_input':length_input
# })


if st.button("Summarize"):
    chain = template | model
    result = chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
    })
    st.write(result.content)

