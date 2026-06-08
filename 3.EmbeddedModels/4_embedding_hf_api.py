import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()


embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-large-en-v1.5",
)

text = "Delhi is the capital of India"

vector = embeddings.embed_query(text)
print("--- Success via Modern API ---")
print(f"Vector Length: {len(vector)}")
print(f"First 3 values: {vector[:3]}")