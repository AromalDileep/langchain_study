from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=32
)

documents = [
    "Delhi is the Capital of India",
    "Kolkata is the Capital of West Bengal",
    "Paris is the Capital of France"
    "Trivandrum is the Capital of  Kerala"
]

result = embeddings.embed_documents(documents)
print("----------------")
print(str(result))