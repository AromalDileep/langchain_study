from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="models/gemma-3-1b-it"   
)

response = model.invoke("What is the capital of India?")
print("------------------")
print(response.content)