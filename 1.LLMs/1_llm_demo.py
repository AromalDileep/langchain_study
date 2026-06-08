from langchain_groq import ChatGroq # we only have chatmodel for groq 
from langchain_openai import OpenAI # no llm like openai
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0 # reduces randomness → more deterministic and consistent outputs
)

response = model.invoke("What is the capital of India?")
print("--------------")
print(response.content)

