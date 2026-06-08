from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(
    llm=llm
)

result = model.invoke("what is the capital of India")
print("--------------")
print(result.content)

