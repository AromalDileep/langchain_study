from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate.from_template("Write a detailed report on {topic}")
template2 = PromptTemplate.from_template("Write a 5 line summary on the following text. \n {text}")
# We add StrOutputParser() to convert the AI Message object into a simple string
chain = template1 | model | StrOutputParser() | template2 | model | StrOutputParser()

"""
WHY THIS WORKS:
LangChain's pipe operator (|) has a 'Short-Circuit' logic for PromptTemplates. 
If a Template has exactly ONE input variable (like our template2 which only has {text}), 
and the previous step outputs a simple string, LangChain automatically assigns 
that string to the single variable. 

It essentially does: template2.invoke({"text": output_from_previous_step}) behind the scenes.
"""

#  Run it
final_result = chain.invoke({"topic": "black hole"})

print("--- FINAL SUMMARY ---")
print(final_result)
"""
THE ROBUST EXAMPLE (For future understanding):
If you ever add a second variable (like {style} or {tone}), the simple chain above will CRASH 
because LangChain won't know which variable to fill with the previous output. 

Use the 'Map' approach below for real-world projects:
# Robust mapping using a dictionary (The professional standard)

"""

'''
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation",
)

template1 = PromptTemplate.from_template("Write a detailed report on {topic}")
template2 = PromptTemplate.from_template("Write a 5 line summary on:\n{text}")

chain = (
    template1
    | llm
    | StrOutputParser()
    | RunnableLambda(lambda output: {"text": output})
    | template2
    | llm
    | StrOutputParser()
)

result = chain.invoke({"topic": "black hole"})
print(result)
'''

