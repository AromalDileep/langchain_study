# proper structured
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# Groq chat model
model= ChatGroq(model="llama-3.3-70b-versatile")

# Prompts
template1 = PromptTemplate.from_template(
    "Write a detailed report on {topic}"
)

template2 = PromptTemplate.from_template(
    "Write a 5 line summary on:\n{text}"
)
str_parser = StrOutputParser()

# Chain
chain = (
    template1
    | model 
    | str_parser
    |RunnableLambda(lambda output: {"text": output})  # map LLM output → {"text": ...} for next prompt
    | template2
    | model
    | str_parser
)

# Run
result = chain.invoke({"topic": "black hole"})
print(result)