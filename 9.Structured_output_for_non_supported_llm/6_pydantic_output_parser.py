from pydantic import BaseModel,Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
)

class Person(BaseModel):
    name : str = Field(description="name of the person")
    age: int = Field(gt=18, description="age of the person")
    city : str = Field(description="Name of the city the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of the fictional {place} person \n {format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)

chain = template | model | parser

result =chain.invoke({"place":"Indian"})
print(result)
print(type(result))
# prompt = template.invoke({"place":"india"})

# print(prompt)

# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)