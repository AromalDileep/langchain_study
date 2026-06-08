'''
StructuredOutputParser helps extract structured JSON data from LLM responses 
using ResponseSchema objects. This is the "Classic" way to handle structured output.
'''
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

schema = [
    ResponseSchema(name="fact_1", description="First fact about the topic"),
    ResponseSchema(name="fact_2", description="Second fact about the topic"),
    ResponseSchema(name="fact_3", description="Third fact about the topic"),
    ResponseSchema(name="fact_4", description="Fourth fact about the topic"),
    ResponseSchema(name="fact_5", description="Fifth fact about the topic")
]

#  Create the Parser
parser = StructuredOutputParser.from_response_schemas(schema)

# 4. Setup the Template
template = PromptTemplate(
    template="Give 5 facts about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
chain = template | model | parser

result = chain.invoke({"topic": "black hole"})
print("--- PARSED DICTIONARY ---")
print(result)
print(f"\nType: {type(result)}")
print(f"Fact 1: {result['fact_1']}")
