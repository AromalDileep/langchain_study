'''
study conditional chains
Feedback of customer -> Analyse and give feedback if positive give positive response if negative give negative response

if positive sent a feedback form if negative sent an email to customer executive.(with agents). We are not doing it because
we didn't complete agents. but the workflow is same, based on condition the model will respond'''
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile"
)

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give the sentiment of the feedback")

str_parser = StrOutputParser()
pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback into positive or negative. \n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="write an appropriate response to this positive response. \n {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="write an appropriate response to this negative response. \n {feedback}",
    input_variables=["feedback"]
)


classifier_chain = prompt1 | model | pydantic_parser

branch_chain = RunnableBranch(                                              # for condition chain
    (lambda x: x.sentiment=="positive", prompt2 | model | str_parser),   # if  (condition, chain)
    (lambda x: x.sentiment=="negative", prompt3 | model | str_parser),   # else
    RunnableLambda(lambda x: "Could not find sentiment")                    # default must provide (since not a chain we make it runnable lambda)
)

chain = classifier_chain | branch_chain 

result = chain.invoke({"feedback":"This is a terrible phone"})
print(result)
