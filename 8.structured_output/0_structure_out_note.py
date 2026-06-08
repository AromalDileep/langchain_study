"""
In Langchain structured output refers to the practice of having language models return responses in a well-defined
data format (for example JSON), rather than free-form text. This makes the model output easier and to parse and work
with programmatically.
    
    example:
    - Data Extraction
    - API building
    - Agents
    
    
    
There are two types of LLM's 
    1. can output structured data
    2. can't output structured data

"with_structured_output" function is used for the llm's with 1st kind of llm
after invoke() we call with_structured_output and specify the data format

we can use:
    Typed Dict
    Pydantic
    json schema
    """


