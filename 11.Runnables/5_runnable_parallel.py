from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

# Prompt for tweet
tweet_prompt = PromptTemplate.from_template(
    "Generate a short tweet about {topic}"
)

# Prompt for LinkedIn
linkedin_prompt = PromptTemplate.from_template(
    "Generate a professional LinkedIn post about {topic}"
)
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)


# Output parser
parser = StrOutputParser()

# Parallel chains
parallel_chain = RunnableParallel({
    "tweet": tweet_prompt | model | parser,
    "linkedin": linkedin_prompt | model | parser
})

# Invoke
result = parallel_chain.invoke({"topic": "AI"})

# Output
print("TWEET:\n")
print(result["tweet"])

print("\n" + "="*50 + "\n")

print("LINKEDIN POST:\n")
print(result["linkedin"])