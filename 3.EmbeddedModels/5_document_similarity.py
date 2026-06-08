import numpy as np

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=300
)
document = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills",
    "Sachin Tendulkar is a legendary batsman often called the God of Cricket",
    "Jasprit Bumrah is a world class fast bowler famous for his deadly yorkers",
    "Rohit Sharma is a prolific opener known for his record-breaking double centuries"
]

query = "Tell me about virat kohli"

doc_embeddings = embedding.embed_documents(document)
query_embeddings = embedding.embed_query(query)

scores = cosine_similarity([query_embeddings], doc_embeddings)[0]# both should be 2d list query is not that is why [] around query
# [0] because we only want to see one dimension


index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(query)
print(document[index])
print("similarity score is ", score)