from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

# schema
class Review(BaseModel):
    key_themes: Annotated[list[str], Field(description="Write down all the key, themes discussed in the review in a list")]
    summary: Annotated[str, Field(description="A brief summary of the review")]
    sentiment:Annotated[Literal["pos", "neg"], Field(description="Return sentiment of the review either positive, negative or neutral")]
    pros: Annotated[Optional[list[str]], Field(description="Write down all the pros inside the list")]
    cons : Annotated[Optional[list[str]], Field(description= "Write down all the cons inside the list")]
    name: Annotated[Optional[str],Field(description= "Write the name of the reviewer who wrote the review not the product")]

structured_model= model.with_structured_output(Review)

result = structured_model.invoke("""
The Quantum-X7 is a hardware marvel with a software soul in crisis. Under the hood, it's 
running the new **Snapdragon X Elite Gen 2**, which provides snappy responsiveness, but 
the thermal throttling is real. The **48-hour battery** is great for endurance, though the 
**5-hour charge time** via the proprietary port is prehistoric. 

The **14-inch OLED panel** hits **120Hz** beautifully, but again, the 'NeuralSync' 
drivers create lag in Windows. It’s a beastly machine on paper, but a headache in practice.

PROS:
- Exceptional build quality (Titanium chassis).
- Industry-leading haptic feedback.
- Massive 48-hour battery life.

CONS:
- Severe software compatibility issues.
- Frustratingly slow charging speeds.

Review by Aromal Dileep
""")
print(type(result))
print(result)
print("-----------")
print(result.model_dump().keys())
print(result.sentiment)
print(result.model_dump()["sentiment"])