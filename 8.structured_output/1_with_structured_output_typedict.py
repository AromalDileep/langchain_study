from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")

# schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key, themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    # sentiment: Annotated[str, "Return sentiment of the review either positive, negative or neutral"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either positive, negative or neutral"] # Just to understand literal
    pros: Annotated[Optional[list[str]], "Write down all the pros inside the list"]
    cons : Annotated[Optional[list[str]], "Write down all the cons inside the list"]
    name: Annotated[Optional[str], "Write the name of the reviewer who wrote the review not the product"]

structured_model= model.with_structured_output(Review)

# result =structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many 
#              pre-installed apps that I can't remove. Also, the UI looks outdated compared to other
#              brands. Hoping for a software update to fix this.
#              """)

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
print(result["sentiment"])
print(result.keys())