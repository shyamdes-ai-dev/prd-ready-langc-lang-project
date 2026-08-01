from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage 
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


def init_model(model_name, temperature=0, model_provider="google_genai"):
    return init_chat_model(model=model_name,temperature=temperature,model_provider=model_provider)

model = init_model("gemini-3.5-flash-lite", temperature=0.7)

#StrOutputParser

parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template("Wire a short poem about {topic}")

chain = prompt | model | parser

print(chain.invoke({"topic": "Cats"}))

#JsonOutputParser

parser = JsonOutputParser()
prompt = ChatPromptTemplate.from_template("Wire a JSON about {topic}")
chain = prompt | model | parser

print(chain.invoke({"topic": "Cats"}))
print("\n\n")
#PydanticOutputParser

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    gender: str = Field(description="Gender of the person")
    
parser = PydanticOutputParser(pydantic_object=Person)
prompt = ChatPromptTemplate.from_template("Wire a Person about {topic}. {format_instructions}")
chain = prompt | model | parser

print(chain.invoke({"topic": "Shyam", "format_instructions": parser.get_format_instructions()}))
print("\n\n")

#Structured Output
class MovieReview(BaseModel):
    title:str = Field(description="Title of the movie")
    review:str = Field(description="Review of the movie")
    rating:int = Field(description="Rating of the movie (1-5)")

structured_output_model = model.with_structured_output(MovieReview)
prompt = ChatPromptTemplate.from_template("Review the movie {movie_title}")

chain = prompt | structured_output_model


print(chain.invoke({"movie_title": "The Matrix"}))