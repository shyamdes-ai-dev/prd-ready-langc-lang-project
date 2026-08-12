"""
Output Parsers & Structured Outputs Demo
========================================
Demonstrates 4 methods for parsing and validating LLM outputs:
1. `StrOutputParser`     : Extracts raw plain-text string from AIMessage.
2. `JsonOutputParser`    : Parses output directly into a Python dictionary.
3. `PydanticOutputParser`: Validates and parses output into a strongly-typed Pydantic model
                           by injecting format instructions into the prompt.
4. `.with_structured_output(Schema)`: Uses native model tool-calling/schema enforcement
                                      to guarantee strongly-typed structured returns.
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Load API credentials from .env
load_dotenv()


def init_model(model_name: str, temperature: float = 0, model_provider: str = "google_genai"):
    """
    Helper function to initialize a chat model.
    """
    return init_chat_model(
        model=model_name, temperature=temperature, model_provider=model_provider
    )


model = init_model("gemini-3.5-flash-lite", temperature=0.7)

# -----------------------------------------------------------------------------
# 1. StrOutputParser (Plain String Extraction)
# -----------------------------------------------------------------------------
print("--- 1. StrOutputParser ---")
str_parser = StrOutputParser()
str_prompt = ChatPromptTemplate.from_template("Write a short poem about {topic}")
str_chain = str_prompt | model | str_parser

print(str_chain.invoke({"topic": "Cats"}))
print("\n" + "=" * 60 + "\n")

# -----------------------------------------------------------------------------
# 2. JsonOutputParser (Raw JSON to Python Dict)
# -----------------------------------------------------------------------------
print("--- 2. JsonOutputParser ---")
json_parser = JsonOutputParser()
json_prompt = ChatPromptTemplate.from_template("Write a JSON object with attributes of {topic}")
json_chain = json_prompt | model | json_parser

json_result = json_chain.invoke({"topic": "Cats"})
print("JSON Output:", json_result)
print("Type:", type(json_result))
print("\n" + "=" * 60 + "\n")

# -----------------------------------------------------------------------------
# 3. PydanticOutputParser (Schema Injected via Prompt Instructions)
# -----------------------------------------------------------------------------
print("--- 3. PydanticOutputParser ---")


class Person(BaseModel):
    """Schema representing a person entity."""
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    gender: str = Field(description="Gender of the person")


pydantic_parser = PydanticOutputParser(pydantic_object=Person)

# Format instructions are injected into the prompt so the LLM understands the expected JSON schema
pydantic_prompt = ChatPromptTemplate.from_template(
    "Write a Person profile about {topic}.\n{format_instructions}"
)
pydantic_chain = pydantic_prompt | model | pydantic_parser

pydantic_result = pydantic_chain.invoke(
    {"topic": "Shyam", "format_instructions": pydantic_parser.get_format_instructions()}
)
print("Pydantic Object:", pydantic_result)
print(f"Name: {pydantic_result.name}, Age: {pydantic_result.age}, Gender: {pydantic_result.gender}")
print("\n" + "=" * 60 + "\n")

# -----------------------------------------------------------------------------
# 4. Native Structured Output (.with_structured_output)
# -----------------------------------------------------------------------------
print("--- 4. Native Structured Output with Pydantic Schema ---")


class MovieReview(BaseModel):
    """Schema representing a movie review critique."""
    title: str = Field(description="Title of the movie")
    review: str = Field(description="Review of the movie")
    rating: int = Field(description="Rating of the movie (1-5)")


# .with_structured_output() uses function/tool calling protocols to ensure 100% compliant schemas
structured_output_model = model.with_structured_output(MovieReview)
review_prompt = ChatPromptTemplate.from_template("Review the movie {movie_title}")

structured_chain = review_prompt | structured_output_model

review_result: MovieReview = structured_chain.invoke({"movie_title": "The Matrix"})
print("Structured Output Result:", review_result)
print(f"Movie: {review_result.title}")
print(f"Rating: {review_result.rating}/5")
print(f"Review: {review_result.review}")
