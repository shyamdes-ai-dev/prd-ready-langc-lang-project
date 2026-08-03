# pyrefly: ignore [missing-import]
from langchain_core import output_parsers
import importlib.metadata

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def demo_basic_chain():

    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Anser in one sentence: {question}"
    )
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.7)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    response = chain.invoke({"question": "Why is the sky blue?"})
    print(response)


def demo_batch_execution():
    prompt = ChatPromptTemplate.from_template(
        "You are a good language translator. Translate the following text to {target_language}: {text}"
    )
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.7)
    output_parsers = StrOutputParser()
    chain = prompt | model | output_parsers
    inputs = [
        {"target_language": "french", "text": "Hello Shyam"},
        {"target_language": "spanish", "text": "Hello Shyam"},
        {"target_language": "gujarati", "text": "Hello Shyam"},
    ]
    response = chain.batch(inputs)
    print(response)


def demo_streaming():
    prompt = ChatPromptTemplate.from_template("{question}")
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.7)
    output_parsers = StrOutputParser()
    chain = prompt | model | output_parsers

    for chunk in chain.stream(
        {"question": "What is happened in Sansad Bhavan in India, in brief?"}
    ):
        print(chunk, end="", flush=True)
    print("\n")


def excercise_first_chain():
    prompt = ChatPromptTemplate.from_template(
        "Generate a market tagline for the product {product} for the target audience {audience}"
    )
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.7)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser

    for chunk in chain.stream(
        {"product": "Shampoo", "audience": "18-25 year old males"}
    ):
        print(chunk, end="", flush=True)
    print("\n")


def universal_way_of_initialization_of_model():
    model = init_chat_model(
        "gemini-3.5-flash-lite", temperature=0.7, model_provider="google_genai"
    )
    response = model.invoke("why the sky is blue in one line?")
    print(response.content[0].get("text"))


def main():
    # demo_basic_chain()
    # demo_batch_execution()
    # demo_streaming()
    # excercise_first_chain()
    universal_way_of_initialization_of_model()


if __name__ == "__main__":
    main()
