# pyrefly: ignore [missing-import]
from langchain_core import output_parsers
import importlib.metadata
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

def demo_basic_chain():
    
    prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Anser in one sentence: {question}")
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.7)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    response = chain.invoke({"question": "Why is the sky blue?"})
    print(response)

def demo_batch_execution():
    prompt = ChatPromptTemplate.from_template("You are a good language translator. Translate the following text to {target_language}: {text}")
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
    

def main():
    demo_basic_chain()
    demo_batch_execution()

if __name__ == "__main__":
    main()
