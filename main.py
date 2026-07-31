# pyrefly: ignore [missing-import]
import importlib.metadata
from dotenv import load_dotenv

from langchain_core import __version__ as version_core
from langchain_google_genai import ChatGoogleGenerativeAI

version_graph = importlib.metadata.version("langgraph")

print(f"Langchain Core Version : {version_core}")
print(f"Langgraph Version : {version_graph}")

load_dotenv()


def main():
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite",temperature=0)
    response = llm.invoke("Say 'setup complete' in one word ")
    print("response from google gemini", response.content)
    print(response.content[0]['text'])


if __name__ == "__main__":
    main()
