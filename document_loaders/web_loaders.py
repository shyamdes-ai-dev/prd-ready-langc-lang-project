import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup

load_dotenv()


def web_loader():
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/LangChain", bs_kwargs={"parse_only":None})
    doc = loader.load()

    print(f"Loaded {len(doc)} documents from the web")
    print(f"Source: {doc[0].metadata['source']}")
    print(f"Title: {doc[0].metadata['title']}")
    print(f"Content Length: {len(doc[0].page_content)} characters")
    print(f"Preview: {doc[0].page_content[:200]}...")

    

web_loader()