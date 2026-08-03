import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from bs4 import BeautifulSoup

def lazy_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(5):
            path = Path(tmpdir) / f"document_{i}.txt"
            path.write_text(f"This is docuemnt {i}. It contains sample content")

        loader = DirectoryLoader(tmpdir, loader_cls=TextLoader, glob="*.txt")

        print("Initialized lazy loader for directory:", tmpdir)

        for doc in loader.lazy_load():
            print("Document Content Preview:", doc.page_content[:100], "....")
            print("Metadata:", doc.metadata["source"])
            print("")

lazy_loader()