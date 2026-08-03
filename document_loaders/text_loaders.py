import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import TextLoader

from dotenv import load_dotenv

load_dotenv()


def load_text_file():
    # create a temporary text file for demonstration
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Hello, this is a smaple text file.\nThsi file is used to demonstrate the load text file")
        temp_file_path = temp_file.name

    try:    
        # load the text file using TextLoader
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        #print the loaded documents
        for doc in documents:
            print(doc)
            print(doc.page_content)
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
load_text_file()