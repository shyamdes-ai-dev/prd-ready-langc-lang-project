"""
Document Loaders: TextLoader
============================
Demonstrates how to load raw plain-text (.txt) files into LangChain `Document` objects.
Each loaded `Document` encapsulates:
- `page_content`: The textual content of the file.
- `metadata`    : Contextual dictionary including source path, file info, etc.
"""

import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

# Load environment configuration
load_dotenv()


def load_text_file():
    """
    Creates a temporary text file, loads it using TextLoader, inspects the Document
    attributes, and cleans up the temporary file.
    """
    # Create a temporary text file for demonstration purposes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(
            b"Hello, this is a sample text file.\nThis file is used to demonstrate LangChain's TextLoader capabilities."
        )
        temp_file_path = temp_file.name

    try:
        # Instantiate TextLoader with the target file path
        loader = TextLoader(temp_file_path)

        # .load() reads the file synchronously and returns a list of Document objects
        documents = loader.load()

        print(f"Loaded {len(documents)} document(s) using TextLoader:")
        for i, doc in enumerate(documents, start=1):
            print(f"\n--- Document {i} ---")
            print("Metadata    :", doc.metadata)
            print("Page Content:\n", doc.page_content)

    finally:
        # Clean up temporary file to prevent disk littering
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"\nCleaned up temporary file: {temp_file_path}")


if __name__ == "__main__":
    load_text_file()
