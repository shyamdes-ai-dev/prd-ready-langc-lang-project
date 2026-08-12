"""
Document Loaders: DirectoryLoader with Lazy Loading
===================================================
Demonstrates memory-efficient document ingestion using `DirectoryLoader` paired with
the `.lazy_load()` Python generator method.

Why Lazy Loading matters in Production:
- `.load()` reads ALL documents into RAM simultaneously, risking Out-Of-Memory (OOM)
  errors when processing large knowledge bases (gigabytes of PDFs, text, etc.).
- `.lazy_load()` yields `Document` objects one by one as a generator, keeping memory
  consumption flat and minimal regardless of corpus size.
"""

import tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load environment configuration
load_dotenv()


def lazy_loader():
    """
    Creates a temporary directory containing multiple files and demonstrates streaming
    them iteratively using `lazy_load()`.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 5 sample text documents on disk
        for i in range(5):
            path = Path(tmpdir) / f"document_{i}.txt"
            path.write_text(f"This is document {i}. It contains sample enterprise knowledge content.")

        # Initialize DirectoryLoader targeting the folder with a glob pattern and specific file loader
        loader = DirectoryLoader(tmpdir, loader_cls=TextLoader, glob="*.txt")

        print(f"Initialized DirectoryLoader for folder: {tmpdir}")
        print("Iterating over documents lazily (generator mode):\n")

        # .lazy_load() yields one Document at a time without loading the whole directory into RAM
        for doc_index, doc in enumerate(loader.lazy_load(), start=1):
            print(f"[{doc_index}] Source : {doc.metadata['source']}")
            print(f"    Preview: {doc.page_content[:80]}...")
            print()


if __name__ == "__main__":
    lazy_loader()
