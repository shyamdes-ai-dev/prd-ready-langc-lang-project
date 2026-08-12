"""
Document Loaders: WebBaseLoader
===============================
Demonstrates how to scrape, clean, and ingest web pages into LangChain `Document` objects
using `WebBaseLoader` powered by `BeautifulSoup4` (bs4).
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

# Load environment configuration
load_dotenv()


def web_loader():
    """
    Fetches the Wikipedia page for LangChain, extracts the main HTML content,
    and displays the parsed metadata and text length.
    """
    target_url = "https://en.wikipedia.org/wiki/LangChain"
    print(f"Fetching web document from: {target_url} ...")

    # Initialize WebBaseLoader with target URL and optional BeautifulSoup parse arguments
    loader = WebBaseLoader(
        web_path=target_url,
        bs_kwargs={"parse_only": None},  # Optional: specify SoupStrainer to extract specific HTML tags/classes
    )

    # .load() executes HTTP GET request, extracts text, and wraps into Document
    docs = loader.load()

    print(f"Successfully loaded {len(docs)} document(s) from the web:")
    if docs:
        first_doc = docs[0]
        print(f"Source URL     : {first_doc.metadata.get('source')}")
        print(f"Page Title     : {first_doc.metadata.get('title')}")
        print(f"Content Length : {len(first_doc.page_content)} characters")
        print("\n--- Content Preview (First 250 chars) ---")
        print(first_doc.page_content[:250].strip(), "...")


if __name__ == "__main__":
    web_loader()
