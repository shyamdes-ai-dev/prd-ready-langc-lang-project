"""
Document Loaders: PyPDFLoader
=============================
Demonstrates how to parse and extract text from PDF documents using `PyPDFLoader` (via `pypdf`).
Each page in the PDF file is loaded as an individual `Document` object with metadata
indicating the source path and page number.
"""

import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

# Load environment configuration
load_dotenv()


def pdf_loader():
    """
    Creates a minimal temporary PDF binary, loads and parses it with PyPDFLoader,
    inspects the extracted page content and metadata, and performs file cleanup.
    """
    # Raw minimal PDF-1.4 binary content for testing
    pdf_content = (
        b"%PDF-1.4\n1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
        b"2 0 obj<< /Type /Pages /Kids[3 0 R] /Count 1 >>endobj\n"
        b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>endobj\n"
        b"4 0 obj<< /Length 49 >>stream\nBT\n100 700 Td\n(LangChain PDF Document) Tj\nET\nendstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n"
        b"0000000182 00000 n \ntrailer<< /Size 5 /Root 1 0 R >>\nstartxref\n264\n%%EOF"
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_content)
        temp_file_path = temp_file.name

    try:
        # Initialize PyPDFLoader pointing to the target PDF file
        loader = PyPDFLoader(temp_file_path)

        # .load() returns a list of Document objects (one per PDF page)
        documents = loader.load()

        print(f"Successfully loaded {len(documents)} page(s) from PDF:")
        for i, doc in enumerate(documents, start=1):
            print(f"\n--- Page {i} ---")
            print("Metadata    :", doc.metadata)
            print("Page Content:", doc.page_content.strip())

    finally:
        # Clean up the temporary PDF file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"\nCleaned up temporary file: {temp_file_path}")


if __name__ == "__main__":
    pdf_loader()
