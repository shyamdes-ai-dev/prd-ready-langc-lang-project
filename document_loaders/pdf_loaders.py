import os
from dotenv import load_dotenv
import tempfile
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()


def pdf_loader():
    # Create a temporary PDF file for demonstration
    pdf_content = b"%PDF-1.4\n1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n2 0 obj<< /Type /Pages /Kids[3 0 R] /Count 1 >>endobj\n3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>endobj\n4 0 obj<< /Length 49 >>stream\nBT\n100 700 Td\n(LangChain PDF Document) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n0000000182 00000 n \ntrailer<< /Size 5 /Root 1 0 R >>\nstartxref\n264\n%%EOF"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_content)
        temp_file_path = temp_file.name

    try:
        # load the PDF file using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # print the loaded documents
        for doc in documents:
            print(doc)
            print(doc.page_content)

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Cleaned up temporary file: {temp_file_path}")


pdf_loader()
