"""
Dense Embeddings & Vector Mathematics
=====================================
Demonstrates generating dense vector embeddings using Google Generative AI (`gemini-embedding-001`),
differentiating between query embeddings and document embeddings, and calculating vector norms
and Cosine Similarity mathematically.

Mathematical Concepts:
- Euclidean (L2) Norm: ||v|| = sqrt(sum(v_i^2))
- Cosine Similarity   : cos(theta) = (u . v) / (||u|| * ||v||)
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np

# Load API credentials
load_dotenv()


def embeddings_deep():
    """
    Demonstrates vector embedding generation and cosine similarity calculation.
    """
    # Step 1: Initialize Google Gemini Embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        model="gemini-embedding-001",
    )

    query_text = "This is the text to embed"
    documents = [
        "This is the first document.",
        "This is the second document.",
    ]

    print("Generating dense vector embeddings...")

    # Step 2: Generate embedding for a single search query string
    vec1 = np.array(embeddings.embed_query(text=query_text))

    # Step 3: Generate embeddings for a list of document strings
    doc_vectors = embeddings.embed_documents(documents)
    vec2 = np.array(doc_vectors[0])

    print(f"Vector Dimensions : {len(vec1)}")
    print(f"Query Vector (first 5 values): {vec1[:5]}")
    print(f"Total Documents Embedded     : {len(doc_vectors)}")

    # Step 4: Calculate L2 Vector Norm (Magnitude)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    print(f"\nVector 1 L2 Norm (Magnitude): {norm_vec1:.4f}")
    print(f"Vector 2 L2 Norm (Magnitude): {norm_vec2:.4f}")

    # Step 5: Compute Dot Product and Cosine Similarity
    dot_product = np.dot(vec1, vec2)
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    print(f"Dot Product                : {dot_product:.4f}")
    print(f"Cosine Similarity (0 to 1) : {cosine_similarity:.4f}")


if __name__ == "__main__":
    embeddings_deep()
