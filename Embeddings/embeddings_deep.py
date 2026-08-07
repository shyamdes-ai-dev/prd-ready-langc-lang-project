from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()


def embeddings_deep():
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=os.environ["GEMINI_API_KEY"], model="gemini-embedding-001"
    )
    text = "This is the text to embed"
    document = ["This is the first document.", "This is the second document."]
    vec1 = embeddings.embed_query(text=text)
    vec2 = embeddings.embed_query(text=document[0])
    print("embed query: ", vec1)
    print("embed documents: ", embeddings.embed_documents(document))
    print(f"Vector Norm: {np.linalg.norm(vec1)}")
    print(
        f"Similarity: {np.dot(vec1,vec2)} / ({np.linalg.norm(vec1)*np.linalg.norm(vec2)})"
    )


embeddings_deep()
