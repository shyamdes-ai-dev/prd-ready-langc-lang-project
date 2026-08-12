"""
ChromaDB Vector Store, Filtering & RAG Pipelines
================================================
Demonstrates production patterns for vector databases using ChromaDB and LangChain:
1. In-Memory & Persistent Vector Store creation.
2. Top-K Similarity Search with Cosine Distance scores.
3. Metadata Filtering for targeted retrieval.
4. Disk Persistence and Reloading verification.
5. Advanced Retrieval: Standard Top-K vs Maximal Marginal Relevance (MMR).
6. End-to-End Retrieval-Augmented Generation (RAG) using LCEL.
"""

import tempfile
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load API credentials
load_dotenv()

# Initialize Chat Model and Embeddings
model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Sample corpus documents with metadata
sample_document = [
    Document(
        page_content="LangChain is a framework for developing applications powered by AI.",
        metadata={"source": "LangChain Documentation", "topic": "LangChain"},
    ),
    Document(
        page_content="LangChain is a development framework for building applications with large language models (LLMs). It allows developers to chain together multiple LLMs and other tools to create complex applications.",
        metadata={"source": "LangChain Documentation", "topic": "LangChain"},
    ),
    Document(
        page_content="LangChain has two main abstractions: Chains and Agents. Chains are sequences of calls to LLMs or other tools, while Agents use LLMs to decide which tools to call and in what order.",
        metadata={"source": "LangChain Documentation", "topic": "LangChain"},
    ),
    Document(
        page_content="LangChain can be used to build a variety of applications, including chatbots, question answering systems, and more.",
        metadata={"source": "LangChain Documentation", "topic": "LangChain"},
    ),
    Document(
        page_content="LangChain is a powerful tool that can be used to build a variety of applications. It is a free and open source framework that is available for use by developers all over the world.",
        metadata={"source": "LangChain Documentation", "topic": "LangChain"},
    ),
    Document(
        page_content="LangChain has integrations with various tools and services, including databases, APIs, and more.",
        metadata={"source": "LangChain Documentation", "topic": "LangChain"},
    ),
    Document(
        page_content="LangChain is a constantly evolving field with new tools, techniques, and applications being developed all the time by Shyam.",
        metadata={"source": "LangChain Documentation", "topic": "Shyam"},
    ),
]


def chroma_basic_operations():
    """
    Demonstrates creating a Chroma vector store collection from documents
    and performing standard similarity search.
    """
    print("=== 1. Basic Chroma Similarity Search ===")
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = Chroma.from_documents(
            documents=sample_document, embedding=embedding, persist_directory=temp_dir
        )
        print(f"Vector store created with {vector_store._collection.count()} document(s).")

        query = "What is langchain"
        result = vector_store.similarity_search(query, k=3)

        print(f"Top 3 results for query '{query}':")
        for i, doc in enumerate(result, start=1):
            print(f"Result {i}: {doc.page_content}\n Metadata: {doc.metadata}\n")


def chroma_similarity_search_with_scores():
    """
    Demonstrates similarity search returning distance scores alongside documents.
    Note: For cosine distance, lower scores indicate higher semantic similarity.
    """
    print("=== 2. Chroma Similarity Search with Scores ===")
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = Chroma.from_documents(
            documents=sample_document, embedding=embedding, persist_directory=temp_dir
        )
        query = "What is langchain"
        result = vector_store.similarity_search_with_score(query, k=3)

        print(f"Top 3 results with distance scores for query '{query}':")
        for i, (doc, score) in enumerate(result, start=1):
            print(f"Result {i} (Distance Score: {score:.4f}):\n  {doc.page_content}\n  Metadata: {doc.metadata}\n")


def metadata_filtering():
    """
    Demonstrates filtering vector search results by structured metadata keys
    (e.g., retrieving only documents where topic == 'Shyam').
    """
    print("=== 3. Vector Search with Metadata Filtering ===")
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = Chroma.from_documents(
            documents=sample_document, embedding=embedding, persist_directory=temp_dir
        )
        query = "What is LangGraph?"
        # Apply filter dictionary to restrict vector candidate pool
        result = vector_store.similarity_search_with_score(
            query=query, k=3, filter={"topic": "Shyam"}
        )

        print(f"Results filtered by topic='Shyam':")
        for i, (doc, score) in enumerate(result, start=1):
            print(f"Result {i} (Score: {score:.4f}):\n  {doc.page_content}\n  Metadata: {doc.metadata}\n")


def persist_chroma():
    """
    Demonstrates saving ChromaDB to disk and reloading it in another session
    without re-generating embeddings.
    """
    print("=== 4. Chroma Disk Persistence & Reloading ===")
    persist_dir = "./chroma_db/"

    # Create and persist vector store
    vector_store = Chroma.from_documents(
        documents=sample_document, embedding=embedding, persist_directory=persist_dir
    )
    original_count = vector_store._collection.count()
    print(f"Persisted vector store with {original_count} documents to '{persist_dir}'")

    # Delete in-memory reference to simulate process restart
    del vector_store

    # Reload vector store from persistent disk directory
    reloaded_vector_store = Chroma(
        persist_directory=persist_dir, embedding_function=embedding
    )
    reloaded_count = reloaded_vector_store._collection.count()
    print(f"Reloaded vector store has {reloaded_count} documents.")
    assert original_count == reloaded_count, "Document count mismatch on reload!"

    results = reloaded_vector_store.similarity_search_with_score("Langchain", k=2)
    print("Search on reloaded store successful!")
    for i, (doc, score) in enumerate(results, start=1):
        print(f"  Result {i} (Score: {score:.4f}): {doc.page_content[:60]}...")
    print()


def retriving_using_chain_and_vector_db():
    """
    Demonstrates building a full Retrieval-Augmented Generation (RAG) pipeline:
    1. Ingest Documents into ChromaDB.
    2. Build an MMR (Maximal Marginal Relevance) Retriever to ensure result diversity.
    3. Compose LCEL RAG Chain: Query -> Retrieve Context -> Format Prompt -> Gemini LLM -> Output Parser.
    """
    print("=== 5. End-to-End RAG Pipeline with MMR Retrieval ===")
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = Chroma.from_documents(
            documents=sample_document, embedding=embedding, persist_directory=temp_dir
        )

        # Standard similarity retriever
        similarity_retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 2}
        )

        user_query = "What is langchain?"
        context = similarity_retriever.invoke(user_query)
        print(f"Standard similarity retriever fetched {len(context)} doc(s).")

        # Maximal Marginal Relevance (MMR) retriever: Fetches 'fetch_k' items, selects top 'k' most diverse
        mmr_retriever = vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 2, "fetch_k": 5}
        )
        mmr_context = mmr_retriever.invoke(user_query)
        print(f"MMR diversity retriever fetched {len(mmr_context)} doc(s).")

        # Define RAG prompt template
        prompt = ChatPromptTemplate.from_template(
            "Answer the question based on the following context:\n\n{context}\n\n"
            "Question: {question}\nProvide a concise, point-by-point answer without markdown formatting."
        )

        # Compose full LCEL RAG Chain
        rag_chain = (
            {
                "context": (lambda x: x["question"]) | mmr_retriever,
                "question": lambda x: x["question"],
            }
            | prompt
            | model
            | StrOutputParser()
        )

        result = rag_chain.invoke({"question": user_query})
        print(f"\nRAG Answer for query '{user_query}':\n{result}")
        print("=" * 60)


def main():
    """Execute all ChromaDB operations."""
    chroma_basic_operations()
    chroma_similarity_search_with_scores()
    metadata_filtering()
    persist_chroma()
    retriving_using_chain_and_vector_db()


if __name__ == "__main__":
    main()
