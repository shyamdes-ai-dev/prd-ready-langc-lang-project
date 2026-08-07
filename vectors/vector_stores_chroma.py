from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(model_provider="google_genai", model="gemini-3.5-flash-lite")
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

sample_document = [
    Document(
        page_content="Langchain is a framework for developing applications powered by AI",
        metadata={"source": "Langchain Documentation", "topic": "Langchain"},
    ),
    Document(
        page_content="LangChain is a development framework for building applications with large language models (LLMs). It allows developers to chain together multiple LLMs and other tools to create complex applications.",
        metadata={"source": "Langchain Documentation", "topic": "Langchain"},
    ),
    Document(
        page_content="LangChain has two main abstractions: Chains and Agents. Chains are sequences of calls to LLMs or other tools, while Agents use LLMs to decide which tools to call and in what order.",
        metadata={"source": "Langchain Documentation", "topic": "Langchain"},
    ),
    Document(
        page_content="Langchain can be used to build a variety of applications, including chatbots, question answering systems, and more.",
        metadata={"source": "Langchain Documentation", "topic": "Langchain"},
    ),
    Document(
        page_content="Langchain is a powerful tool that can be used to build a variety of applications. It is a free and open source framework that is available for use by developers all over the world.",
        metadata={"source": "Langchain Documentation", "topic": "Langchain"},
    ),
    Document(
        page_content="Langchain has integrations with various tools and services, including databases, APIs, and more.",
        metadata={"source": "Langchain Documentation", "topic": "Langchain"},
    ),
    Document(
        page_content="Langchain is a constantly evolving field with new tools, techniques, and applications being developed all the time.",
        metadata={"source": "Langchain Documentation", "topic": "Shyam"},
    ),
]


def chroma_basic_operations():
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = Chroma.from_documents(
            documents=sample_document, embedding=embedding, persist_directory=temp_dir
        )
        print(
            f"Vecotr store created {vector_store._collection.count()} at the directory"
        )

        query = "What is langchain"
        result = vector_store.similarity_search(query, k=3)
        print(f"Top 3 results for query '{query}' ")
        for i, doc in enumerate(result, start=1):
            print(f"Result{i}: {doc.page_content} \n Metadata: {doc.metadata}\n")


def chroma_similarity_search_with_scores():
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = Chroma.from_documents(
            documents=sample_document, embedding=embedding, persist_directory=temp_dir
        )
        print(
            f"Vecotr store created {vector_store._collection.count()} at the directory {temp_dir}"
        )

        query = "What is langchain"
        result = vector_store.similarity_search_with_score(query, k=3)
        print(f"Top 3 results for query '{query}' ")
        for i, (doc, score) in enumerate(result, start=1):
            print(
                f"Result{i}: {doc.page_content} (score: {score}) \n Metadata: {doc.metadata}\n"
            )


def metadata_filtering():
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = Chroma.from_documents(
            documents=sample_document, embedding=embedding, persist_directory=temp_dir
        )
        query = "What is LangGraph?"
        print("Metadata filtering")
        result = vector_store.similarity_search_with_score(
            query=query, k=3, filter={"topic": "Shyam"}
        )
        print(f"Top 3 results for query '{query}' ")
        for i, (doc, score) in enumerate(result, start=1):
            print(
                f"Result{i}: {doc.page_content} (score: {score})\n Metadata: {doc.metadata}\n"
            )


def persist_chroma():
    persist_dir = "./chroma_db/"
    vector_store = Chroma.from_documents(
        documents=sample_document, embedding=embedding, persist_directory=persist_dir
    )
    print(
        f"Vecotr store created {vector_store._collection.count()} at the directory {persist_dir}"
    )

    original_count = vector_store._collection.count()
    print(f"Persisted vector store with {original_count} documents.")
    print(f"Vectore store persisted at: {persist_dir}")

    # Delete the vector store to simulate a restart or another process loading it
    del vector_store

    # Reload the vector store from the same directory
    reloaded_vector_store = Chroma(
        persist_directory=persist_dir, embedding_function=embedding
    )

    reloaded_count = reloaded_vector_store._collection.count()
    print(f"Reloaded vector store has {reloaded_count} documents.")
    assert original_count == reloaded_count

    results = reloaded_vector_store.similarity_search_with_score("Langchain", k=2)
    print(f"Results")
    for i, (doc, score) in enumerate(results, start=1):
        print(
            f"Result{i}: {doc.page_content} (score: {score}) \n Metadata: {doc.metadata}"
        )


def retriving_using_chain_and_vector_db():

    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = Chroma.from_documents(
            documents=sample_document, embedding=embedding, persist_directory=temp_dir
        )
        print(
            f"Vector store created with {vector_store._collection.count()} documents."
        )

        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 2}
        )

        user_query = "What is langchain?"
        context = retriever.invoke(user_query)
        print(f"Retrieved {len(context)} documents for query: {user_query}")

        mmr_retriever = vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 2, "fetch_k": 5}
        )

        mmr_context = mmr_retriever.invoke(user_query)
        print(f"Retrieved {len(mmr_context)} documents for query: {user_query}")

        # Now you can use this context with a retriever chain to answer questions
        # from the documents. For example:
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(
            """Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}. create point by point reply also not in the markdown language"""
        )

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
        print(f"\nAnswer: {result}")
        print("=" * 40)


retriving_using_chain_and_vector_db()
