"""
Main Entry Point
================
This repository serves as a production-grade reference architecture for building
stateful, observable, and multi-agent GenAI applications using LangChain and LangGraph.

Explore the subdirectories to learn:
- `langchain_fundamentals/` : LCEL chaining, prompt engineering, structured outputs.
- `chains/`                 : Parallel execution, passthroughs, conditional branching.
- `document_loaders/`       : Ingestion from PDFs, web pages, and directories.
- `text_splitters/`         : Recursive, Markdown header, and code splitters.
- `Embeddings/`             : Vector embeddings and similarity computations.
- `vectors/`                : ChromaDB vector store, filtering, MMR retrieval, and RAG.
- `smart_QA_bot/`           : Enterprise Q&A microservice with LangSmith tracing.
- `langgraph/`              : Cyclic multi-agent workflows, checkpointing, and HITL.
"""


def main():
    """Prints a welcome message indicating the repository environment is ready."""
    print("Welcome to LangChain & LangGraph Production Reference Architecture!")
    print("Explore the respective subdirectories to run specific modules and examples.")


if __name__ == "__main__":
    main()
