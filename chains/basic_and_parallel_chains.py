"""
Advanced LCEL Chains & Execution Topologies
===========================================
Demonstrates advanced LangChain Expression Language (LCEL) constructs:
1. Sequential Chains (`prompt | model | parser`)
2. Concurrent Execution with `RunnableParallel`
3. Context Forwarding with `RunnablePassthrough` & `RunnableLambda`
4. Dynamic Conditional Routing with `RunnableBranch`
5. Chain Introspection, Debugging, and Step-Level Logging
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

# Load environment configuration
load_dotenv()

# Initialize default chat model for chains
model = init_chat_model(model="gemini-3.5-flash", model_provider="google_genai")


def basic_chain_illustration():
    """
    Demonstrates a simple sequential LCEL chain that summarizes text.
    Data flow: Input Dict -> Prompt Template -> Model -> Output Parser -> Clean String
    """
    print("=== 1. Basic Sequential Chain ===")
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in one sentence: {text}"
    )
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser

    sample_text = (
        "Scripts and CLI programs: Yes, it's a good practice because the process may exit "
        "immediately after the last LLM call. FastAPI, Django, or long-running servers: Usually "
        "not necessary after every request, because the process stays alive and the background "
        "uploader has time to send traces. You might only call it during application shutdown if needed."
    )

    response = chain.invoke({"text": sample_text})
    print("Summary:", response)
    print()


def parallel_chain_illustration():
    """
    Demonstrates `RunnableParallel` to execute multiple independent chains concurrently.
    In this example, one branch generates a summary while another branch extracts keywords.
    """
    print("=== 2. Parallel Chain Execution (RunnableParallel) ===")
    prompt1 = ChatPromptTemplate.from_template(
        "Summarize the following text in one sentence: {text}"
    )
    prompt2 = ChatPromptTemplate.from_template(
        "Extract keywords from the following text: {text1}"
    )
    output_parser = StrOutputParser()

    # Define parallel branches as keyword arguments
    analysis_chain = RunnableParallel(
        summary=prompt1 | model | output_parser,
        keywords=prompt2 | model | output_parser,
    )

    response = analysis_chain.invoke(
        {
            "text": "Scripts and CLI programs: Yes, it's a good practice because the process may exit immediately after the last LLM call.",
            "text1": "LangSmith uses a background uploader to send traces to the server. Flushing ensures all queued traces are sent before exit.",
        }
    )
    print("Summary Branch Output :", response["summary"])
    print("Keywords Branch Output:", response["keywords"])
    print()


def passthrough_chain_illustration():
    """
    Demonstrates `RunnablePassthrough` and `RunnableLambda`.
    - `RunnablePassthrough`: Passes the input dictionary through unchanged.
    - `RunnableLambda`: Wraps arbitrary Python functions into LCEL-compatible Runnables.
    This pattern is standard for building RAG pipelines where retrieved context and the original
    user question must both be fed into the prompt.
    """
    print("=== 3. Passthrough & Custom Lambda Chain ===")
    prompt = ChatPromptTemplate.from_template(
        "Original question: {question}\nContext: {context}\n\nAnswer the question based on the context."
    )

    # Simulated retriever function returning external context
    def fake_retriever(input_dict):
        return "LangChain was created by Harrison Chase in 2022."

    chain = (
        RunnableParallel(
            context=RunnableLambda(fake_retriever),
            question=RunnablePassthrough(),
        )
        | RunnableLambda(
            lambda x: {"context": x["context"], "question": x["question"]["question"]}
        )
        | prompt
        | model
        | StrOutputParser()
    )

    response = chain.invoke({"question": "Who created LangChain?"})
    print("RAG Response:", response)
    print()


def chain_branching():
    """
    Demonstrates dynamic conditional branching using `RunnableBranch`.
    `RunnableBranch` evaluates a list of `(condition_callable, runnable)` pairs.
    The first condition that evaluates to True executes its corresponding branch;
    otherwise, the default fallback branch is executed.
    """
    print("=== 4. Conditional Branching (RunnableBranch) ===")

    code_prompt = ChatPromptTemplate.from_template(
        "You are a coding expert. Help with: {input}"
    )
    general_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer: {input}"
    )

    # Classifier chain to determine intent
    classifier_prompt = ChatPromptTemplate.from_template(
        "Classify this as 'code' or 'general': {input}\nReturn only the classification word."
    )
    classifier_chain = classifier_prompt | model | StrOutputParser()

    def is_code_question(input_dict: dict) -> bool:
        """Predicate function returning True if question is code-related."""
        classification = classifier_chain.invoke(input_dict)
        return "code" in classification.lower()

    # Define branch structure: [(condition, branch_runnable), default_runnable]
    branch = RunnableBranch(
        (is_code_question, code_prompt | model | StrOutputParser()),
        (general_prompt | model | StrOutputParser()),
    )

    questions = [
        "How do I write a for loop in Python?",
        "What's the weather like today?",
    ]

    for q in questions:
        result = branch.invoke({"input": q})
        print(f"Question: {q}")
        print(f"Answer  : {result}")
        print("-" * 60)
    print()


def demo_debbuging():
    """
    Demonstrates methods to inspect, debug, and monitor LCEL chains:
    1. Inspecting Pydantic JSON Schemas for input and output.
    2. Attaching execution run metadata using `.with_config()`.
    3. Injecting intermediate logger steps using `RunnableLambda`.
    """
    print("=== 5. Chain Debugging & Intermediate Step Logging ===")
    prompt = ChatPromptTemplate.from_template("Say hello to {name}")
    chain = prompt | model | StrOutputParser()

    # Method 1: Inspect input and output schemas
    print("Chain input schema :\n", chain.input_schema.model_json_schema())
    print("\nChain output schema:\n", chain.output_schema.model_json_schema())

    # Method 2: Configure run tags for telemetry and tracing
    result = chain.with_config(run_name="greeting_chain").invoke({"name": "Alice"})
    print(f"\nGreeting: {result}\n")

    # Method 3: Inspect intermediate steps using custom logger lambdas
    def log_step(x, step_name: str = ""):
        print(f"[{step_name}] Type: {type(x).__name__} | Preview: {str(x)[:80]}...")
        return x

    debug_chain = (
        prompt
        | RunnableLambda(lambda x: log_step(x, "Step 1: After prompt formatting"))
        | model
        | RunnableLambda(lambda x: log_step(x, "Step 2: After model generation"))
        | StrOutputParser()
        | RunnableLambda(lambda x: log_step(x, "Step 3: After output parsing"))
    )

    debug_result = debug_chain.invoke({"name": "Bob"})
    print("Debug chain final result:", debug_result)


def main():
    """Execute all chain demonstrations."""
    basic_chain_illustration()
    parallel_chain_illustration()
    passthrough_chain_illustration()
    chain_branching()
    demo_debbuging()


if __name__ == "__main__":
    main()
