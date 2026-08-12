"""
LangChain Expression Language (LCEL) Core Concepts
==================================================
Demonstrates foundational LCEL paradigms:
1. Basic Chain Composition using pipe operator (`prompt | model | parser`)
2. Batch Processing for concurrent requests (`chain.batch(...)`)
3. Real-Time Token Streaming (`chain.stream(...)`)
4. Prompt templating with input variables
5. Universal Chat Model Initialization via `init_chat_model`
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables (such as GEMINI_API_KEY)
load_dotenv()


def demo_basic_chain():
    """
    Demonstrates a standard LCEL chain:
    [Prompt Template] -> [Chat Model] -> [String Output Parser]
    
    The pipe operator `|` connects Runnables, passing the output of the previous
    component directly as the input to the next component.
    """
    print("--- 1. Demo Basic Chain ---")
    # Step 1: Define template with placeholder variable {question}
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Anser in one sentence: {question}"
    )

    # Step 2: Initialize LLM
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.7)

    # Step 3: Parse AIMessage output into clean string
    output_parser = StrOutputParser()

    # Step 4: Assemble chain via LCEL
    chain = prompt | model | output_parser

    # Step 5: Invoke chain with dictionary of inputs
    response = chain.invoke({"question": "Why is the sky blue?"})
    print("Result:", response)
    print()


def demo_batch_execution():
    """
    Demonstrates processing multiple inputs concurrently using `.batch()`.
    LCEL automatically handles threading and concurrency under the hood to maximize
    throughput without manual async/multiprocessing setup.
    """
    print("--- 2. Demo Batch Execution ---")
    prompt = ChatPromptTemplate.from_template(
        "You are a good language translator. Translate the following text to {target_language}: {text}"
    )
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.7)
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    # Provide a list of input dictionaries
    inputs = [
        {"target_language": "french", "text": "Hello Shyam"},
        {"target_language": "spanish", "text": "Hello Shyam"},
        {"target_language": "gujarati", "text": "Hello Shyam"},
    ]

    # Process all inputs in parallel
    responses = chain.batch(inputs)
    for inp, res in zip(inputs, responses):
        print(f"[{inp['target_language'].capitalize()}]: {res}")
    print()


def demo_streaming():
    """
    Demonstrates token-by-token streaming using `.stream()`.
    Essential for low-latency interactive applications (e.g. Chatbots, UIs).
    """
    print("--- 3. Demo Streaming ---")
    prompt = ChatPromptTemplate.from_template("{question}")
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.7)
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    print("Streaming response: ", end="")
    # Iterate over incoming chunks as they arrive from the model
    for chunk in chain.stream(
        {"question": "What is happened in Sansad Bhavan in India, in brief?"}
    ):
        print(chunk, end="", flush=True)
    print("\n")


def excercise_first_chain():
    """
    Demonstrates a multi-variable prompt template streamed in real-time.
    """
    print("--- 4. Exercise First Chain ---")
    prompt = ChatPromptTemplate.from_template(
        "Generate a market tagline for the product {product} for the target audience {audience}"
    )
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.7)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser

    print("Tagline output: ", end="")
    for chunk in chain.stream(
        {"product": "Shampoo", "audience": "18-25 year old males"}
    ):
        print(chunk, end="", flush=True)
    print("\n")


def universal_way_of_initialization_of_model():
    """
    Demonstrates using `init_chat_model()` to instantiate models uniformly
    regardless of whether the backend is Google, Anthropic, OpenAI, or Ollama.
    """
    print("--- 5. Universal Model Initialization ---")
    model = init_chat_model(
        "gemini-3.5-flash-lite", temperature=0.7, model_provider="google_genai"
    )
    response = model.invoke("why the sky is blue in one line?")
    print("Response:", response.content[0].get("text") if isinstance(response.content, list) else response.content)
    print()


def main():
    """Execute all core concepts demos."""
    demo_basic_chain()
    demo_batch_execution()
    demo_streaming()
    excercise_first_chain()
    universal_way_of_initialization_of_model()


if __name__ == "__main__":
    main()
