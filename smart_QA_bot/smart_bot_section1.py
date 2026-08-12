"""
Production-Ready Smart Q&A Bot Microservice
============================================
Demonstrates an enterprise-ready question-answering architecture featuring:
1. Strict Pydantic Schema Contracts (`QAResponse`) for deterministic outputs.
2. End-to-end distributed tracing & monitoring via LangSmith (`@traceable`).
3. Graceful error handling & fallback responses to avoid runtime crashes.
4. High-throughput parallel execution via `chain.batch()`.
5. Safe telemetry flushing with `Client().flush()`.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client, traceable
from pydantic import BaseModel, Field

# Load API credentials from .env
load_dotenv()

# Configure LangSmith telemetry and project metadata if API key is provided
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv(
        "LANGSMITH_PROJECT", "Smart Q&A Bot Project"
    )
    print("LangSmith environment setup complete.")
else:
    print("LangSmith API key not found; running in local-only mode.")


# -----------------------------------------------------------------------------
# Pydantic Schema Definition
# -----------------------------------------------------------------------------
class QAResponse(BaseModel):
    """
    Structured response contract returned by the Smart Q&A Bot.
    Guarantees consistent, type-safe attributes for downstream consumers/APIs.
    """
    answer: str = Field(description="The direct answer to the user's question.")
    confidence: str = Field(description="Confidence level: 'high', 'medium', or 'low'.")
    reasoning: str = Field(description="The underlying reasoning behind the provided answer.")
    follow_up_questions: List[str] = Field(
        description="Suggested follow-up questions related to the topic.",
        default_factory=list,
    )
    sources_needed: bool = Field(
        description="Flag indicating if external verification or sources are required.",
        default=False,
    )


# -----------------------------------------------------------------------------
# Smart Q&A Bot Implementation
# -----------------------------------------------------------------------------
class SmartQABot:
    """
    Encapsulates the LLM chain, prompt guidelines, and structured output formatting.
    """

    def __init__(self, model_name: str = "gemini-3.5-flash-lite"):
        """
        Initializes the model with native structured output binding and system instructions.
        """
        # Bind Pydantic schema to the model
        self.model = init_chat_model(
            model=model_name, model_provider="google_genai"
        ).with_structured_output(QAResponse)

        # Define system instructions and user message placeholders
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a knowledgeable Q&A assistant.
                 
Your Guidelines:
- Answer questions accurately and concisely.
- Be honest about uncertainty: set confidence to 'low' if unsure.
- Provide clear reasoning for your answers.
- Suggest relevant follow-up questions.
- Flag if external sources are needed.
                 
Always respond with accurate, helpful information.""",
                ),
                ("human", "{question}"),
            ]
        )
        # Assemble LCEL chain
        self.chain = self.prompt | self.model

    @traceable(name="ask_question", run_type="chain")
    def ask(self, question: str) -> QAResponse:
        """
        Asynchronously or synchronously invokes the Q&A chain with robust error recovery.

        Args:
            question (str): The user query.

        Returns:
            QAResponse: Structured response object or graceful fallback on exception.
        """
        try:
            response: QAResponse = self.chain.invoke({"question": question})
            return response
        except Exception as e:
            # Graceful error fallback to avoid unhandled service exceptions
            return QAResponse(
                answer=f"Sorry - I couldn't process that question. Error: {str(e)}",
                confidence="low",
                reasoning=str(e),
                follow_up_questions=[],
                sources_needed=False,
            )

    @traceable(name="ask_batch", run_type="chain")
    def ask_batch(self, questions: List[str]) -> List[QAResponse]:
        """
        Processes multiple user questions in parallel using LCEL batching.

        Args:
            questions (List[str]): List of question strings.

        Returns:
            List[QAResponse]: List of structured response objects.
        """
        inputs = [{"question": q} for q in questions]
        return self.chain.batch(inputs)


# -----------------------------------------------------------------------------
# Demonstrations
# -----------------------------------------------------------------------------
def qa_bot():
    """Demonstrates standard single-question invocations."""
    bot = SmartQABot()
    questions = [
        "What is the capital of France?",
        "What is the largest city in the United States?",
        "What is the smallest country in the world?",
    ]
    print("\n" + "=" * 60)
    print("Smart Q&A Bot - Structured Output Demo")
    print("=" * 60)

    for question in questions:
        print(f"\nQ: {question}")
        print("-" * 60)
        response = bot.ask(question)
        print(f"Answer             : {response.answer}")
        print(f"Confidence         : {response.confidence}")
        print(f"Reasoning          : {response.reasoning}")
        print(f"Follow-up Questions: {', '.join(response.follow_up_questions)}")
        print(f"Sources Needed     : {response.sources_needed}")
        print("-" * 60)


@traceable(name="error_handling_demo", run_type="chain")
def error_handling():
    """Demonstrates resilient fallback on malformed or stressful inputs."""
    bot = SmartQABot()
    print("\n" + "=" * 60)
    print("Smart Q&A Bot - Error Handling Demo")
    print("=" * 60)

    long_question = "What is " + "Very " * 100 + "important?"
    response = bot.ask(long_question)

    print(f"Answer     : {response.answer}")
    print(f"Confidence : {response.confidence}")
    print(f"Reasoning  : {response.reasoning}")


@traceable(name="batch_processing_demo", run_type="chain")
def batch_processing():
    """Demonstrates high-throughput parallel batch querying."""
    bot = SmartQABot()
    print("\n" + "=" * 60)
    print("Smart Q&A Bot - Batch Processing Demo")
    print("=" * 60)

    questions = [
        "What is the capital of France?",
        "What is the largest desert in the world?",
        "What is the speed of light in vacuum?",
    ]

    responses = bot.ask_batch(questions)
    for question, response in zip(questions, responses):
        print(f"\nQ: {question}")
        print(f"A: {response.answer} (Confidence: {response.confidence})")


def main():
    """
    Main runner executing all demonstrations and guaranteeing telemetry flush on exit.
    """
    try:
        qa_bot()
        batch_processing()
        error_handling()
    finally:
        # Crucial for scripts/CLIs: Ensure all asynchronous background telemetry traces
        # are delivered to LangSmith before the Python process exits.
        client = Client()
        client.flush()
        print("\nLangSmith traces successfully flushed.")


if __name__ == "__main__":
    main()
