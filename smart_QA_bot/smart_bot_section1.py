"""
A production-ready question-answering both with structured output
"""

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
from langsmith import traceable, Client
import os

load_dotenv()

if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv(
        "LANGSMITH_PROJECT", "Smart Q&A Bot Project"
    )
    print("Langsmith environment setup complete")
else:
    print("Langsmith API key not found")


# Schema Definition


class QAResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    confidence: str = Field(description="Confidence leverl: high, medium or low")
    reasoning: str = Field(description="The reasoning behind the answer provided")
    follow_up_questions: List[str] = Field(
        description="List of follow-up questions related to the topic",
        default_factory=list,
    )
    sources_needed: bool = Field(
        description="Whether sources are needed to answer the question", default=False
    )


class SmartQABot:
    def __init__(self, model_name="gemini-3.5-flash-lite"):
        self.model = init_chat_model(
            model=model_name, model_provider="google_genai"
        ).with_structured_output(QAResponse)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a knowledgeable Q&A assistant
                 
                 Your Guidelines:
                 - Answer questions accurately and concidely
                 - Be honest about uncertainity - set confidence to 'low' if unsure
                 - Provide clear reasoning for your answers
                 - Suggest relevant follow-up questions
                 - Flag if sources are needed
                 - Indicate if external sources would help
                 
                Always response with accurate, helpful information.""",
                ),
                ("human", "{question}"),
            ]
        )
        self.chain = self.prompt | self.model

    @traceable(name="ask_question", run_type="chain")
    def ask(self, question: str) -> QAResponse:
        try:
            response = self.chain.invoke({"question": question})
            return response
        except Exception as e:
            return QAResponse(
                answer=f"Sorry - I couldn't process that question. Error: {str(e)}",
                confidence="low",
                reasoning=str(e),
                follow_up_questions=[],
                sources_needed=False,
            )

    @traceable(name="ask_batch", run_type="chain")
    def ask_batch(self, questions: List[str]) -> List[QAResponse]:
        """Ask multiple questions in parallel using batch invoke"""
        inputs = [{"question": q} for q in questions]
        return self.chain.batch(inputs)


def qa_bot():
    bot = SmartQABot()

    questions = [
        "What is the capital of France?",
        "What is the largest city in the United States?",
        "What is the smallest country in the world?",
        "What is the most populous country in the world?",
        "What is the largest desert in the world?",
    ]
    print("=" * 60)
    print("Smart Q&A Bot - Structured Output Demo")
    print("=" * 60)

    print("Asking questions...")

    for question in questions:
        print(f"\nQ: {question}")
        print("-" * 60)

        response = bot.ask(question)

        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Follow-up Questions: {', '.join(response.follow_up_questions)}")
        print(f"Sources Needed: {response.sources_needed}")
        print("-" * 60)


@traceable(name="error_handling_demo", run_type="chain")
def error_handling():
    """Demonstrate error handling"""

    bot = SmartQABot()

    print("\n", "=" * 60)
    print("Smart Q&A Bot - Error Handling Demo")
    print("=" * 60)

    long_question = "What is " + "Very " * 100 + "important?"

    response = bot.ask(long_question)

    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence}")
    print(f"Reasoning: {response.reasoning}")
    print(f"Follow-up Questions: {', '.join(response.follow_up_questions)}")
    print(f"Sources Needed: {response.sources_needed}")
    print("-" * 60)


@traceable(name="batch_processing_demo", run_type="chain")
def batch_processing():
    """Demontrating batch processing."""

    bot = SmartQABot()

    print("\n", "=" * 60)
    print("Smart Q&A Bot - Batch Processing Demo")
    print("=" * 60)

    questions = [
        "What is the capital of France?",
        "What is the largest city in the United States?",
        "What is the smallest country in the world?",
        "What is the most populous country in the world?",
        "What is the largest desert in the world?",
    ]

    responses = bot.ask_batch(questions)

    for question, response in zip(questions, responses):
        print(f"\nQ: {question}")
        print("-" * 60)

        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Follow-up Questions: {', '.join(response.follow_up_questions)}")
        print(f"Sources Needed: {response.sources_needed}")
        print("-" * 60)


def main():
    try:
        qa_bot()
        batch_processing()
        error_handling()

    finally:
        client = Client()
        client.flush()  # ensure all traces are sent to langsmith


if __name__ == "__main__":
    main()
