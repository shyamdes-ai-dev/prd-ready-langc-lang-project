"""
Prompt Engineering & Few-Shot Learning
======================================
Demonstrates:
1. Creating single-string templates via `ChatPromptTemplate.from_template()`
2. Creating structured multi-message templates via `ChatPromptTemplate.from_messages()`
3. In-context learning using `FewShotChatMessagePromptTemplate` to steer LLM outputs
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

# Load environment variables
load_dotenv()


def init_model(model_name: str, temperature: float = 0, model_provider: str = "google_genai"):
    """
    Helper function to instantiate a configured chat model.
    """
    return init_chat_model(
        model=model_name, temperature=temperature, model_provider=model_provider
    )


# -----------------------------------------------------------------------------
# 1. Simple Single-String Prompt Template
# -----------------------------------------------------------------------------
# Replaces {adjective} and {topic} placeholders dynamically
prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}")
messages = prompt.format_messages(adjective="funny", topic="chickens")
print("--- 1. Simple String Template Formatted Messages ---")
print(messages)
print()

# -----------------------------------------------------------------------------
# 2. Multi-Role Prompt Template (System + Human)
# -----------------------------------------------------------------------------
# Structures prompts by specifying distinct roles: system directives vs human queries
prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "Tell me a {adjective} joke about {topic}"),
    ]
)
messages2 = prompt2.format_messages(adjective="funny", topic="chickens")
print("--- 2. Multi-Role Formatted Messages ---")
print(messages2)
print()

# Initialize the model instance
model = init_model("gemini-3.5-flash-lite", temperature=0.7)

# -----------------------------------------------------------------------------
# 3. Few-Shot Prompting with Examples
# -----------------------------------------------------------------------------
# Few-shot prompting provides example input-output pairs to guide the model's
# formatting, tone, and reasoning style without model fine-tuning.

# Define training/demonstration examples
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "energetic", "output": "tired"},
    {"input": "joyful", "output": "sorrowful"},
]

# Define the template for how each individual example should be formatted
example_prompt = ChatPromptTemplate.from_messages(
    [
        HumanMessage(content="Translate the following word to its opposite: {input}"),
        AIMessage(content="{output}"),
    ]
)

# Bundle examples and example_prompt into a FewShotChatMessagePromptTemplate
fewshot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt, examples=examples
)

# Compose the final prompt containing system instruction, few-shot examples, and the target input
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a good translator. "),
        fewshot_prompt,
        ("human", "Translate the following word to its opposite: {input}"),
    ]
)

print("--- 3. Few-Shot Prompt Execution ---")
# Format the prompt with a new input word and invoke the model
response = model.invoke(final_prompt.format(input="happy"))
print("Opposite of 'happy':", response.content[0].get("text") if isinstance(response.content, list) else response.content)
