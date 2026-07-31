from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


def init_model(model_name, temperature=0, model_provider="google_genai"):
    return init_chat_model(model=model_name,temperature=temperature,model_provider=model_provider)


prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}")
messages = prompt.format_messages(adjective="funny", topic="chickens")

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "Tell me a {adjective} joke about {topic}")
])
messages2 = prompt2.format_messages(adjective="funny", topic="chickens")
# print(messages2)

model = init_model("gemini-3.5-flash-lite", temperature=0.7)
# response = model.invoke(messages2)
# print(response.content[0].get("text"))

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "energetic", "output": "tired"},
    {"input": "joyful", "output": "sorrowful"},
]

example_prompt = ChatPromptTemplate.from_messages([
      HumanMessage(content="Translate the following word to its opposite: {input}"),
      AIMessage(content="{output}")
])

fewshot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a good translator. "),
    fewshot_prompt,
    ("human", "Translate the following word to its opposite: {input}")
])

response = model.invoke(final_prompt.format(input="happy"))
print(response.content[0].get("text"))