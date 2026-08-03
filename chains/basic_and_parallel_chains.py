from google.genai.chats import Chat
from langchain_core.runnables import RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
load_dotenv()

model = init_chat_model(model="gemini-3.5-flash", model_provider="google_genai")

def basic_chain_illustration():
    prompt = ChatPromptTemplate.from_template("Summarize the following text in one sentence: {text}")
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser

    response = chain.invoke({"text": "Scripts and CLI programs: Yes, it's a good practice because the process may exit immediately after the last LLM call.FastAPI, Django, or long-running servers: Usually not necessary after every request, because the process stays alive and the background uploader has time to send traces. You might only call it during application shutdown if needed"})
    print("Summary: ", response)


def parallel_chain_illustration():
    prompt1 = ChatPromptTemplate.from_template("Summarize the following text in one sentence: {text}")
    prompt2 = ChatPromptTemplate.from_template("Extract keywords from the following text: {text1}")
    output_parser = StrOutputParser()
    
    analysis_chain = RunnableParallel(
        summary =  prompt1 | model | output_parser,
        keywords = prompt2 | model | output_parser,
    )

    response = analysis_chain.invoke({
        "text": "Scripts and CLI programs: Yes, it's a good practice because the process may exit immediately after the last LLM call.FastAPI, Django, or long-running servers: Usually not necessary after every request, because the process stays alive and the background uploader has time to send traces. You might only call it during application shutdown if needed",
        "text1": "Why call flush(): Langsmith uses a background uploader to send traces to the server, but if the application exits before the uploader has sent all traces (common in scripts/CLIs), those traces will be lost. Flushing ensures all queued traces are sent before exit. When *not* to call flush(): For long-running servers, the uploader typically sends traces in the background without blocking, so explicit flushing after every request is usually unnecessary unless you want to guarantee immediate delivery."
        })
    print("Summary: ", response["summary"])
    print("Keywords: ", response["keywords"])


def passthrough_chain_illustration():
    prompt = ChatPromptTemplate.from_template(
        """
            Original question: {question}\n
            Context: {context}\n\n
            Answer the question based on the context
        
        """
    )

    def fake_retriever(input_dict):
        return "Langchain was created by Harrison Chase in 2022"

    chain = (
        RunnableParallel(context=RunnableLambda(fake_retriever), question=RunnablePassthrough()) 
        | RunnableLambda(
            lambda x: {"context": x["context"],
                       "question": x["question"]["question"]}
        ) 
        | prompt 
        | model
        | StrOutputParser()
    )
    response = chain.invoke({"question": "Who created Langchain?"})
    print("response: ", response)



def chain_branching():
    """Demonstrates conditional branching in chains using RunnableBranch"""
    
    code_prompt = ChatPromptTemplate.from_template("You are a coding expert. Help with: {input}")
    general_prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer: {input}")

    classifier_prompt = ChatPromptTemplate.from_template("Classifiy this as 'code' or 'general' {input}\n Return only the classification")
    
    classifier_chain = classifier_prompt | model | StrOutputParser()

    def is_code_question(input_dict):
        """Check if the question is a code-related question"""
        classification = classifier_chain.invoke(input_dict)
        return "code" in classification.lower() 
    
    branch = RunnableBranch(
        (is_code_question, code_prompt | model | StrOutputParser()),
        (general_prompt | model | StrOutputParser()),
    )

    questions = [
        "How do I write a for loop in Python?",
        "What's the weather like today?"
    ]

    for q in questions:
        result = branch.invoke({"input": q})
        print(f"Question: {q}")
        print(f"Answer: {result}")
        print("-" * 60) 
    

def demo_debbuging():
    prompt = ChatPromptTemplate.from_template("Say hello to {name}")
    chain = prompt | model | StrOutputParser()

    # Method 1: Get configuration
    print("Chain input schema: \n", chain.input_schema.model_json_schema())    
    print("\nChain output schema: \n", chain.output_schema.model_json_schema())

    # Method 2: Use with_config for tracing
    result= chain.with_config(run_name="greeting_chain").invoke({"name": "Alice"})
    print(f"Greeting: {result}")

    # Method 3: Inspect intermediate Steps
    # Using RunnableLambda for logging
    
    def log_step(x, step_name=""):
        print(f"[{step_name}] {type(x).__name__} : {str(x)}[:100]")
        return x

    debug_chain = (
        prompt
        | RunnableLambda(lambda x: log_step(x, "After prompt"))
        | model
        | RunnableLambda(lambda x: log_step(x, "After model"))
        | StrOutputParser()
        | RunnableLambda(lambda x: log_step(x, "After output_parser"))
    )
    debug_chain.invoke({"name": "Bob"})


# basic_chain_illustration()
# parallel_chain_illustration()
# passthrough_chain_illustration()
# chain_branching()
# demo_debbuging()