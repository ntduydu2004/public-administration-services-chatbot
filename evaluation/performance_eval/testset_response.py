import pandas as pd

import os

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv

def get_chat_prompt_template() -> ChatPromptTemplate:
    """
    Creates a LangChain ChatPromptTemplate with system and user messages.
    """
    system_message = """
I am Alice, a very kind and enthusiastic customer support agent who loves to help customers. I will answer the [USER] questions using only the [DOCUMENT].

[DOCUMENT]:
{context_str}
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "{user_query}")
    ])
    
    return prompt

def get_llm_response(user_query: str, context_str: str) -> dict:
    """
    Retrieves and parses a structured response from the LLM.

    Args:
        user_query: The user's question.
        context_str: The context document for the LLM to use.

    Returns:
        A dictionary matching the AgentResponse structure.
    """
    llm = ChatOpenAI(
        temperature=0.0,
        model="gpt-4.1-nano",
        max_tokens=512
    )

    prompt_template = get_chat_prompt_template()

    chain = prompt_template | llm
    try:
        response = chain.invoke({
            "user_query": user_query,
            "context_str": context_str,
        })
        return response.content

    except Exception as e:
        print(f"An error occurred: {e}")
        return e
        
def get_local_llm_response(user_query: str, context_str: str) -> dict:
    """
    Retrieves and parses a structured response from the LLM.

    Args:
        user_query: The user's question.
        context_str: The context document for the LLM to use.

    Returns:
        A dictionary matching the AgentResponse structure.
    """
    llama_cpp_url = "https://dogfish-close-separately.ngrok-free.app:443"
    
    llm = ChatOpenAI(
        base_url=llama_cpp_url,
        api_key="not-needed",
        temperature=0.0,
        model="local-model-name", 
        max_tokens=512
    )

    prompt_template = get_chat_prompt_template()

    chain = prompt_template | llm

    try:
        response = chain.invoke({
            "user_query": user_query,
            "context_str": context_str,
        })
        return response

    except Exception as e:
        print(f"An error occurred!")
        return e
        
def get_testset_response(testset_path: str, output_path: str) -> pd.DataFrame:
    dataFrame = pd.read_csv(testset_path)
    dataFrame = dataFrame.head(5)
    
    responses = []
    
    for index, row in dataFrame.iterrows():
        print(f"Processing row {index + 1}")
        user_query = row['user_input']
        context_str = row['retrieved_contexts']
        
        response = get_local_llm_response(user_query, context_str)
        
        responses.append(response)
    
    dataFrame['response'] = responses
    
    dataFrame.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"Responses saved to {output_path}")
    
def get_testset_response_fix(testset_path: str, output_path: str) -> pd.DataFrame:
    dataFrame = pd.read_excel(testset_path)
    
    responses = []
    
    for index, row in dataFrame.iterrows():
        avail_response = row['response']
        if avail_response != "ERROR":
            responses.append(avail_response)
            continue
        
        print(f"Processing row {index + 1}")
        user_query = row['user_input']
        context_str = row['retrieved_contexts']
        
        response = get_local_llm_response(user_query, context_str)
        responses.append(response)
    
    dataFrame['response'] = responses
    
    dataFrame.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"Responses saved to {output_path}")
    
env = os.getenv("ENV", None)

if not env:
    if os.path.exists('.env'):
        load_dotenv(dotenv_path='.env')
    else:
        raise Exception(f"Env file file not found")

API_KEY = os.getenv("OPENAI_API_KEY")

get_testset_response('retrieved_contexts.csv', 'qwen_testset_response.csv')