import pandas as pd

import os
import json
from typing import List, Tuple

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

class AgentResponse(BaseModel):
    message: str = Field(description="The full, clear, and concise answer in Vietnamese, based only on the provided document.")
    images_list: List[str] = Field(description="A list of relevant image URLs or file paths from the document.")

def get_chat_prompt_template() -> ChatPromptTemplate:
    """
    Creates a LangChain ChatPromptTemplate with system and user messages.
    """
    system_message = """
I am Alice, a very kind and enthusiastic customer support agent who loves to help customers. I will answer the [USER] questions using only the [DOCUMENT] and following the [RULES].

[DOCUMENT]:
{context_str}

[RULES]:
- I will answer the user's questions using only the [DOCUMENT] provided.
- I am a kind and helpful human, the best customer support agent in existence.
- I never lie or invent answers not explicitly provided in [DOCUMENT].
- If I am unsure of the answer or the answer is not explicitly contained in [DOCUMENT], I will say: "Không tìm thấy thông tin liên quan.".
- I will only answer in Vietnamese
- I will always respond in JSON format with the following form:
{format_instructions}
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

    parser = JsonOutputParser(pydantic_object=AgentResponse)

    prompt_template = get_chat_prompt_template()

    chain = prompt_template | llm | parser

    try:
        response = chain.invoke({
            "user_query": user_query,
            "context_str": context_str,
            "format_instructions": parser.get_format_instructions(),
        })
        return response

    except Exception as e:
        # Handle potential errors from the API or parsing
        print(f"An error occurred: {e}")
        # Return a default error structure
        return {
            "message": "ERROR",
            "images_list": []
        }
        
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
        max_tokens=512 # Make sure this is sufficient for the JSON response
    )

    parser = JsonOutputParser(pydantic_object=AgentResponse)

    prompt_template = get_chat_prompt_template()

    chain = prompt_template | llm | parser

    try:
        response = chain.invoke({
            "user_query": user_query,
            "context_str": context_str,
            "format_instructions": parser.get_format_instructions(),
        })
        return response

    except Exception as e:
        print(f"An error occurred!")
        return {
            "message": e,
            "images_list": []
        }
        
def get_testset_response(testset_path: str, output_path: str) -> pd.DataFrame:
    dataFrame = pd.read_csv(testset_path)
    
    responses = []
    
    for index, row in dataFrame.iterrows():
        print(f"Processing row {index + 1}")
        user_query = row['user_input']
        context_str = row['retrieved_contexts']
        
        response = get_llm_response(user_query, context_str)
        
        responses.append(response['message'])
    
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

get_testset_response('retrieved_contexts.csv', 'gpt_testset_response.csv')