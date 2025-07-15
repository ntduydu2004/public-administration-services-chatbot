import pandas as pd

import os
import json
from typing import List, Tuple

# Make sure to set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Pydantic is used to define the desired JSON output structure
from pydantic import BaseModel, Field

# LangChain components for interacting with the LLM
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv


# 1. Define the desired output structure using Pydantic
class AgentResponse(BaseModel):
    message: str = Field(description="The full, clear, and concise answer in Vietnamese, based only on the provided document.")
    images_list: List[str] = Field(description="A list of relevant image URLs or file paths from the document.")


# 2. Refactored function to create the prompt template
def get_chat_prompt_template() -> ChatPromptTemplate:
    """
    Creates a LangChain ChatPromptTemplate with system and user messages.
    """
    system_message = """[AGENT]:
I am Alice, a very kind and enthusiastic customer support agent who loves to help customers. I will answer the [USER] questions using only the [DOCUMENT] and following the [RULES].

[DOCUMENT]:
{context_str}

[RULES]:
- I will answer the user's questions using only the [DOCUMENT] provided.
- I am a kind and helpful human, the best customer support agent in existence.
- I never lie or invent answers not explicitly provided in [DOCUMENT].
- If I am unsure of the answer or the answer is not explicitly contained in [DOCUMENT], I will say: "Không tìm thấy thông tin liên quan.".
- I will always respond in Vietnamese.
- I will only provide my response in the specified JSON format.

{format_instructions}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", "[USER]:\n{user_query}")
    ])
    
    return prompt

# 3. Main function to get the structured response from the LLM
def get_llm_response(user_query: str, context_str: str) -> dict:
    """
    Retrieves and parses a structured response from the LLM.

    Args:
        user_query: The user's question.
        context_str: The context document for the LLM to use.

    Returns:
        A dictionary matching the AgentResponse structure.
    """
    # Initialize the model
    llm = ChatOpenAI(
        temperature=0.0,
        model="gpt-4.1-nano",
        max_tokens=512
    )

    # Initialize the JSON output parser with our Pydantic model
    parser = JsonOutputParser(pydantic_object=AgentResponse)
    
    # Get the prompt template
    prompt_template = get_chat_prompt_template()

    # Create the chain: prompt -> model -> parser
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
            "is_escalate": True,
            "images_list": []
        }


env = os.getenv("ENV", None)

if not env:
    if os.path.exists('.env'):
        load_dotenv(dotenv_path='.env')
    else:
        raise Exception(f"Env file file not found")

API_KEY = os.getenv("OPENAI_API_KEY")


# response = get_llm_response(query, context)
# message_text = response['message']
        

def get_testset_response(testset_path: str, output_path: str) -> pd.DataFrame:
    dataFrame = pd.read_csv(testset_path)
    dataFrame = dataFrame.head(5)
    
    responses = []
    
    for index, row in dataFrame.iterrows():
        user_query = row['user_input']
        context_str = row['retrieved_contexts']
        
        response = get_llm_response(user_query, context_str)
        
        responses.append(response['message'])
    
    dataFrame['response'] = responses
    
    dataFrame.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    
        
    