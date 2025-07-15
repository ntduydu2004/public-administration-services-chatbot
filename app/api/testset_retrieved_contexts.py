import pandas as pd
from util import sanitize_input, sanitize_output
from llm import (
    get_token_count,
    get_embeddings,
    get_nodes_by_embedding,
    get_prompt_template,
)
from models import Node, Project, Organization, get_session, get_engine
from fastapi import Depends, HTTPException
import json
from typing import List, Optional
from config import LLM_DEFAULT_TEMPERATURE
from langchain import OpenAI
import openai

from sqlmodel import Session

import time

def read_testset_with_pandas(filename):
    """
    Reads a CSV file into a pandas DataFrame.

    Args:
        filename (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing the data, or None if an error occurs.
    """
    try:
        # read_csv automatically uses the first row as the header
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def retrieve_llm_response(    
    query_str: str,
    model_name: str = "gpt-4.1-nano",
    temperature: Optional[float] = LLM_DEFAULT_TEMPERATURE,
    max_output_tokens: Optional[int] = 256,
    prefix_messages: Optional[List[dict]] = None,
):
    llm = OpenAI(
        temperature=temperature,
        model_name=model_name,
        max_tokens=max_output_tokens,
        prefix_messages=prefix_messages,
    )
    
    try:
        result = llm(prompt=query_str)
        
    except openai.error.InvalidRequestError as e: 
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")
    
    return sanitize_output(result)

def get_result(query_str: str):
    project: Project = None
    organization: Organization = None
    context_str = ""
    session: Session = Session(get_engine())
    
    query_str = sanitize_input(query_str)

    # -----------------------
    # Create input embeddings
    # -----------------------
    _, embeddings = get_embeddings(query_str)

    query_embeddings = embeddings[0]

    # ------------------------
    # Search for similar nodes
    # ------------------------
    nodes = get_nodes_by_embedding(
        query_embeddings,
        session=session,
    )

    if len(nodes) > 0:
        if (not project or not organization) and session:
            document = session.get(Node, nodes[0].id).document
            project = document.project
            organization = project.organization

        context_str = "\n\n".join([node.text for node in nodes])
            
    return context_str
    

input_file = 'testset.csv'
output_file = 'retrieved_contexts.csv'

# Add retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5

dataframe = read_testset_with_pandas(input_file)

if dataframe is not None:
    retrieved_contexts_list = []

    for index, row in dataframe.iterrows():
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                retrieved_contexts = get_result(row['user_input'])
                retrieved_contexts_list.append(retrieved_contexts)
                success = True
                break

            except Exception as e:
                
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY) 
                else:
                    retrieved_contexts_list.append("ERROR_DATABASE_CONNECTION_FAILED")
                    break 

    # --- End of the resilient block ---

    # This part of the code is now safe because retrieved_contexts_list will always have the correct length
    print("\nProcessing finished. Adding new columns to the DataFrame.")
    dataframe['retrieved_contexts'] = retrieved_contexts_list
    
    dataframe.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSuccessfully created '{output_file}'. Check for 'ERROR_' values.")