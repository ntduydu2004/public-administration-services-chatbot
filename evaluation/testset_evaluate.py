import os
import pandas as pd
from dotenv import load_dotenv
from ragas.metrics import Faithfulness, ResponseRelevancy

from datasets import Dataset

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

def loadDataset(dataset_path: str):
    df = pd.read_excel(dataset_path)
    df['retrieved_contexts'] = df['retrieved_contexts'].apply(lambda x: [str(x)])
    dataset = Dataset.from_pandas(df)
    return dataset

env = os.getenv("ENV", None)

if not env:
    if os.path.exists('.env'):
        load_dotenv(dotenv_path='.env')
    else:
        raise Exception(f"Env file file not found")

API_KEY = os.getenv("OPENAI_API_KEY")

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-nano"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

metrics = [ 
    Faithfulness(llm=evaluator_llm),
    ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
]

eval_dataset = loadDataset("qwen_testset_response.xlsx")

results = evaluate(dataset=eval_dataset, metrics=metrics)

results_df = results.to_pandas()
results_df.to_csv("qwen_evaluation_results.csv", index=False, encoding='utf-8-sig')