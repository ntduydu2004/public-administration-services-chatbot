# 1. Load Markdown Files
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from dotenv import load_dotenv
import os

path = "app/api/data/training_data/"
loader = DirectoryLoader(path, glob="*.md") 
docs = loader.load()

# 2. Split Text into Chunks (docs)
env = os.getenv("ENV", None)

if not env:
    if os.path.exists('.env'):
        load_dotenv(dotenv_path='.env')
    else:
        raise Exception(f"Env file file not found")

API_KEY = os.getenv("OPENAI_API_KEY")

generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-nano"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

from ragas.testset.transforms.extractors.llm_based import NERExtractor
from ragas.testset.transforms.splitters import HeadlineSplitter

transforms = [NERExtractor()]

from ragas.testset.persona import Persona

personas = [
    Persona(
        name="curious student",
        role_description="A student who is curious about the world and wants to learn more about different procedures.",
    ),
    Persona(
        name="civillian",
        role_description="A civillian who is having trouble with registering procedures on public service portal.",
    ),
    Persona(
        name="retired veteran",
        role_description="A retired military officer looking to claim retirement and veteran benefits through public services.",
    ),
    Persona(
        name="tech-savvy helper",
        role_description="A tech-savvy person helping family members navigate complicated e-government services and digital platforms.",
    ),
    Persona(
        name="immigrant worker",
        role_description="A foreign worker in Vietnam looking for help with residency procedures.",
    ),
    Persona(
        name="government clerk",
        role_description="A frontline public servant who explains administrative procedures and helps people complete service requests accurately.",
    ),
]

from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings, persona_list=personas)

from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0),
]

import asyncio

async def language_adapt():
    for query, _ in distribution:
        prompts = await query.adapt_prompts("vietnamese", llm=generator_llm)
        query.set_prompts(**prompts)

    dataset = generator.generate_with_langchain_docs(
        docs[:],
        testset_size=20,
        transforms=transforms,
        query_distribution=distribution,
    )
    return dataset

dataset = asyncio.run(language_adapt())

# 3. Save the Testset to a File
dataset.to_pandas()
dataset.to_csv("testset.csv")


###Requirement
#pip install ragas
#pip install langchain-community
#pip install langchain-openai
#pip install "unstructured[local-inference]" (if u encounter any error related to unstructured)