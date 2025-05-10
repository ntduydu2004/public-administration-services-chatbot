# 1. Load Markdown Files
from langchain_community.document_loaders import DirectoryLoader
from langchain import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

path = "app/api/data/training_data/"
loader = DirectoryLoader(path, glob="*.md")
datafile_text = loader.load()

# 2. Split Text into Chunks (docs)
from ragas.testset import TestsetGenerator

generator_llm = "gpt-3.5-turbo"
generator_embeddings = "text-embedding-ada-002"

env = os.getenv("ENV", None)

if not env:
    if os.path.exist('.env'):
        load_dotenv(dotenv_path='.env')
    else:
        raise Exception(f"Env file file not found")

API_KEY = os.getenv("OPENAI_API_KEY")

embedding = OpenAIEmbeddings(openai_api_key=API_KEY)

llm = OpenAI(
    temparature = 0,
    model_name = generator_llm
)

generator = TestsetGenerator(llm=llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(datafile_text, testset_size=10)

# 3. Save the Testset to a File
dataset.to_pandas()