# 1. Load Markdown Files
from langchain_community.document_loaders import DirectoryLoader

path = "app/api/data/training_data/"
loader = DirectoryLoader(path, glob="*.md")
datafile_text = loader.load()

# 2. Split Text into Chunks (docs)
from ragas.testset import TestsetGenerator

generator_llm = "gpt-3.5-turbo"
generator_embeddings = "text-embedding-ada-002"

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(datafile_text, testset_size=10)

# 3. Save the Testset to a File
dataset.to_pandas()