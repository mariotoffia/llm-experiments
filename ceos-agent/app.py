import os

from emb import EmbeddingsDb
from retriever import retrieve_directory
from langchain.embeddings import OpenAIEmbeddings

## Initialize Chroma
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

embDb = EmbeddingsDb(embeddings=embeddings)
# d = embDb.embed(text="hello world") # NOSONAR
# print(d)

def retrieve(pattern: str):
  training = retrieve_directory(pattern)

  for text in training:
    if len(text) == 0:
      continue

    stored = embDb.store_structured_data(data=text, id=text[0].File)

    if stored:
      print(f'\n*** STORED: {text[0].File} ***')
    else:
      print(f'\n*** SKIPPED: {text[0].File} ***')

# Index training and knowledge (if needed) - delete data/embeddings if you want to reindex
retrieve(pattern="data/training/*.md")
retrieve(pattern="data/knowledge/*.md")