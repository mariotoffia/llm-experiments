import os

from emb import EmbeddingsDb
from retriever import retrieve_directory
from langchain.embeddings import OpenAIEmbeddings

## Initialize Chroma
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

embDb = EmbeddingsDb(embeddings=embeddings)
# d = embDb.embed(text="hello world")
# print(d)

all = retrieve_directory("data/training/*.md")

for text in all:
  if len(text) == 0:
    continue

  stored = embDb.store_structured_data(data=text, id=text[0].File)

  if stored:
    print(f'\n*** STORED: {text[0].File} ***')
  else:
    print(f'\n*** SKIPPED: {text[0].File} ***')

# for d in data:
#     print(d.Question)
#     print("_" * 80)
#     print(d.Answer)
#     print("_" * 80)
#     print(d.Metadata)
#     print("_" * 80)
#     print()
