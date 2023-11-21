import os
from retriever import retrieve_file
from langchain.embeddings import OpenAIEmbeddings

## Initialize Chroma
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
# query_result = embeddings.embed_query("hello world")
# print(query_result)

# https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed

data = retrieve_file("data/training/smarter-heating.md")

for d in data:
    print(d.Question)
    print("_" * 80)
    print(d.Answer)
    print("_" * 80)
    print(d.Metadata)
    print("_" * 80)
    print()