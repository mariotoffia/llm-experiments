from typing import List
from retriever import StructuredData
from langchain.vectorstores.chroma import Chroma
from langchain.schema.embeddings import Embeddings

def store_structured_data(data: List[StructuredData], embeddings: Embeddings):
    texts = []
    metadata = []

    for res in data:
        if res.Question is None:
            # Enclosing Answer in triple quotes
            texts.append(f'Answer: """{res.Answer}"""')
        else:
            # Enclosing both Question and Answer in triple quotes
            formatted_text = f'Question: """{res.Question}"""\nAnswer: """{res.Answer}"""'
            texts.append(formatted_text)

        metadata.append(res.Metadata)

    Chroma.from_texts(
        texts=texts,
        metadatas=metadata,
        embedding=embeddings,
    )
