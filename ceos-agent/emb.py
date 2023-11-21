import os
import shutil
import json
from typing import List
from retriever import StructuredData
from langchain.vectorstores.chroma import Chroma
from langchain.schema.embeddings import Embeddings


class EmbeddingsDb:
    """
    Embeddings database
    """
    chroma: Chroma
    embeddings_path: str = "./data/embeddings"
    embeddings: Embeddings

    def __init__(self, embeddings: Embeddings):
        """
        Constructor
        :param embeddings: The embeddings creator to use.
        """
        self.chroma = Chroma(
            embedding_function=embeddings,
            persist_directory=self.embeddings_path,
        )

        self.embeddings = embeddings

    def as_retriever(self):
        """
        Return the Chroma object as a retriever
        :return: Chroma object
        """
        return self.chroma.as_retriever()
    
    def reset(self):
        """
        Reset the vector store by delete all files and recreating the directory
        where the embeddings are stored.        
        :return:
        """
        if not os.path.exists(self.embeddings_path):        
            return

        for item in os.listdir(self.embeddings_path):
            item_path = os.path.join(self.embeddings_path, item)

            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    
    def store_structured_data(self, data: List[StructuredData], id: str = None) -> bool:
        """
        Store structured data in the vector store
        :param data:    List of StructuredData objects
        :param id:      Optional id, of which it checks if already indexed and skips
                        if such is the case. 
        :return:        True if the data was stored, False if the data was skipped
        """
        texts = []
        metadata = []

        id_path = os.path.join(self.embeddings_path,"indexed", id)

        if id is not None and os.path.exists(id_path):
            return False
        
        for res in data:
            if res.Question is None:
                # Enclosing Answer in triple quotes
                texts.append(f'Answer: """{res.Answer}"""')
            else:
                # Enclosing both Question and Answer in triple quotes
                formatted_text = f'Question: """{res.Question}"""\nAnswer: """{res.Answer}"""'
                texts.append(formatted_text)

            if "languages" in res.Metadata and isinstance(res.Metadata["languages"], list):
                res.Metadata["languages"] = ", ".join(res.Metadata["languages"])
                        
            metadata.append(res.Metadata)

        self.chroma.from_texts(
            texts=texts,
            metadatas=metadata,
            persist_directory=self.embeddings_path,
            embedding=self.embeddings,
        )        

        # Mark id as already done
        if id is not None:
            os.makedirs(os.path.dirname(id_path), exist_ok=True)
            with open(id_path, "w") as f:
                f.write(id)

        return True