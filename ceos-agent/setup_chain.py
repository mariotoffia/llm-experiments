from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.pydantic_v1 import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from embeddingsdb import EmbeddingsDb

# Prompt
prompt = ChatPromptTemplate.from_template("""{context}

                                          
-----------
Answer the question below based only on the above context without mention the context.

Question: {question}
""")


class Question(BaseModel):
    __root__: str


def format_docs(docs):
    """
    Format the documents into a string. otherwise the output would
    be a list of [Document(page_content=...),...]
    """
    return "\n\n".join(doc.page_content for doc in docs)


def setup_chain(
        model: str,
        temperature: float,
        streaming: bool,
        embeddings_db: EmbeddingsDb,
        max_tokens=4096,        
        ) -> any:
    """
    Setup the LLM model and the chain.

    :param model: The model name
    :param temp: The temperature
    :param streaming: The streaming flag
    :params embeddings_db: The embeddings database
    :param max_tokens: The max tokens
    :return: The chain
    """
    print(
        f'model:{model}, temp:{temperature}, streaming:{streaming}, max_tokens:{max_tokens}')

    chat_model = ChatOpenAI(
        model_name=model,
        streaming=streaming,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    chain = (
        {
            "context": embeddings_db.as_retriever() | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )

    return chain.with_types(input_type=Question)
