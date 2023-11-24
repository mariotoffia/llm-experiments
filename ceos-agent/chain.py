from operator import itemgetter
from typing import List, Tuple, TypeVar
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, Runnable
from langchain.pydantic_v1 import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

from embeddingsdb import EmbeddingsDb

# Conversational Prompt is a prompt of which gets loaded with the conversation
# in order to keep a history in the chat.
conversation_prompt = PromptTemplate.from_template("""Given the following conversation and a follow up question, rephrase the follow up 
question to be a standalone question, in its original language.

Chat History:
{chat_history}
                                                   
------------------------------
Follow Up Input: {question}
Standalone question:""")

# Answer Prompt is the one that we expect the assistant to provide an answer in
answer_prompt = ChatPromptTemplate.from_template("""{context}

                                          
-----------
Answer the question below based only on the above context (without mention the context in 
the response).

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


def _format_chat_history(chat_history: List[Tuple]) -> str:
    if not chat_history:
        return ""

    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer


def setup_chain(
        model: str,
        temperature: float,
        streaming: bool,
        embeddings_db: EmbeddingsDb,
        max_tokens=4096,
        use_history=False
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

    if use_history:
        # First define the chain that will generate a new question
        # from the original question and the the chat history
        conversation_chain = RunnableParallel(
            chat_history=lambda x: _format_chat_history(x["chat_history"]),
            question=lambda x: x["question"],
        ) | conversation_prompt | chat_model | { "question": StrOutputParser() }
        # This is the "answer" chain that gets newly generated question from
        # the conversation chain and populated the context from chroma.
        #
        # Then it performs the formatting of the answerPrompt -> call the model
        chain = conversation_chain | {
             "question": RunnablePassthrough(),
             "context": itemgetter("question") | embeddings_db.as_retriever() | format_docs,
         } | answer_prompt | chat_model | StrOutputParser()
        return chain
    else:
        chain = (
            {
                "context": embeddings_db.as_retriever() | format_docs,
                "question": RunnablePassthrough()
            }
            | answer_prompt
            | chat_model
            | StrOutputParser()
        )

        return chain
    # return chain.with_types(input_type=Question)