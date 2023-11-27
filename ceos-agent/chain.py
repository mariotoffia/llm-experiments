from chainlit import ChatSettings
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import initialize_agent, AgentType
from operator import itemgetter
from typing import List, Tuple
from tools.smhi import ForecastTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel, Runnable
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from langchain.globals import set_verbose, set_debug

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


def setup_chain_from_chat_settings(
        settings: any,
        embeddings_db: EmbeddingsDb,
) -> any:
    return setup_chain(
        model=settings["Model"],
        temperature=settings["Temperature"],
        streaming=settings["Streaming"],
        embeddings_db=embeddings_db,
        max_tokens=settings["MaxTokens"],
        chat_type=settings["Chain"],
        debug=settings["Debug"],
    )


def setup_chain(
        model: str,
        temperature: float,
        streaming: bool,
        embeddings_db: EmbeddingsDb,
        chat_type: str,
        debug: bool,
        max_tokens=4096,
) -> any:
    """
    Setup the LLM model and the chain.

    :param model: The model name
    :param temp: The temperature
    :param streaming: The streaming flag
    :params embeddings_db: The embeddings database
    :params chat_type: The chat type (see chat_start.py)
    :params debug: The debug flag
    :param max_tokens: The max tokens
    :return: The chain
    """
    print(
        f'model:{model}, temp:{temperature}, streaming:{streaming}, max_tokens:{max_tokens}'
    )

    print(f"chat_type: {chat_type} debug: {debug}")

    set_verbose(debug)
    set_debug(debug)

    chat_model = ChatOpenAI(
        model_name=model,
        streaming=streaming,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    switch_chat_type = {
        "no-history": case_chat_type_no_history,
        "history": case_chat_type_with_history,
        "history-with-tools": case_chat_type_with_history_and_tools,
    }

    chain = switch_chat_type[chat_type](chat_model, embeddings_db, debug)

    return chain


def case_chat_type_with_history_and_tools(
        model: ChatOpenAI,
        embeddings_db: EmbeddingsDb,
        debug: bool,
) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are very powerful assistant, but bad at calculating lengths of words."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | model
        | OpenAIFunctionsAgentOutputParser()
    )

    chain = initialize_agent(
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        llm=model,
        # return_direct=True?? -> https://github.com/Chainlit/chainlit/issues/377
        tools=[ForecastTool()], verbose=debug, return_direct=True 
    )

    return chain


def case_chat_type_with_history(
        model: ChatOpenAI,
        embeddings_db: EmbeddingsDb,
        debug: bool,
) -> Runnable:
    # First define the chain that will generate a new question
    # from the original question and the the chat history
    conversation_chain = RunnableParallel(
        chat_history=lambda x: _format_chat_history(x["chat_history"]),
        question=lambda x: x["question"],
    ) | conversation_prompt | model | {"question": StrOutputParser()}
    # This is the "answer" chain that gets newly generated question from
    # the conversation chain and populated the context from chroma.
    #
    # Then it performs the formatting of the answerPrompt -> call the model
    chain = conversation_chain | {
        "question": RunnablePassthrough(),
        "context": itemgetter("question") | embeddings_db.as_retriever() | format_docs,
    } | answer_prompt | model | StrOutputParser()

    return chain


def case_chat_type_no_history(
        model: ChatOpenAI,
        embeddings_db: EmbeddingsDb,
        debug: bool,
) -> Runnable:
    chain = (
        {
            "context": itemgetter("question") | embeddings_db.as_retriever() | format_docs,
            "question": RunnablePassthrough()
        }
        | answer_prompt
        | model
        | StrOutputParser()
    )

    return chain
