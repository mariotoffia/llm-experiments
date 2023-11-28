from chains.history_tools import HistoryWithToolsChain
from chains.history import HistoryChain
from chains.no_history import NoHistoryChain
from chains.base import BaseChain
from embeddingsdb import EmbeddingsDb

from langchain.chat_models import ChatOpenAI
from langchain.globals import set_verbose, set_debug


def setup_chain_from_chat_settings(
        settings: any,
        embeddings_db: EmbeddingsDb,
) -> BaseChain:
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
) -> BaseChain:
    """
    Setup the LLM model and the chain.

    :param model: The model name
    :param temp: The temperature
    :param streaming: The streaming flag
    :params embeddings_db: The embeddings database
    :params chat_type: The chat type (see chat_start.py)
    :params debug: The debug flag
    :param max_tokens: The max tokens
    :return: The BaseChain derivate
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
) -> BaseChain:
    return HistoryWithToolsChain(model=model, embeddings_db=embeddings_db, debug=debug)


def case_chat_type_with_history(
        model: ChatOpenAI,
        embeddings_db: EmbeddingsDb,
        debug: bool,
) -> BaseChain:
    return HistoryChain(model=model, embeddings_db=embeddings_db, debug=debug)

def case_chat_type_no_history(
        model: ChatOpenAI,
        embeddings_db: EmbeddingsDb,
        debug: bool,
) -> BaseChain:
    return NoHistoryChain(model=model, embeddings_db=embeddings_db, debug=debug)