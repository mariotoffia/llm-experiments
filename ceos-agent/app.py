import os
from dotenv import load_dotenv
import chainlit as cl
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import Runnable, RunnableConfig

from chat_start import get_chat_settings, get_avatar, get_initial_messages
from chain import setup_chain_from_chat_settings
from index import system_index, user_index, add_file_to_user_index
from embeddingsdb import EmbeddingsDb

# Load environment variables from .env file
load_dotenv()

# Set debug level
debug = True


# Initialize System
ceos_user = "mario.toffia@crossbreed.se"
user_file_path = os.path.join(os.path.dirname(__file__), "data", "user")

if not os.path.exists(user_file_path):
    os.makedirs(user_file_path)

# Embeddings
embeddings_db = EmbeddingsDb(
    embeddings=OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]),
)

# Make sure index is up to date
system_index(embeddings=embeddings_db)
user_index(embeddings=embeddings_db)


@cl.on_chat_start
async def on_chat_start():
    settings = get_chat_settings()
    chain = setup_chain_from_chat_settings(settings.settings(), embeddings_db)

    cl.user_session.set("chain", chain)
    cl.user_session.set("chat_history", [])

    await settings.send()
    await get_avatar(ceos_user).send()

    for message in get_initial_messages(ceos_user):
        await message.send()


@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set(
        "chain", setup_chain_from_chat_settings(settings, embeddings_db),
    )


@cl.on_message
async def on_message(message: cl.Message):
    for element in message.elements:
        if isinstance(element, cl.File):
            file = element

            await cl.Message(content=f"Processing `{file.name}`...").send()
            add_file_to_user_index(file.name, embeddings_db, file.content)
            await cl.Message(content="...done!").send()
            return

    # Get the current chain
    chain = cl.user_session.get("chain")  # type: Runnable

    if chain is None:
        msg = cl.Message(content="chain is None!")
        await msg.send()
        return

    # clear message
    msg = cl.Message(content="", author=ceos_user)
    await msg.send()

    chat_history = cl.user_session.get("chat_history")  # type: list

    async for chunk in chain.astream(
        input={
            "question": message.content,
            "input": message.content,  # Duplicate for tool chain - should be merged!
            "chat_history": chat_history,  # Used when history is enabled
            "agent_scratchpad": [],  # Used when tools is enabled
        },
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        if chunk is str:
            await msg.stream_token(token=chunk)

        if isinstance(chunk, dict) and chunk and "output" in chunk:
            await msg.stream_token(token=chunk["output"])

    # append to chat history
    chat_history += [(message.content, msg.content)]

    await msg.update()
