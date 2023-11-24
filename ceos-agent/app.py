import os
from dotenv import load_dotenv
import chainlit as cl
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import Runnable, RunnableConfig
from langchain.globals import set_verbose, set_debug

from chat_start import get_chat_settings, get_avatar, get_initial_messages
from chain import setup_chain
from index import system_index, user_index, add_file_to_user_index
from embeddingsdb import EmbeddingsDb

# Load environment variables from .env file
load_dotenv()

# Set debug level
debug = True

set_verbose(debug)
set_debug(debug)

# Initialize System
use_history = False
max_tokens = 4096
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
    chain = setup_chain(
        model="gpt-4-1106-preview",
        temperature=0.0,
        streaming=True,
        embeddings_db=embeddings_db,
        use_history=use_history,
        max_tokens=max_tokens,
    )

    cl.user_session.set("chain", chain)
    cl.user_session.set("chat_history", [])

    await get_chat_settings(use_history=use_history).send()
    await get_avatar(ceos_user).send()

    for message in get_initial_messages(ceos_user):
        await message.send()


@cl.on_settings_update
async def setup_agent(settings):    
    cl.user_session.set("chain", setup_chain(
        model=settings["Model"],
        temperature=settings["Temperature"],
        streaming=settings["Streaming"],
        embeddings_db=embeddings_db,
        max_tokens=settings["MaxTokens"],
        use_history=settings["UseHistory"],
    ))


@cl.on_message
async def on_message(message: cl.Message):
    for element in message.elements:
        if isinstance(element, cl.File):
            file = element

            await cl.Message(content=f"Processing `{file.name}`...").send()            
            add_file_to_user_index(file.name, embeddings_db, file.content)
            await cl.Message(content=f"...done!").send()
            return

    chain = cl.user_session.get("chain")  # type: Runnable

    if chain is None:
        msg = cl.Message(content="runnable is None!")
        await msg.send()
        return

    # clear message
    msg = cl.Message(content="", author=ceos_user)
    await msg.send()

    chat_history = cl.user_session.get("chat_history")  # type: list

    async for chunk in chain.astream(
        input={
            "question": message.content,
            "chat_history": chat_history,  # Used when history is enabled
        },
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    # append to chat history
    chat_history += [(message.content, msg.content)]

    await msg.update()
