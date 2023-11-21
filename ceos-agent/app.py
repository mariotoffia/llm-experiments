import os
from system_index import system_index, user_index
from embeddingsdb import EmbeddingsDb
from dotenv import load_dotenv
from langchain.pydantic_v1 import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.prompts.chat import ChatPromptTemplate

import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider

# Load environment variables from .env file
load_dotenv()

# Initialize System
ceos_user = "mario.toffia@crossbreed.se"
user_file_path = os.path.join(os.path.dirname(__file__), "data", "user")

if not os.path.exists(user_file_path):
    os.makedirs(user_file_path)

# Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
embeddings_db = EmbeddingsDb(embeddings=embeddings)

# Make sure index is up to date
system_index(embeddings=embeddings_db)
user_index(embeddings=embeddings_db)

# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


class Question(BaseModel):
    __root__: str


def setup_chain(model: str, temperature: float, streaming: bool) -> any:
    """
    Setup the LLM model and the chain.

    :param model: The model name
    :param temp: The temperature
    :param streaming: The streaming flag
    :return: The chain
    """
    print(f'model:{model}, temp:{temperature}, streaming:{streaming}')

    chat_model = ChatOpenAI(
        model_name=model,
        streaming=streaming,
        temperature=temperature,
    )

    chain = (
        {"context": embeddings_db.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

    return chain.with_types(input_type=Question)


def is_binary_file(file_name):
    # Common binary file extensions
    binary_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff',
        '.pdf', '.zip', '.rar',
        '.7z', '.mp3', '.wav', '.wma', '.mp4', '.mov',
        '.avi', '.flv', '.mkv'
    }

    # Get the file extension
    extension = file_name.lower().rsplit('.', 1)[-1]
    extension = '.' + extension

    # Check if the extension is in the list of binary extensions
    return extension in binary_extensions


@cl.on_chat_start
async def on_chat_start():
    # Set the chain into the user session
    chain = setup_chain(model="gpt-4-1106-preview", temperature=0.0, streaming=True)
    cl.user_session.set("chain", chain.with_types(input_type=Question))

    await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k",
                        "gpt-4", "gpt-4-1106-preview"],
                initial_index=3,
            ),
            Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=0,
                min=0,
                max=2,
                step=0.1,
            ),
        ]
    ).send()

    await cl.Avatar(
        name=ceos_user,
        url="https://avatars.githubusercontent.com/u/1716527?v=4"
    ).send()

    # Sending an image...
    elements = [
        cl.Image(
            name="CEOS",
            display="inline",
            url="https://www.crossbreed.se/wp-content/uploads/2021/06/energyos-top-text-mobile-webb.jpg",
        ),
    ]

    await cl.Message(content="", author=ceos_user, elements=elements).send()
    await cl.Message(
        content="Hi I'm here to help 😊 Please ask me any question about CEOS!",
        author=ceos_user,
    ).send()


@cl.on_settings_update
async def setup_agent(settings):
    chain = setup_chain(
        model=settings["Model"],
        temperature=settings["Temperature"],
        streaming=settings["Streaming"],
    )

    cl.user_session.set("chain", chain.with_types(input_type=Question))


@cl.on_message
async def on_message(message: cl.Message):
    for element in message.elements:
        if isinstance(element, cl.File):
            file = element

            if is_binary_file(file.name):
                with open(os.path.join(user_file_path, file.name), "wb") as f:
                    f.write(file.content)
            else:
                with open(os.path.join(user_file_path, file.name), "w") as f:
                    f.write(file.content.decode("utf-8"))

            user_index(embeddings=embeddings_db)

            await cl.Message(content=f"Processing `{file.name}` done!").send()
            return

    runnable = cl.user_session.get("chain")  # type: Runnable

    if runnable is None:
        msg = cl.Message(content="runnable is None!")
        await msg.send()
        return

    # clear message
    msg = cl.Message(content="", author=ceos_user)
    await msg.send()

    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.update()

# https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed
# https://github.com/sudarshan-koirala/langchain-openai-chainlit/blob/main/txt_qa.py
# https://github.com/langchain-ai/langchain/tree/master/templates
# https://unstructured-io.github.io/unstructured/core/embedding.html
# https://github.com/langchain-ai/langchain/blob/master/templates/rag-conversation/rag_conversation/chain.py
