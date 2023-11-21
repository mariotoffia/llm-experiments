import os
from system_index import system_index, user_index
from embeddingsdb import EmbeddingsDb
from dotenv import load_dotenv
from langchain.pydantic_v1 import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnableParallel, RunnablePassthrough, RunnableConfig
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

system_index(embeddings=embeddings_db)

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


@cl.on_chat_start
async def on_chat_start():
    # Set the chain into the user session
    chain = setup_chain(model="gpt-4", temperature=0.0, streaming=True)
    cl.user_session.set("chain", chain.with_types(input_type=Question))

    await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k",
                        "gpt-4", "gpt-4-32k"],
                initial_index=2,
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
        content="Welcome to Crossbreed Energy OS. Please ask me a question!",
        author=ceos_user,
    ).send()

    # files = None

    # Wait for the user to upload a file
    # while files == None:
    #     files = await cl.AskFileMessage(
    #         content="Please upload a text file if you need som additional to discuss around!",
    #         accept=["text/plain"],
    #         max_size_mb=20,
    #         timeout=180,
    #     ).send()

    # file = files[0]

    # msg = cl.Message(content=f"Processing `{file.name}`...")
    # await msg.send()

    # # Decode the file
    # text = file.content.decode("utf-8")

    # # Write the file to user directory
    # with open(os.path.join(user_file_path, file.name), "w") as f:
    #     f.write(text)

    # user_index(embeddings=embeddings_db)

    # # Let the user know that the system is ready
    # msg.content = f"Processing `{file.name}` done!"

    # await msg.update()


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

# embedding = embeddings_db.embed(text="hello world") # NOSONAR
# print(embedding)

# https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed
# https://github.com/sudarshan-koirala/langchain-openai-chainlit/blob/main/txt_qa.py
# https://github.com/langchain-ai/langchain/tree/master/templates
# https://unstructured-io.github.io/unstructured/core/embedding.html
# https://github.com/langchain-ai/langchain/blob/master/templates/rag-conversation/rag_conversation/chain.py
