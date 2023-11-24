from typing import List
import chainlit as cl
from chainlit.input_widget import Select, Switch, Slider


def get_chat_settings(use_history=False) -> cl.ChatSettings: 
  """
  gets the chat setting.
  :returns: cl.ChatSettings
  """
  return cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k",
                        "gpt-4", "gpt-4-1106-preview"],
                initial_index=3,
            ),
            Switch(id="Streaming", label="OpenAI - Stream Tokens", initial=True),
            Switch(id="UseHistory", label="Use History", initial=use_history),
            Slider(
                id="Temperature",
                label="OpenAI - Temperature",
                initial=0,
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id="MaxTokens",
                label="Max Tokens",
                initial=4096,
                min=1024,
                max=128*1024,
                step=512,
            ),
        ]
    )

def get_avatar(ceos_user: str) -> cl.Avatar:
    """
    gets the avatar.
    """
    return   cl.Avatar(
      name=ceos_user,
      url="https://avatars.githubusercontent.com/u/1716527?v=4"
  )

def get_initial_messages(ceos_user: str) -> List[cl.Message]:
  """
  returns the initial messages.
  :returns: List[cl.Message] (use await msg.sen() to send the messages.
  """
  elements = [
      cl.Image(
          name="CEOS",
          display="inline",
          url="https://www.crossbreed.se/wp-content/uploads/2021/06/energyos-top-text-mobile-webb.jpg",
      ),
  ]

  return [
    cl.Message(content="", author=ceos_user, elements=elements),
    cl.Message(
        content="Hi I'm here to help 😊 Please ask me any question about CEOS!",
        author=ceos_user,
    ),
  ]
