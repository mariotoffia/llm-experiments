from typing import List
from langchain.schema.messages import HumanMessage, BaseMessage


def docs_as_messages(docs) -> List[BaseMessage]:
    """
    Format the documents into a string. otherwise the output would
    be a list of [Document(page_content=...),...]
    """
    msg = HumanMessage(
        content="\n\n".join(doc.page_content for doc in docs)
    )

    return [msg]

def format_docs(docs):
    """
    Format the documents into a string. otherwise the output would
    be a list of [Document(page_content=...),...]
    """
    return "\n\n".join(doc.page_content for doc in docs)
