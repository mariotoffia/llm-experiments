from abc import ABC, abstractmethod
from embeddingsdb import EmbeddingsDb
from typing import TypeVar
from langchain.schema.runnable import Runnable
from langchain.chat_models import ChatOpenAI

T = TypeVar('T', bound='BaseChain')

class ChainNotCreatedError(Exception):
    """Raised when trying to access a chain that has not been created yet."""
    pass

class BaseChain(ABC):
    name: str
    model: ChatOpenAI
    embeddings_db: EmbeddingsDb
    debug: bool
    current_chain: Runnable

    def __init__(self, name: str,
                 model: ChatOpenAI,
                 embeddings_db: EmbeddingsDb,
                 debug: bool, **kwargs: any):
        super().__init__(**kwargs)

        self.name = name
        self.model = model
        self.embeddings_db = embeddings_db
        self.debug = debug

    @abstractmethod
    def create(self: type[T],
               model: ChatOpenAI = None,
               embeddings_db: EmbeddingsDb = None,
               debug: bool = None
               ) -> T:
        """
        Re/Creates a chain from the given parameters or defaults set
        in the constructor. It returns itself (subclass of BaseChain)
        :param model: The model. If omitted, the default model is used.
        :param embeddings_db: The embeddings database. If omitted, the default embeddings database is used.
        :param debug: The debug flag. If omitted, the default debug flag is used.
        :return: The itself (subclass of BaseChain) for fluent access.
        """

    def chain(self) -> Runnable:
        """
        Gets the created chain.
        :return: The runnable chain
        :raises ChainNotCreatedError: If the chain has not been created yet.
        """
        if self.current_chain is None:
            raise ChainNotCreatedError("Chain not created")
        
        return self.current_chain

    def __str__(self):
        return self.name

    def destroy(self):
        """
        The chain is no longer needed, this gives the chain a chance to clean up
        """
        pass

    def before(self, chain_message: any) -> any:
        """
        This method is called before the chain is executed
        :param chain_message: The message sent to the chain.
        :return: The chain message (possibly modified)
        """
        return chain_message

    def after(self, chain_message: any):
        """
        This method is called after the chain is executed
        :param chain_message: The message returned from the chain.
        """
        pass

    def chunk(self, chunk: any) -> any:
        """
        This method is called for each chunk emitted from the chain
        :param chunk: The chunk
        :return: The chunk (possibly modified)
        """
        return chunk

    def get_output(self, chunk: any) -> str:
        """
        This method is called for each chunk emitted from the chain
        :param chunk: The chunk to be converted to a output string
        :return: The string that shall be presented. If empty string 
        is returned, the chunk will be ignored
        """
        return chunk
