from abc import abstractmethod, ABC
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


class AbstractAgent(ABC):
    """
    Abstract class of each RAG Agent
    TODO: Finish this after researching LangGraph and stuffs
    """

    def __init__(self, model: ChatGoogleGenerativeAI):
        self.__model = model
        self.__prompt_text: str

    @abstractmethod
    def answer(query: str):
        pass
