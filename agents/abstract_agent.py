from abc import ABC, abstractmethod
from collections.abc import Callable
from langchain_core.language_models import BaseChatModel
from .states import AgentState

from langchain_core.prompts import PromptTemplate


class AbstractAgent(ABC):
    def __init__(self, model: BaseChatModel, **kwargs):
        self._agent = None
        raise NotImplementedError()

    @property
    @abstractmethod
    def prompt(self) -> str | PromptTemplate:
        """
        The prompt of the agent

        :return: the agent's prompt.
        """
        raise NotImplementedError()

    def _tools(self, **kwargs) -> list[Callable]:
        """Initialise tools for the agent (if any)

        :return: list of tools
        """
        raise NotImplementedError()

    @property
    def agent(self):
        """Returns the inner agent (if any)"""
        raise NotImplementedError()

    @abstractmethod
    def act(self, state: AgentState, **kwargs):
        """The action of the agent"""
        raise NotImplementedError()
