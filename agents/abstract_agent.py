from abc import ABC, abstractmethod
from collections.abc import Callable
from langchain_core.language_models import BaseChatModel
from common.schemas import AgentState

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

    @abstractmethod
    def act(self, state: AgentState, **kwargs):
        """The action of the agent"""
        raise NotImplementedError()
