from langchain.agents import create_agent
from collections.abc import Callable
from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    def __init__(self, model, **kwargs):
        self.__prompt = self._init_prompt()
        self.__tools = self._init_tools(**kwargs)
        self.__agent = create_agent(
            model=model,
            tools=self.__tools,
            system_prompt=self.__prompt,
        )

    @abstractmethod
    def _init_tools(self, **kwargs) -> list[Callable]:
        """Initialise tools for the agent (if any)"""
        return []

    @abstractmethod
    def _init_prompt(self) -> str:
        """Initialise the prompt"""
        return ""

    @property
    def agent(self):
        return self.__agent
