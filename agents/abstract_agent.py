from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel


class AbstractAgent(ABC):
    @classmethod
    @abstractmethod
    def _tools(cls, **kwargs) -> list[Callable]:
        """Initialise tools for the agent (if any)"""
        return []

    @classmethod
    @abstractmethod
    def _prompt(cls) -> str:
        """Returns the prompt of the agent"""
        return ""

    @classmethod
    def build(cls, model: BaseChatModel, **kwargs) -> Any:
        return create_agent(
            model=model,
            tools=cls._tools(**kwargs),
            system_prompt=cls._prompt(),
            name=cls.__name__,
        )
