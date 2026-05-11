"""
Natural Language Inference (NLI) Agent:

This agent receives both the output from Initial cosine similarity-based
RAG and the initial user's context. The agent then check how the
metadata of the candidate items aligns with the user context and rate
it. Then, the agent filter only candidate items that has the alignment
score above the predefined threshold.
"""

from langchain_core.prompts import PromptTemplate
from .abstract_agent import AbstractAgent
import logging
from .states import SelectedMenuList

logger = logging.getLogger(__name__)


class NLIAgent(AbstractAgent):
    def __init__(self, model, **kwargs):
        self._llm = model.with_structured_output(SelectedMenuList)
        self.__threshold = 25
        self.__min_score = 0
        self.__max_score = 100

    @property
    def prompt(self):
        return PromptTemplate.from_template(
            "You are a helpful assistant who is an expert in food and culinary recommendations.\n\n"
            "Your job is to evaluate menu items retrieved by by your assistant freind"
            "Your job is to rate those menu from the scores of {min_score} to {max_score}. "
            "Use your artificial brain to rate it."
            "Select only items that has scored greater or equal to {threshold}"
            "Provide score,"
            "Also provide justification of why you selected it.\n\n"
            "User's context: {context}"
            "Retrieved menu items: {menu_list}"
        )

    def act(self, state, **kwargs):
        chain = self.prompt | self._llm

        result: SelectedMenuList = chain.invoke(
            {
                "min_score": self.__min_score,
                "max_score": self.__max_score,
                "threshold": self.__threshold,
                "context": state.messages,  # latest user message
                "menu_list": state.menu_list,
            }
        )

        logger.info(result)

        return {"selected_menu_list": result}
