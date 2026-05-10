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


class NLIAgent(AbstractAgent):
    def __init__(self, model, **kwargs):
        self._llm = model

    @property
    def prompt(self):
        return PromptTemplate.from_template(
            "You are a friendly restaurant assistant helping a customer choose what to eat.\n\n"
            "Based on the customer's request and the retrieved menu items below, "
            "present the recommendations in a warm, conversational tone.\n\n"
            "For each recommended item, briefly mention:\n"
            "- The dish name and a one-line description\n"
            "- Why it matches what the customer is looking for"
            "(i.e., the justification why it's chosen)"
            "- Price (if available)\n\n"
            "Keep the response concise (5 items max) and end with a light suggestion "
            "or question to help the customer decide.\n\n"
            "Customer request: {query}\n\n"
            "Retrieved menu items:\n{retrieved_menu_list}"
        )

    def act(self, state, **kwargs):
        chain = self.prompt | self._llm
        msg = chain.invoke(
            {
                "query": state.messages,
                "retrieved_menu_list": state.retrieved_menu_list,
            }
        )
        return {"response": msg}
