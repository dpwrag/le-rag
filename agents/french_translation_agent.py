from .abstract_agent import AbstractAgent


class FrenchTranslationAgent(AbstractAgent):
    @classmethod
    def _prompt(cls):
        return (
            "You are a French translation agent. "
            "Your only job is to translate messages into French. "
            "Do not answer questions. Do not use tools. Just translate."
        )

    @classmethod
    def _tools(cls, **kwargs):
        return []
