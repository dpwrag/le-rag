"""Agent states"""

import operator
from typing import Annotated

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


class SelectedMenu(BaseModel):
    name: str = Field(description="The name of the menu")
    description: str = Field(description="The description of the menu")
    score: float = Field(description="The score you gave to this particular menu")
    justification: str = Field(description="Justification of why you picked this menu.")


class SelectedMenuList(BaseModel):
    menu_list: list[SelectedMenu] = Field(description="A list of retrieved menus")


class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], operator.add] = Field(default_factory=list)
    menu_list: list[Document] = Field(
        default_factory=list, description="Raw retrieved documents from vector store"
    )
    selected_menu_list: SelectedMenuList | None = Field(
        default=None, description="Filtered and scored menus from NLI agent"
    )
    response: str | None = Field(default=None)
