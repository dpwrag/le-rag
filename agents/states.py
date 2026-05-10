"""Agent states"""

import operator
from typing import Annotated

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class RetrievedMenu(BaseModel):
    name: str = Field(description="The name of the menu")
    description: str = Field(description="The description of the menu")
    justification: str = Field(description="Justification of why you picked this menu.")


class RetrievedMenuList(BaseModel):
    menu_list: list[RetrievedMenu] = Field(description="A list of retrieved menus")


class AgentState(BaseModel):
    # Use Annotated and operator.add so messages append instead of overwrite!
    messages: Annotated[list[BaseMessage], operator.add] = Field(
        default_factory=list, description="Message from Human"
    )

    retrieved_menu_list: RetrievedMenuList | None = Field(
        default=None, description="The retrieved menus"
    )

    response: str | None = Field(
        default=None, description="The natural language response of the model"
    )
