import operator
from typing import Annotated, List

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class SelectedMenu(BaseModel):
    name: str = Field(description="The name of the menu")
    description: str = Field(description="The description of the menu")
    score: float = Field(description="The score you gave to this particular menu")
    justification: str = Field(description="Justification of why you picked this menu.")


class SelectedMenuList(BaseModel):
    menu_list: list[SelectedMenu] = Field(description="A list of retrieved menus")


class QueryRequest(BaseModel):
    """Request model for processing a query through the agent graph."""

    query: str


class MessageContent(BaseModel):
    """Message content in the response."""

    role: str
    content: str


class QueryResponse(BaseModel):
    """Response model for query processing."""

    query: str
    messages: List[dict]
    chain_of_thought: List[str]
    selected_menu: List[SelectedMenu] = []
    thread_id: str = "session-1"


class AgentState(BaseModel):
    messages: Annotated[list[BaseMessage], operator.add] = Field(default_factory=list)
    menu_list: list[Document] = Field(
        default_factory=list, description="Raw retrieved documents from vector store"
    )
    selected_menu_list: SelectedMenuList | None = Field(
        default=None, description="Filtered and scored menus from NLI agent"
    )
    chain_of_thought: Annotated[list[str], operator.add] | None = Field(
        default_factory=list
    )
    response: str | None = Field(default=None)
