from pydantic import BaseModel
from typing import List


class QueryRequest(BaseModel):
    """Request model for processing a query through the agent graph."""
    query: str


class MessageContent(BaseModel):
    """Message content in the response."""
    role: str
    content: str


class SelectedMenu(BaseModel):
    """Model for a selected menu item with scoring details."""
    name: str
    description: str
    score: float
    justification: str


class QueryResponse(BaseModel):
    """Response model for query processing."""
    query: str
    messages: List[dict]
    selected_menu: List[SelectedMenu] = []
    thread_id: str = "session-1"
