"""
Setup and initialization for the LE-RAG agent graph and models.
This module handles the creation of the model, vector store, and agent graph.
"""

import logging

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents import NLIAgent, RetrieveContextAgent
from agents.states import AgentState
from infrastructure import get_qdrant_client

logger = logging.getLogger(__name__)

LLM_MODEL = "hf.co/unsloth/Qwen3-1.7B-GGUF:Q4_K_M"


def create_graph() -> CompiledStateGraph:
    """
    Create and initialize the agent graph with model and vector store.

    Returns:
        A CompiledStateGraph
    """
    logger.info("Initializing LE-RAG components...")

    model = ChatOllama(
        model=LLM_MODEL,
        temperature=1.0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    logger.info(f"Initialized model: {LLM_MODEL}")

    client = get_qdrant_client()
    encoder = FastEmbedEmbeddings()
    collection_name = "menu"

    vector_store = QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=encoder
    )
    logger.info("Initialized vector store")

    retrieve_context_agent = RetrieveContextAgent(model, vector_store=vector_store)
    nli_agent = NLIAgent(model)

    graph = (
        StateGraph(AgentState)
        .add_node("retrieve_context", retrieve_context_agent.act)
        .add_node("nli_agent", nli_agent.act)
        .add_edge(START, "retrieve_context")
        .add_edge("retrieve_context", "nli_agent")
        .add_edge("nli_agent", END)
        .compile(checkpointer=InMemorySaver())
    )
    logger.info("Agent graph compiled successfully")

    return graph
