import logging

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents import NLIAgent, RetrieveContextAgent
from agents.states import AgentState
from infrastructure import get_qdrant_client

logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.DEBUG)

LLM_MODEL = "hf.co/unsloth/Qwen3-1.7B-GGUF:Q4_K_M "
model = ChatOllama(
    model=LLM_MODEL,
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


client = get_qdrant_client()
encoder = FastEmbedEmbeddings()
collection_name = "menu"

vector_store = QdrantVectorStore(
    client=client, collection_name=collection_name, embedding=encoder
)

query = "I'm a muslim and I want healthy diet. What menu should I eat?"
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
for event in graph.stream(
    {"messages": [HumanMessage(content=query)]},
    stream_mode="debug",  # emits {node_name: state_delta} instead
    config={"configurable": {"thread_id": "session-1"}},
):
    print(event)

# final_state = graph.invoke(
#     {"messages": [{"role": "user", "content": query}]},
#     config={"configurable": {"thread_id": "session-1"}},
# )

# final_message = final_state["messages"][-1]

# print("\n--- [Final Result] ---")
# raw_content = final_message.content

# if isinstance(raw_content, list):
#     for block in raw_content:
#         if block.get("type") == "text":
#             final_text = block["text"]
#             break
# else:
#     final_text = raw_content

# print(final_text)
