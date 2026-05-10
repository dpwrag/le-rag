import getpass
import logging
import os

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import RetryPolicy

from agents import RetrieveContextAgent
from infrastructure import get_qdrant_client

logger = logging.getLogger(__name__)
logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.DEBUG)

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")


model = ChatGoogleGenerativeAI(
    model="gemma-4-26b-a4b-it",
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
retrieve_context_agent = RetrieveContextAgent.build(model, vector_store=vector_store)


def retrieve_context_node(state: MessagesState) -> MessagesState:
    logger.debug(msg=state["messages"])
    return retrieve_context_agent.invoke(state)


def french_translation_node(state: MessagesState) -> MessagesState:
    logger.debug(msg=state["messages"])
    text_to_translate = state["messages"][-1].content
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a professional translator."
                "Translate the user's text into French."
                "Output ONLY the translation.",
            ),
            ("user", "{text}"),
        ]
    )

    # 3. Simple Chain: Prompt -> Model
    chain = prompt | model
    response = chain.invoke({"text": text_to_translate})

    return {"messages": [response]}


# --- Build the graph ---
graph = (
    StateGraph(MessagesState)
    .add_node(
        "retrieve_context",
        retrieve_context_node,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=3.0),
    )
    .add_node(
        "french_translation",
        french_translation_node,
        retry_policy=RetryPolicy(max_attempts=3, initial_interval=3.0),
    )
    .add_edge(START, "retrieve_context")
    .add_edge("retrieve_context", "french_translation")
    .add_edge("french_translation", END)
    .compile(checkpointer=InMemorySaver())  # enables multi-turn memory
)

# Option B — print each node's final message with a label
# for event in graph.stream(
#     {"messages": [{"role": "user", "content": query}]},
#     stream_mode="updates",  # emits {node_name: state_delta} instead
#     config={"configurable": {"thread_id": "session-1"}},
# ):
#     node_name, state = next(iter(event.items()))
#     print(f"\n--- [{node_name}] ---")
#     state["messages"][-1].pretty_print()

final_state = graph.invoke(
    {"messages": [{"role": "user", "content": query}]},
    config={"configurable": {"thread_id": "session-1"}},
)

# Grab the very last message from the final state
final_message = final_state["messages"][-1]

print("\n--- [Final Result] ---")
raw_content = final_message.content

if isinstance(raw_content, list):
    for block in raw_content:
        if block.get("type") == "text":
            final_text = block["text"]
            break
else:
    final_text = raw_content

# 3. Printyour final, clean string!
print(final_text)
