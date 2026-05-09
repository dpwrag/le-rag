import getpass
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import FastEmbedEmbeddings
from infrastructure import get_qdrant_client
from langchain_qdrant import QdrantVectorStore
from agents import RetrieveContextAgent

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")


model = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
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

agent = RetrieveContextAgent.build(model, vector_store=vector_store)

query = "I'm a muslim and I want healthy diet. What menu should I eat?"


for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
