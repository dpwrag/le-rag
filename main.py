import getpass
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from infrastructure import get_qdrant_client, QdrantVectorRepository
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

encoder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

repo = QdrantVectorRepository(
    client=get_qdrant_client(),
    encoder=encoder,
)

menu_vector_store = repo.menu_vector_store


agent = RetrieveContextAgent.build(model, vector_store=menu_vector_store)
query = "I want to eat something healthy and I'm a muslim so pork is out of a question. What should I eat?"

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
