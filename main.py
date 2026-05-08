import getpass
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from vector_repository import QdrantVectorRepository
from agents import RetrieveContextAgent

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")


model = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


vector_repo = QdrantVectorRepository("test")

# loader = TextLoader("bee.txt")
# docs = loader.load()
# document_ids = vector_repo.add_documents(documents=docs)

agent = RetrieveContextAgent(model, vector_store=vector_repo).agent
query = "Who is barry?"

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
