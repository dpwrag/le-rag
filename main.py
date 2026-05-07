import getpass
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")


model = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

client = QdrantClient(
    url=os.environ["QDRANT_URL"] if "QDRANT_URL" in os.environ else ":memory:"
)

vector_size = len(embeddings.embed_query("sample text"))

if not client.collection_exists("test"):
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
vector_store = QdrantVectorStore(
    client=client,
    collection_name="test",
    embedding=embeddings,
)

loader = TextLoader("bee.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")


document_ids = vector_store.add_documents(documents=all_splits)


def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


prompt_text = """
Role: You are a helpful assistanat who is an expert in bee movie. You shall answer the question based on the following context. If the context doesn't contain relevant information, or you're unsure, say that you don't know.

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(prompt_text)

rag_chain = (
    {
        # The chain executes the retrieval dynamically using the input query (x)
        # We use [0] because your retrieve_context returns (serialized, docs)
        "context": lambda x: retrieve_context(x)[0],
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

query = "Who is barry?"
response = rag_chain.invoke(query)

print(response)
