import os

from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .abstract_vector_repository import AbstractVectorRepository


class QdrantVectorRepository(AbstractVectorRepository):
    def __init__(self, collection_name):
        self.__embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        self.__text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # chunk size (characters)
            chunk_overlap=200,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )

        self.__client = QdrantClient(
            os.environ["QDRANT_URL"] if "QDRANT_URL" in os.environ else ":memory:"
        )

        self.__vector_size = len(self.__embeddings.embed_query("sample text"))

        self.__ensure_collection(collection_name)
        self.__vector_store = QdrantVectorStore(
            client=self.__client,
            collection_name=collection_name,
            embedding=self.__embeddings,
        )

    def __ensure_collection(self, collection_name):
        """Ensures that a collection exists in the vector db.

        :param collection_name: The name of the collection
        """
        if not self.__client.collection_exists(collection_name):
            self.__client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.__vector_size, distance=Distance.COSINE
                ),
            )

    def add_documents(self, documents: list[Document]):
        # TODO: Implement TextHandler class
        all_splits = self.__text_splitter.split_documents(documents)
        print(f"Split blog post into {len(all_splits)} sub-documents.")
        document_ids: list[str] = self.__vector_store.add_documents(
            documents=all_splits
        )
        return document_ids

    def similarity_search(self, query: str, k: int):
        return self.__vector_store.similarity_search(query, k)
