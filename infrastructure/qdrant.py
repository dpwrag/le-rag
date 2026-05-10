import os
from qdrant_client import QdrantClient, models


_client = QdrantClient(os.environ.get("QDRANT_URL", ":memory:"))


def get_qdrant_client() -> QdrantClient:
    return _client


def init_collection(
    collection_name: str, vector_size: int, client: QdrantClient = get_qdrant_client()
):
    """Initialise the collection"""
    if client.collection_exists(collection_name):
        return
    # size = self.__encoder.embedding_size
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
        # Low memory usage during upload (according to Qdrant docs)
        # Otherwise, I'm afraid that my meagre 8GB of Random Access Memory
        # Will explode.
        hnsw_config=models.HnswConfigDiff(m=0),
    )
