import os

from qdrant_client import QdrantClient


_client = QdrantClient(os.environ.get("QDRANT_URL", ":memory:"))


def get_qdrant_client() -> QdrantClient:
    return _client
