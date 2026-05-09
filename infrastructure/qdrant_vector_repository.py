from qdrant_client import QdrantClient, models
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import uuid
import pandas as pd

DEFAULT_COLLECTION_NAME = "restaurant_rag"


class QdrantVectorRepository:
    def __init__(
        self,
        client: QdrantClient,
        encoder: HuggingFaceEmbeddings,
        collection_name: str | None = None,
    ):
        self.__client = client
        self.__encoder = encoder
        self.__collection_name = collection_name or DEFAULT_COLLECTION_NAME
        self.init_collection()
        self.__menu_vector_store = QdrantVectorStore(
            client=self.__client,
            collection_name=self.__collection_name,
            embedding=self.__encoder,
            vector_name="menu",
        )
        self.__restaurant_vector_store = QdrantVectorStore(
            client=self.__client,
            collection_name=self.__collection_name,
            embedding=self.__encoder,
            vector_name="restaurant",
        )

    def init_collection(self):
        """Initialise the collection"""
        if self.__client.collection_exists(self.__collection_name):
            return
        size = len(self.__encoder.embed_query("sample text"))
        self.__client.create_collection(
            collection_name=self.__collection_name,
            vectors_config={
                "menu": models.VectorParams(
                    size=size,
                    distance=models.Distance.COSINE,
                ),
                "restaurant": models.VectorParams(
                    size=size,
                    distance=models.Distance.COSINE,
                ),
            },
            # Low memory usage during upload (according to Qdrant docs)
            # Otherwise, I'm afraid that my meagre 8GB of Random Access Memory
            # Will explode.
            hnsw_config=models.HnswConfigDiff(m=0),
        )

    @property
    def menu_vector_store(self):
        return self.__menu_vector_store

    @property
    def restaurant_vector_store(self):
        return self.__restaurant_vector_store

    def upsert_from_dataframe(self, dataframe: pd.DataFrame, batch_size: int = 100):
        """Upsert from pandas dataframe using optimized batch encoding"""

        # 1. Prepare strings for batch encoding
        # This is much faster than row-by-row string formatting
        menu_texts = (
            dataframe["menuItemName"] + ": " + dataframe["menuItemDescription"]
        ).tolist()
        res_texts = (
            dataframe["restaurantName"] + ": " + dataframe["restaurantDescription"]
        ).tolist()

        # 2. Batch Encode (This leverages GPU/Parallelism)
        # HuggingFaceEmbeddings.embed_documents is designed for lists
        print(f"Encoding {len(menu_texts)} items...")
        menu_vectors = self.__encoder.embed_documents(menu_texts)
        res_vectors = self.__encoder.embed_documents(res_texts)

        # 3. Prepare Points
        points = []
        # Using to_dict('records') allows for fast iteration over pre-computed values
        records = dataframe.to_dict("records")

        for i, row in enumerate(records):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "menu": menu_vectors[i],
                        "restaurant": res_vectors[i],
                    },
                    payload={
                        "menu_item": row["menuItemName"],
                        "restaurant": row["restaurantName"],
                        "city": row["market"],
                        "price": row["price"],
                        "lat": row["restaurantLatitude"],
                        "long": row["restaurantLongitude"],
                    },
                )
            )

        # 4. Upload in chunks to avoid memory/timeout issues
        print(f"Uploading to Qdrant collection: {self.__collection_name}...")
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.__client.upsert(collection_name=self.__collection_name, points=batch)

        print("Upsert complete.")

    def upsert(self):
        # TODO: To be implemented (maybe refactor from upsert_from_dataframe)
        pass

    def update(self):
        # TODO: to be implemented
        pass

    def delete(self):
        # TODO: to be implemented
        pass
