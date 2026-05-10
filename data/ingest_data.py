"""
Initial data ingestion

Run with `uv run --env-file .env python -m data.ingest_data` at the root dir.
"""

import pandas as pd
from infrastructure import get_qdrant_client, init_collection
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from typing import Callable, Tuple
import uuid
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


def dataframe_to_documents(
    dataframe: pd.DataFrame, page_content_func: Callable, metadata_func: Callable
) -> Tuple[list[str], list[Document]]:
    records = dataframe.to_dict("records")
    documents = []
    document_ids = []
    for row in records:
        new_doc = Document(
            page_content=page_content_func(row),
            metadata=metadata_func(row),
        )
        documents.append(new_doc)
        document_ids.append(uuid.uuid4())
    return document_ids, documents


def menu_metadata(row: dict) -> dict:
    return {
        "menu_item": row["menuItemName"],
        "restaurant": row["restaurantName"],
        "city": row["market"],
        "price": row["price"],
        "lat": row["restaurantLatitude"],
        "long": row["restaurantLongitude"],
    }


def menu_page_content(row: dict) -> str:
    return row["menuItemName"] + ": " + row["menuItemDescription"]


if __name__ == "__main__":
    df = pd.read_csv("data/restaurant_menu_preprocessed.csv")
    df = df.fillna("")
    client = get_qdrant_client()
    encoder = FastEmbedEmbeddings()
    collection_name = "menu"
    init_collection(
        collection_name, vector_size=len(encoder.embed_query("test")), client=client
    )

    vectorstore = QdrantVectorStore(
        client=client, collection_name=collection_name, embedding=encoder
    )

    document_ids, documents = dataframe_to_documents(
        df, menu_page_content, menu_metadata
    )
    vectorstore.add_documents(documents, ids=document_ids)
