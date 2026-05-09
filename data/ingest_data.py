import pandas as pd
from infrastructure import QdrantVectorRepository, get_qdrant_client
from langchain_huggingface import HuggingFaceEmbeddings

df = pd.read_csv("data/restaurant_menu_preprocessed.csv")
df = df.fillna("")
QdrantVectorRepository(
    client=get_qdrant_client(),
    encoder=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
).upsert_from_dataframe(dataframe=df)
