from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)
restaurant_documents = []


def add_restaurants():
    global ids, i, row
    ids = []
    for i, row in df.iterrows():
        restaurant_document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        restaurant_documents.append(restaurant_document)


if add_documents:
    add_restaurants()

restaurant_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)
vector_store_ipcc = Chroma(
    collection_name="syr_ipcc",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    restaurant_store.add_documents(documents=restaurant_documents, ids=ids)
    
retriever = restaurant_store.as_retriever(
    search_kwargs={"k": 5}
)