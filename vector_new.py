from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

db_location = "./chrome_langchain_db"

class RetrieverContainer():

    def __init__(self):
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")

        pass

    def read_csv(self, csv_file):
        df = pd.read_csv("realistic_restaurant_reviews.csv")
        documents = []
        ids = []

        for i, row in df.iterrows():
            document = Document(
                page_content=row["Title"] + " " + row["Review"],
                metadata={"rating": row["Rating"], "date": row["Date"]},
                id=str(i)
            )
            ids.append(str(i))
            documents.append(document)

    add_documents = not os.path.exists(db_location)

    if add_documents:
        self.read_csv(csv_file)

    vector_store = Chroma(
        collection_name="restaurant_reviews",
        persist_directory=db_location,
        embedding_function=embeddings
    )

    if add_documents:
        vector_store.add_documents(documents=documents, ids=ids)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )
pass