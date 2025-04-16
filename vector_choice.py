from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd


df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

# replace by enum and constants
PIZZA = "pizza"
MS = "MS"
input_types = [PIZZA, MS]
input_type = PIZZA
nkwargs = 5

def make_document(i, row, input_type):
    if not input_type in input_types:
        raise ValueError(f"bad input {input_type}")
    document = None
    if input_type == PIZZA:
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
    elif input_type == MS:
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
    else:
        document = None
    return document



def get_documents_ids():
    documents = []
    ids = []
    for i, row in df.iterrows():
        document = make_document(i, row)
        ids.append(str(i))
        documents.append(document)
    return documents, ids

if add_documents:
    documents, ids = get_documents_ids()


def get_collection_name():
    name = "UNKNOWN"
    if input_type == PIZZA:
        name = "restaurant_reviews"
    elif input_type == MS:
        name = "makespace"
    return name


vector_store = Chroma(
    collection_name=get_collection_name(),
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    documents, ids = get_documents_ids()
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": nkwargs}
)