import logging
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

logger = logging.getLogger(__file__)

# borrowed with thanks from TechWithTim tutorial

# option = "pizza"
option = "climate"
#option = "makespace"
if option == "pizza":
    csvin = "realistic_restaurant_reviews.csv"
    title = "Title"
    text = "Review"
    collection_name = "restaurant_reviews"
    k_val = 5

elif option == "climate":
    TEMP_DIR = Path(Path(__file__).parent.parent.parent, "amilib", "temp", "csv", "ipcc")
    logger.info(f"ipcc {TEMP_DIR}")
    csvin = str(Path(TEMP_DIR, "syr_paras.csv"))
    csvin = "syr_paras.csv"
    title = "title"
    text = "text"
    collection_name = "climate_chapters"
    k_val = 5

assert Path(csvin).exists(), f"no csvin {csvin}"
# logger.info(f"input from {csvin}")
df = pd.read_csv(csvin)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# db_location = "./chrome_langchain_db"
db_location = f"./chrome_{option}_db"
add_documents = not os.path.exists(db_location)

if add_documents or True:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        if option == "climate":
            metadata = {}
        elif option == "makespace":
            metadata = {}
        elif option == "pizza":
            metadata = {"rating": row["Rating"], "date": row["Date"]}
        document = Document(
            page_content=row[title] + " " + row[text],
            metadata=metadata,
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name=collection_name,
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": k_val}
)
