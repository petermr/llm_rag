import logging

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
logger = logging.getLogger(__file__)

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about climate
Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

print(f"template {template}")
while True:
    print("\n\n-------------------------------")
    question = input("Please ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)
