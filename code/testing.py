import openai
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
import pinecone
import time

from responseGeneration import *

openai.api_key="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"

def generate_response(query, history, relevant_docs):
    if query == "":
        return
    history.append({
        "message": query, 
        "is_user": True
    })
    messages = construct_messages(history)
    messages.append(query_handler(query, relevant_docs))
    response = openai.ChatCompletion.create(
        temperature=.6,
        model="gpt-3.5-turbo", 
        messages=messages
    )
    assistant_message = response["choices"][0]["message"]["content"]
    history.append({
        "message": assistant_message, 
        "is_user": False
    })
    return assistant_message, history

def main():
    load_dotenv()
    history = []
    pinecone.init()
    embedding = OpenAIEmbeddings()
    vector_database = Pinecone.from_existing_index(index_name="llm4tesis", embedding=embedding)
    retriever = vector_database.as_retriever(search_type="mmr")
    while True:
        query = input("Escriba su consulta: ")
        relevant_docs = retriever.get_relevant_documents(query)
        message, history = generate_response(query, history, relevant_docs)
        print(message)

if __name__ == "__main__":
    main()