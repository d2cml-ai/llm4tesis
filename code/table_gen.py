from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import pandas as pd
import os

def get_metadata(text):
    """Extract metadata from the first page of the thesis.
    Returns a string with a short string with information about the document
    """
    while "  " in text:
        text = text.replace("  ", " ")
    text = text.replace(" \n", "\n").replace("\n", ". ").replace(" , ", ", ")

    while text[-1] == " ":
        text = text[:-1]
    text = "Información sobre el siguiente documento: " + text + "\n"
    return text

def doc_to_text(doc):
    text = ""
    for index, page in enumerate(doc):
        if index == 0:
            text += get_metadata(page.page_content)
        text += page.page_content
    return text

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000, 
        chunk_overlap=100, 
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(
        texts = chunks, 
        embedding = embeddings
    )
    return vector_store

def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(), 
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def main():
    load_dotenv()
    paths = [os.path.join("raw", file) for file in os.listdir("raw")[:5]] # indexes used so far: :5
    loaders = [PyPDFLoader(path) for path in paths]
    
    master = pd.read_csv("output/table.csv", encoding = "latin1")

    for loader in loaders:
        doc = loader.load()
        text = doc_to_text(doc)
        chunks = get_chunks(text)
        vector_store = get_vector_store(chunks)
        conversation = get_conversation_chain(vector_store)

        title = conversation({"question": "Cuál es el título del documento?"})["answer"]
        author = conversation({"question": "Quién es el autor del documento?"})["answer"]
        advisor = conversation({"question": "Quién es el asesor para documento?"})["answer"]
        summary = conversation({"question": "Puedes hacer un resumen corto del documento (alrededor de 200 palabras)"})["answer"]
        keywords = conversation({"question": "Dame una lista de 5 palabras clave del documento"})["answer"]

        temp = pd.DataFrame({"title": [title], "author": [author], "advisor": [advisor], "summary": [summary], "keywords": [keywords]})
        master = pd.concat([master, temp])
    master.to_csv("output/table.csv", encoding = "latin1")


if __name__ == "__main__":
    main()