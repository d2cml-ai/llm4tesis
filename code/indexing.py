from dotenv import load_dotenv
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
from random import sample
from utils import add_info

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
def count_tokens(text):
    return len(tokenizer.encode(text))

def main():
    load_dotenv()
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, 
        chunk_overlap=24, 
        length_function=count_tokens
    )
    paths = [os.path.join("raw", file) for file in os.listdir("raw")]
    
    loaders = [PyPDFLoader(path) for path in paths]
    docs = [loader.load() for loader in loaders]
    chunks = []

    for doc in docs:
        doc = add_info(doc)
        chunks += text_splitter.split_documents(doc)
    
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="output/test_db"
    )
    db.persist()

if __name__ == "__main__":
    main()