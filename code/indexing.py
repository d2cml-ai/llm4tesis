from dotenv import load_dotenv
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import pandas as pd
import os

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
def count_tokens(text):
    return len(tokenizer.encode(text))

def main():
    load_dotenv()
    embeddings = OpenAIEmbeddings()
    df = pd.read_csv("output/tesis_data.csv", encoding="latin1").head(5)
    keys = ["title", "author", "advisor", "date_created", "abstract", "subject", "date_created"]
    tesis_list = [dict(zip(keys, row[keys]))
        for index, row in df.iterrows()
    ]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, 
        chunk_overlap=24, 
        length_function=count_tokens
    )
    tesis = tesis_list
    abstracts = [item["abstract"] for item in tesis]
    metadatas = [{"title": item["title"], "author": item["author"], "year": item["date_created"], "subject": item["subject"]} for item in tesis]
    
    chunks = text_splitter.create_documents(
        abstracts, 
        metadatas=metadatas
    )
    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="output/test_db")
    db.persist()

if __name__ == "__main__":
    main()