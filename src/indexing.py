from dotenv import load_dotenv
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import os

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def extract_metadata(doc):
    cover = doc[0]
    text = cover.page_content

    while "  " in text:
        text = text.replace("  ", " ")
    text = text.replace(" \n", "\n")
    text = text.replace("\n", ". ")
    text = text.replace(" , ", ", ")
    text = text.replace("1 PONTIFICIA", "PONTIFICIA")

    while text[-1] == " ":
        text = text[:-1]
    
    return text

def add_info(doc):
    filename = doc[0].metadata["source"].split("raw/")[1]
    url = f"https://tesis.pucp.edu.pe/repositorio/bitstream/handle/20.500.12404/24882/{filename}?sequence=1&isAllowed=y"
    info = extract_metadata(doc)
    
    for page in doc:
        page.metadata.update({"info": info, "url": url})
    
    return doc

def count_tokens(text):
    return len(tokenizer.encode(text))

def main():
    load_dotenv()
    index_name = "llm4tesis"
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, 
        chunk_overlap=24, 
        length_function=count_tokens, 
        separators=["\n\n", "\n", " ", ""]
    )
    paths = [os.path.join("raw", file) for file in os.listdir("raw")]    
    loaders = [PyPDFLoader(path) for path in paths]
    docs = [loader.load() for loader in loaders]
    chunks = []

    for doc in docs:
        doc = add_info(doc)
        chunks += text_splitter.split_documents(doc)
    
    pinecone.init()
    
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name = index_name, 
            metric = "cosine", 
            dimension = 1536
        )
    
    db = Pinecone.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        index_name=index_name
    )

if __name__ == "__main__":
    main()