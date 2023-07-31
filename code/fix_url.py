import requests
from bs4 import BeautifulSoup
import warnings
import os
from dotenv import load_dotenv
import pinecone

load_dotenv()

doc_names = os.listdir("raw")
query_template = "https://tesis.pucp.edu.pe/repositorio/handle/20.500.12404/1026/discover?query={doc_name}"
index_name = "llm4tesis"
pinecone.init()
index = pinecone.Index(index_name)

def create_pdf_link_dictionary(file_name):
    response = requests.get(
        query_template.format(doc_name=file_name), 
        verify=False
    )
    soup = BeautifulSoup(response.content, "html.parser")
    header = soup.find("div", class_="col-sm-9 artifact-description")
    link = "https://tesis.pucp.edu.pe" + header.find("a")["href"]
    return {file_name: link}

def get_matching_vectors_ids(key):
    matches = index.query(
        vector=[0] * 1536, # embedding dimension
        namespace='',
        filter={"source": f'/kaggle/input/llm4tesis/raw/{key}'},
        top_k=500,
        include_metadata=True
    )["matches"]
    ids = [match["id"] for match in matches]
    return ids

def modify_vector_metadata(key, value):
    ids = get_matching_vectors_ids(key)
    for id in ids:
        index.update(
            id=id, 
            set_metadata={"url": value}
        )

def main():
    warnings.filterwarnings("ignore")
    pdf_link_dictionary = {}
    i = 0
    print(f"Fetching links for all {len(doc_names)} files. This might take a while")
    print(" ----+---- ----+---- ----+---- ----+---- ----+---- ")
    for name in doc_names:
        print(".", end="")

        if (i + 1) % 50 == 0:
            print("    ", i + 1, sep="")
        
        pdf_link_dictionary.update(
            create_pdf_link_dictionary(name)
        )
        i += 1
    
    i = 0
    print(f"Updating links for all {len(pdf_link_dictionary)} files. This might take a while")
    print(" ----+---- ----+---- ----+---- ----+---- ----+---- ")
    
    for key, value in pdf_link_dictionary.items():
        print(".", end="")

        if (i + 1) % 50 == 0:
            print("    ", i + 1, sep="")
        
        modify_vector_metadata(key, value)
        i += 1

if __name__ == "__main__":
    main()
