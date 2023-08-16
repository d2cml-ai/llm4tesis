from dotenv import load_dotenv
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import os
import pandas as pd
import uuid
from ensureMatches import matchingPDFs, dataFilePath, pdfFilesPath

load_dotenv()
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
index_name = "llm4tesis"
embeddings = OpenAIEmbeddings()
metadataFields = [
	"advisor", 
	"author", 
	"year",
	"title",
	"url_thesis"
]

def updateMetadata(loadedDoc, metadata):
	metadataDict = {
		metadataField: metadata[metadataField].values[0] 
		for metadataField in metadataFields
	}

	for page in loadedDoc:
		page.metadata.update(metadataDict)
		
	return loadedDoc

def getDocWithMetadata(path, metadata):
	loadedDoc = PyPDFLoader(path).load()
	docWithMetadata = updateMetadata(loadedDoc, metadata)
	return docWithMetadata

def embedFromDocuments(docChunks):
	vectorsWithMetadata = []
	texts = [docChunk.page_content for docChunk in docChunks]
	metadatas = [docChunk.metadata for docChunk in docChunks]
	textEmbeddings = embeddings.embed_documents(texts)
	hashes = [str(uuid.uuid4()) for _ in texts]

	for i, (text, textEmbedding) in enumerate(zip(texts, textEmbeddings)):
		textMetadata = metadatas[i]
		textMetadata["text"] = text
		vectorsWithMetadata.append((hashes[i], textEmbedding, textMetadata))
	
	return vectorsWithMetadata

def count_tokens(text):
	return len(tokenizer.encode(text))

def main():
	pinecone.init()
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=512, 
		chunk_overlap=24, 
		length_function=count_tokens, 
		separators=["\n\n", "\n", " ", ""]
	)
	metadata = pd.read_json(f"{dataFilePath}/00_metadata/eco_tesis.json")
	paths = [os.path.join(pdfFilesPath, file) for file in matchingPDFs]

	if index_name not in pinecone.list_indexes():
		pinecone.create_index(
			name=index_name, 
			metric="cosine",
			dimension=1536
		)
	
	pineconeIndex = pinecone.Index(index_name)

	for path in paths:
		pathInMetadata = path.replace(dataFilePath, "dspace_home")
		docMetadata = metadata[metadata.pdf_file_local == pathInMetadata][metadataFields]
		docWithMetadata = getDocWithMetadata(path, docMetadata)
		docChunks = text_splitter.split_documents(docWithMetadata)
		docEmbeddings = embedFromDocuments(docChunks)
		pineconeIndex.upsert(
			vectors = docEmbeddings
		)

if __name__ == "__main__":
	main()