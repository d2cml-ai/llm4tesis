from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from prompts import *
from dotenv import load_dotenv
import openai

load_dotenv()

embedding = OpenAIEmbeddings()
vec_db = Chroma(persist_directory="output/test_db", embedding_function=embedding)
retriever = vec_db.as_retriever(search_type="mmr", search_kwargs = {"k":8, "fetch_k":40})

import tiktoken

def message_token_count(message, num_tokens, model):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    for key, value in message.items():
        num_tokens += len(encoding.encode(value))

        if key == "name": num_tokens -= 1
        
    return num_tokens
    

def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    if model != "gpt-3.5-turbo":
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}. 
                                  See https://github.com/openai/openai-python/blob/main/chatml.md 
                                  for information on how messages are converted to tokens.""")
    
    num_tokens = 0

    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n        
        num_tokens = message_token_count(message, num_tokens, model)
    
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def ensure_fit_tokens(messages, max_tokens = 4096):
    """
    Ensure that total tokens in messages is less than MAX_TOKENS.
    If not, remove oldest messages until it fits.
    """
    total_tokens = num_tokens_from_messages(messages)

    while total_tokens > max_tokens:
        messages.pop(0)
        total_tokens = num_tokens_from_messages(messages)
    
    return messages

def get_page_contents(docs):
    contents = ""

    for i, doc in enumerate(docs, 1):
        info = doc['metadata']['info']
        url = doc["metadata"]["url"]
        summary = doc['summary']
        contents += f"Document #{i}:\nInfo: {info}\nLink: {url}\n{summary}\n\n"
    
    return contents


def construct_messages(history):
    messages = [{"role": "system", "content": system_prompt}]

    for entry in history:
        role = "user" if entry["is_user"] else "assistant"
        messages.append({"role": role, "content": entry["message"]})
    
    messages = ensure_fit_tokens(messages)
    return messages

def reduce(doc, query):
    reduce_query_content = reduce_query_template.format(
        query = query, 
        context = doc.page_content
    )
    messages = [
        {"role": "system", "content": reduce_system_prompt}, 
        {"role": "user", "content": reduce_query_content}
    ]
    summary = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    reduced_doc = {"summary": summary["choices"][0]["message"]["content"], "metadata": doc.metadata}
    return reduced_doc

def map_reduce_relevant_documents(query):
    relevant_docs = retriever.get_relevant_documents(query)
    reduced_docs = []

    for doc in relevant_docs:
        reduced_docs += [reduce(doc, query)]
    
    return reduced_docs

def query_handler(query):
    relevant_docs = map_reduce_relevant_documents(query)
    context = get_page_contents(relevant_docs)
    query_with_context = human_template.format(query=query, context=context)
    return {"role": "user", "content": query_with_context}