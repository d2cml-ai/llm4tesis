from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from prompts import system_prompt, human_template
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings()
vec_db = Chroma(persist_directory="output/test_db", embedding_function=embedding)
retriever = vec_db.as_retriever(search_type="mmr") # search_kwargs = {}

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

def ensure_fit_tokens(messages):
    """
    Ensure that total tokens in messages is less than MAX_TOKENS.
    If not, remove oldest messages until it fits.
    """
    total_tokens = num_tokens_from_messages(messages)
    while total_tokens > 4096:
        messages.pop(0)
        total_tokens = num_tokens_from_messages(messages)
    return messages

def get_page_contents(docs):
    contents = ""
    metadata = ""
    for i, doc in enumerate(docs, 1):
        contents += f"Document #{i}:\n{doc.page_content}\n\n"
        metadata += f"Document #{i}\n{doc.metadata}\n\n"
    return contents, metadata


def construct_messages(history):
    messages = [{"role": "system", "content": system_prompt}]

    for entry in history:
        role = "user" if entry["is_user"] else "assistant"
        messages.append({"role": role, "content": entry["message"]})
    
    messages = ensure_fit_tokens(messages)
    return messages

def query_handler(query):
    relevant_docs = retriever.get_relevant_documents(query)
    context, metadata = get_page_contents(relevant_docs)
    query_with_context = human_template.format(query=query, context=context, metadata = metadata)
    return {"role": "user", "content": query_with_context}