import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from prompts import system_prompt, human_template
from utils import ensure_fit_tokens, get_page_contents
from render import bot_msg_container_html_template, user_msg_container_html_template
import openai

load_dotenv()

embedding = OpenAIEmbeddings()
vec_db = Chroma(persist_directory="output/test_db", embedding_function=embedding)
retriever = vec_db.as_retriever(search_type="mmr")
llm = OpenAI()
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=retriever
)

def construct_messages(history):
    messages = [{"role": "system", "content": system_prompt}]

    for entry in history:
        role = "user" if entry["is_user"] else "assistant"
        messages.append({"role": role, "content": entry["message"]})
    
    messages = ensure_fit_tokens(messages)
    return messages

def query_handler(query):
    relevant_docs = compression_retriever.get_relevant_documents(query)
    context, metadata = get_page_contents(relevant_docs)
    query_with_context = human_template.format(query=query, context=context, metadata = metadata)
    return {"role": "user", "content": query_with_context}

def generate_response():
    if st.session_state.prompt == "":
        return
    st.session_state.history.append({
        "message": st.session_state.prompt, 
        "is_user": True
    })
    messages = construct_messages(st.session_state.history)
    messages.append(query_handler(st.session_state.prompt))
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=messages
    )
    assistant_message = response["choices"][0]["message"]["content"]
    st.session_state.history.append({
        "message": assistant_message, 
        "is_user": False
    })

def main():
        
    if "history" not in st.session_state:
        st.session_state.history = []
    
    st.text_input(
        "Consulta al asistente virtual:", 
        key="prompt", 
        placeholder="ej.: ¿Existen tesis sobre desigualdad de ingresos?", 
        on_change=generate_response
    )

    for message in st.session_state.history:
        if message["is_user"]:
            st.write(user_msg_container_html_template.replace("$MSG", message["message"]), unsafe_allow_html=True)
        else:
            st.write(bot_msg_container_html_template.replace("$MSG", message["message"]), unsafe_allow_html=True)

if __name__ == "__main__":
    main()