import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

def extract_metadata(text):
    """Extract metadata from the first page of the thesis.
    Returns a string with a short string with information about the document
    """
    while "  " in text:
        text = text.replace("  ", " ")
    text = text.replace(" \n", "\n").replace("\n", ". ").replace(" , ", ", ")

    while text[-1] == " ":
        text = text[:-1]
    text = "Información sobre el siguiente documento: " + text
    return text

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)

#         for index, page in enumerate(pdf_reader.pages):
#             page_text = page.extract_text()

#             if index == 0:
#                 page_text = extract_metadata(page_text)
#             text += page_text
#         text += "\nFIN DE LA TESIS\n"
#     return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(
        texts = text_chunks, 
        embedding = embeddings
    )
    return vector_store

def pdf_to_string(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for index, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if index == 0:
            page_text = extract_metadata(page_text)
        text += page_text
    return text

def vector_store_from_pdfs(pdf_docs):
    for i, pdf in enumerate(pdf_docs):
        text = pdf_to_string(pdf)
        text += "\nFIN DEL DOCUMENTO\n"
        chunks = get_text_chunks(text)
        if i == 0:
            vector_store = get_vector_store(chunks)
        else:
            vector_store.add_texts(chunks)
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

def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    
    # print("Hello World!")

    # must set up .env with API keys
    load_dotenv()
    st.set_page_config(page_title="Herramienta de Tesis", 
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Obten informacion sobre las tesis publicas de la especialidad de economia")
    user_question = st.text_input("Pregunta sobre las tesis de la especialidad:")
    if user_question:
        handle_user_input(user_question)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.write(user_template.replace("{{MSG}}", "Saludos, Chatbot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Saludos, Tesista"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("documentos")
        pdf_docs = st.file_uploader("sube pdf", accept_multiple_files=True)
        
        if st.button("process"):
            with st.spinner("Procesando"):
                vector_store = vector_store_from_pdfs(pdf_docs)
                
                st.session_state.conversation = get_conversation_chain(vector_store)
    

if __name__ == '__main__':
    main()