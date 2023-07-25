import openai
import streamlit as st
from dotenv import load_dotenv
import os

from render import bot_msg_container_html_template, user_msg_container_html_template
from responseGeneration import *

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

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
        messages=messages, 
        temperature=0.6
    )
    assistant_message = response["choices"][0]["message"]["content"]
    st.session_state.history.append({
        "message": assistant_message, 
        "is_user": False
    })

def main():
        
    if "history" not in st.session_state:
        st.session_state.history = []
    
    for message in st.session_state.history:
        if message["is_user"]:
            st.write(user_msg_container_html_template.replace("$MSG", message["message"]), unsafe_allow_html=True)
        else:
            st.write(bot_msg_container_html_template.replace("$MSG", message["message"]), unsafe_allow_html=True)
    
    st.text_input(
        "Consulta al asistente virtual:", 
        key="prompt", 
        placeholder="ej.: ¿Existen tesis sobre desigualdad de ingresos?", 
        on_change=generate_response
    )

if __name__ == "__main__":
    main()