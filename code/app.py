import streamlit as st
import pandas as pd, numpy as np

import os, pinecone, openai
import os, pinecone, openai
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI


load_dotenv(find_dotenv(), override=True)

pncone_api_key = os.environ.get("PINECONE_API_KEY")
pncone_env = os.environ.get("PINECONE_ENVIRONMENT")

index_name = 'llm4tesis'

pinecone.init(api_key=pncone_api_key, environment=pncone_env)
pinecone.whoami()
index = pinecone.Index(index_name)
openai.api_key = os.getenv("OPENAI_API_KEY")
embed_model = "text-embedding-ada-002"

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate

from joblib import Parallel, delayed

template = '''
Resume el texto de la tesis en maximo 40 palabras, simula que estas explicando el contenido a una persona
TEXTO: {text}
'''

def gen_chatAI(tem_plate = template, temperature = 0, model_name = 'gpt-4'):
	llm = ChatOpenAI(temperature=temperature, model_name=model_name)
	prompt = PromptTemplate(
		input_variables = ['text'],
		template = tem_plate
	)
	chain = load_summarize_chain(
		llm, chain_type='stuff', prompt = prompt, verbose=False
	)
	return chain



st.title("tesisMenthor")

# textos = ['mucho texto1', 'mucho texto2', 'mucho texto3']
# dict_metadata = [
# 	{'author': 'Herrera', 'anio': 1212, 'url': 'http://github.com', 'title': 'titulo de la tesis'},
# 	{'author': 'Rodriges', 'anio': 12, 'url': 'https://www.google.com', 'title': 'titulo de la tesis'},
# 	{'author': 'Quispe', 'anio': 12, 'url': 'www.google.com', 'title': 'titulo de la tesis'}
# ]



def answer_query(query, n_max = 6):
	embed = openai.Embedding.create(
		input=[query], engine = embed_model
	)
	embedding = embed['data'][0]['embedding']
	answer = index.query(embedding, top_k = 20, include_metadata=True)
	found = {}
	for i, matches in enumerate(answer['matches']):
		id_name = matches['id']
		score = matches['score']
		metadata = matches.metadata
		url = metadata['url']
		if url in list(found.keys()):
			found[url]['n_found'] += 1
			if found[url]['n_found'] < 4: 
				found[url]['context'] += metadata['text']
			if found[url]['score'] < score:
				found[url]['score'] = score
		else:
			found[url] = {
				'url': metadata['url'],
				'context': metadata['text'],
				'n_found': 1,
				'score': score
			}
	data_thesis = pd.DataFrame(found).T.sort_values(['n_found'], ascending=0).reset_index()
	return data_thesis.head(n_max)

# data_thesis = answer_query("crecimiento economico")
# chain = gen_chatAI()

# parallell from jo
from joblib import Parallel, delayed
def gen_response_options(data, cb):
	n, k = data.shape
	def gen_sum(row):
		docs = [Document(page_content=data.loc[row].context)]
		sum_row = cb.run(docs)
		print(row)
		return sum_row
	summary_arr = Parallel(n_jobs=-1)\
		(delayed(gen_sum)(i) for i in range(n))
	return summary_arr




def fetch_response(query):
	data_thesis = answer_query(query)
	print(data_thesis)
	chain = gen_chatAI()
	textos = gen_response_options(data_thesis, chain)
	urls = data_thesis.url.to_numpy()

	puro_texto = ''
	for i, texto in enumerate(textos):
		# metadata = dict_metadata[i]
		# author = metadata['author']
		# year = metadata['anio']
		# url = metadata['url']
		# title = metadata['title']
		# summary = f"{i+1}. [{author} ({year})]({url}): {texto} sobre {query}\n"
		summary = f"{i+1}. {texto}, [referencia]({urls[i]})\n"
		puro_texto += summary
	return puro_texto

import time

def generate_response():
	if st.session_state.prompt == '':
		return 
	start = time.time()
	puro_texto = fetch_response(st.session_state.prompt)
	time.sleep(.5)
	end = time.time()
	elapsed = end - start
	st.session_state.history.append({
			'prompt': st.session_state.prompt,
			'message': puro_texto,
			'time': elapsed
		})

def main():
	if "history" not in st.session_state:
		st.session_state.history = []

	for message in st.session_state.history:

		respuesta = message['message']
		prompt = message['prompt']
		tt = message['time']

		st.write(f"**{prompt}**\n")
		st.markdown(respuesta)
		st.markdown(f"""<div style="display: flex;justify-content: flex-end "><code>{tt:.2f}</code></div>""", unsafe_allow_html=True)
		st.markdown("""<hr style="height:0.5px;border:none; margin:0; padding:0 2px;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

	st.text_input(
			"Consulta al asistente virtual:", 
			key="prompt", 
			placeholder="ej.: ¿Existen tesis sobre desigualdad de ingresos?", 
			on_change=generate_response
	)

if __name__ == "__main__":
    main()