from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
import os

os.environ["OPENAI_API_KEY"]="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"

df = pd.read_csv("output/table.csv", encoding="latin1")
chat = ChatOpenAI(temperature=0, model_name = "gpt-4")
agent = create_pandas_dataframe_agent(chat, df)