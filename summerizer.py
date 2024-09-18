import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import retrieval_qa
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dotenv
dotenv.load_dotenv()
