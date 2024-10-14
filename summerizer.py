import streamlit as st
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import dotenv
dotenv.load_dotenv()

st.title("Summerizer")


def get_text():
    input_text = st.text_input("Type in the url link below", key="input")
    return input_text


user_input = get_text()

if (user_input):
    loader = WebBaseLoader(user_input)
    data = loader.load()
# less than 4096(total maximum token size)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    vectorstore = Chroma.from_documents(
        documents=all_splits, embedding=OpenAIEmbeddings())

    # question = "Summerize the main points of this article."
    # docs = vectorstore.similarity_search(question)

    # retrivalqa를 사용하는 이유: llm chain 구조에 대해 내 프롬프트에 대한 답변
    question = "Summarize the main points of this proposal"
    docs = vectorstore.similarity_search(question)
    len(docs)

    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum. Return the result as a list. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    prompts = hub.pull("langchain-ai/retrieval-qa-chat")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    combine_docs_chain = create_stuff_documents_chain(llm, prompts)
    rag_chain = create_retrieval_chain(
        vectorstore.as_retriever(), combine_docs_chain)

    # qa_chain = RetrievalQAWithSourcesChain.from_llm(
    #     llm,
    #     retriever=vectorstore.as_retriever(),
    #     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    # )

if (user_input):
    # result = qa_chain({"query": question})
    result = rag_chain.invoke(
        {"input": "Summarize the main points of this proposal"})
    result["answer"]
