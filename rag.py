import os
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from ESeimasHtmlLoader import ESeimasHtmlLoader
from pprint import pprint
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_embedding_model():
    return OpenAIEmbeddings(model="text-embedding-3-large")

def prefill_rag(urls, db_name):
    openai_api_key = st.secrets["OPENAI_API_KEY"]    
    os.environ["OPENAI_API_KEY"] = openai_api_key

    embeddings = get_embedding_model()
    chunks = retrieve_chunks(urls)

    with open("result.txt", "w", encoding="utf-8") as f:
        for doc in chunks:
            f.write(doc.page_content)
            f.write("\n" + "-"*28 + "\n")
            for key, value in doc.metadata.items():
                f.write(f"{key}: {value}\n")
            f.write("\n" + "-"*28 + "\n")

    persist_directory = f"./{db_name}"
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"ChromaDB stored at {persist_directory}")

def retrieve_chunks(urls):
    loader = ESeimasHtmlLoader()
    docs = []

    for url in urls:
        print(f"Loading document from {url}...")
        loaded_docs = loader.load(url)
        print(f" - Loaded {len(loaded_docs)} documents.")
        docs.extend(loaded_docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []

    for doc in docs:
        chunks = text_splitter.split_documents([doc])
        total = len(chunks)
        for idx, chunk in enumerate(chunks, 1):
            if not hasattr(chunk, "metadata") or chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["chunk_number"] = idx
            chunk.metadata["chunk_total"] = total
            chunk.metadata["is_last_chunk"] = (idx == total)
        all_chunks.extend(chunks)
    return all_chunks

def query_rag(query, db_name):
    openai_api_key = st.secrets["OPENAI_API_KEY"]    
    os.environ["OPENAI_API_KEY"] = openai_api_key

    embeddings = get_embedding_model()
    persist_directory = f"./{db_name}"

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    retrieved_docs = vector_store.similarity_search(query, k=5)
    return retrieved_docs
