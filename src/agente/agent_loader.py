import os
import sys
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.prompts.product_prompt import PROMPT_TEMPLATE
from src.llms.local_llm import local_llm
def cargar_llm_local():
    llm = local_llm()
    if not llm:
        print(" Error al cargar el modelo LLM local.")
        sys.exit(1)
    return llm

def cargar_vectorstore(persist_directory: str):
    if not os.path.isdir(persist_directory):
        print(f"❌ Índice no encontrado en: {persist_directory}")
        sys.exit(1)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    ejemplos = vectordb.similarity_search("validación rápida", k=2)
    if ejemplos:
        print("✅ Vectorstore cargado correctamente.")
        print(ejemplos[0].page_content[:200], "...\n")
    else:
        print("⚠ El índice está vacío o mal cargado.")

    return vectordb

def cargar_agente(persist_directory: str):
    vectordb = cargar_vectorstore(persist_directory)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=5,
        return_messages=True
    )

    llm = local_llm()
    if not llm:
        print(" Error al cargar el modelo LLM local.")
        sys.exit(1)
    prompt = PromptTemplate(
        input_variables=["chat_history", "question", "context"],
        template=PROMPT_TEMPLATE
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False
    )