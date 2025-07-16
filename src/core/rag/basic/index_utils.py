# src/core/rag/basic/index_utils.py

from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import faiss

def batch_embed_documents(embedder: HuggingFaceEmbeddings, documents: List[Document], batch_size: int = 128) -> List[np.ndarray]:
    embeddings = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        texts = [doc.page_content for doc in batch]
        emb = embedder.embed_documents(texts)
        embeddings.extend(emb)
    return embeddings

def build_faiss_index(embedder: HuggingFaceEmbeddings, documents: List[Document], batch_size: int = 128):
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import _build_index

    embeddings = batch_embed_documents(embedder, documents, batch_size=batch_size)
    index = _build_index(embeddings)
    store = FAISS(embedding_function=embedder, index=index, docstore=documents)
    return store