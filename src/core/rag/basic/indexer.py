# src/core/rag/basic/indexer.py

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.core.utils.logger import get_logger
from src.core.data.product import Product

logger = get_logger(__name__)


def validate_product_data(data: dict) -> Optional[Product]:
    try:
        return Product(**data)
    except Exception as e:
        logger.warning(f"Producto inválido: {e}")
        return None


class Indexer:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vectorstore_type: str = "chroma",
        device: str = "cpu"
    ):
        self.embedder = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vectorstore_type = vectorstore_type.lower()
        if self.vectorstore_type not in ["chroma", "faiss"]:
            raise ValueError("Vectorstore must be 'chroma' or 'faiss'")

    def build_index_from_products(
        self,
        products: List[Union[Dict, Product]],
        persist_dir: Union[str, Path],
        batch_size: int = 1000
    ) -> None:
        persist_dir = Path(persist_dir)
        os.makedirs(persist_dir, exist_ok=True)

        documents = []
        for p in products:
            if isinstance(p, Product):
                documents.append(Document(**p.to_document()))
            elif isinstance(p, dict):
                validated = validate_product_data(p)
                if validated:
                    documents.append(Document(**validated.to_document()))

        logger.info(f"Indexando {len(documents)} documentos...")

        if self.vectorstore_type == "chroma":
            self._build_chroma_index(documents, persist_dir, batch_size)
        else:
            self._build_faiss_index(documents, persist_dir, batch_size)

    def _build_chroma_index(
        self,
        documents: List[Document],
        persist_dir: Path,
        batch_size: int
    ) -> None:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Batch embeddings to avoid memory issues
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings.extend(self.embedder.embed_documents(batch))

        Chroma.from_texts(
            texts=texts,
            embedding=self.embedder,
            metadatas=metadatas,
            persist_directory=str(persist_dir)
        )
        logger.info(f"Índice Chroma guardado en {persist_dir}")

    def _build_faiss_index(
        self,
        documents: List[Document],
        persist_dir: Path,
        batch_size: int
    ) -> None:
        texts = [doc.page_content for doc in documents]

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings.extend(self.embedder.embed_documents(batch))

        embeddings_np = np.array(embeddings).astype('float32')
        dim = embeddings_np.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(embeddings_np)
        faiss.write_index(index, str(persist_dir / "faiss_index.index"))

        self._save_faiss_documents(documents, persist_dir)
        logger.info(f"Índice FAISS guardado en {persist_dir}")

    def _save_faiss_documents(self, documents: List[Document], persist_dir: Path) -> None:
        docs_path = persist_dir / "faiss_documents.json"
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump([
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in documents
            ], f)

    @classmethod
    def from_product_jsonl(
        cls,
        jsonl_path: Union[str, Path],
        persist_dir: Union[str, Path],
        max_products: Optional[int] = None,
        **kwargs
    ) -> None:
        indexer = cls(**kwargs)

        products = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_products and i >= max_products:
                    break
                try:
                    product = json.loads(line)
                    validated = validate_product_data(product)
                    if validated:
                        products.append(validated)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error al leer línea {i}: {e}")
                    continue

        indexer.build_index_from_products(products, persist_dir)
