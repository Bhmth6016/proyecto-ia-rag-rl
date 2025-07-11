# src/core/rag/basic/retriever.py

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.base import VectorStoreRetriever

from src.core.utils.logger import get_logger
from src.core.data.product import Product

logger = get_logger(__name__)


class Retriever:
    def __init__(
        self,
        index_path: Union[str, Path],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vectorstore_type: str = "chroma",
        device: str = "cpu"
    ):
        self.index_path = Path(index_path)
        self.embedding_model = embedding_model
        self.vectorstore_type = vectorstore_type.lower()
        self.device = device
        self._load_index()

    def _load_index(self) -> None:
        try:
            self.embedder = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True}
            )

            if self.vectorstore_type == "chroma":
                self.vectorstore = Chroma(
                    persist_directory=str(self.index_path),
                    embedding_function=self.embedder
                )
                self.retriever = self.vectorstore.as_retriever()
            else:
                self.faiss_index = faiss.read_index(str(self.index_path / "faiss_index.index"))
                self._load_faiss_documents()

            logger.info(f"Loaded {self.vectorstore_type} index from {self.index_path}")
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            raise

    def retrieve_products(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None,
        score_threshold: Optional[float] = None
    ) -> List[Product]:
        docs = self.retrieve(query, k, filters, score_threshold)
        return [self._document_to_product(doc) for doc in docs]

    def _document_to_product(self, doc: Document) -> Product:
        metadata = doc.metadata
        content = doc.page_content

        # Extract specifications from content
        specs = {}
        if "Specs:" in content:
            specs_section = content.split("Specs:")[1].strip()
            for line in specs_section.split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    specs[key.strip()] = val.strip()

        product_data = {
            "id": metadata.get("id", ""),
            "title": metadata.get("title", ""),
            "main_category": metadata.get("category", ""),
            "categories": [metadata.get("category", "")],
            "price": metadata.get("price"),
            "average_rating": metadata.get("rating"),
            "rating_count": metadata.get("rating_count", 0),
            "images": None,
            "details": {
                "brand": metadata.get("brand"),
                "specifications": specs
            }
        }

        return Product(**product_data)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        try:
            if self.vectorstore_type == "chroma":
                search_kwargs = {"k": k}
                if filters:
                    search_kwargs["filter"] = self._build_chroma_filter(filters)
                if score_threshold:
                    search_kwargs["score_threshold"] = score_threshold
                return self.retriever.get_relevant_documents(query, **search_kwargs)
            else:
                return self._retrieve_faiss(query, k, filters)
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return []

    def _retrieve_faiss(self, query: str, k: int, filters: Optional[Dict]) -> List[Document]:
        query_embedding = np.array(self.embedder.embed_query(query)).astype('float32')
        distances, indices = self.faiss_index.search(np.array([query_embedding]), k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.faiss_documents):
                doc = self.faiss_documents[idx]
                doc.metadata["score"] = float(1 - score)
                results.append(doc)

        if filters:
            results = [doc for doc in results if self._matches_filters(doc.metadata, filters)]

        return sorted(results, key=lambda x: x.metadata["score"], reverse=True)[:k]

    def _load_faiss_documents(self) -> None:
        docs_path = self.index_path / "faiss_documents.json"
        try:
            with open(docs_path, "r", encoding="utf-8") as f:
                self.faiss_documents = [
                    Document(
                        page_content=item["content"],
                        metadata=item["metadata"]
                    ) for item in json.load(f)
                ]
        except Exception as e:
            logger.error(f"Error loading FAISS documents: {str(e)}")
            self.faiss_documents = []

    def _build_chroma_filter(self, filters: Dict) -> Dict:
        return {
            field: {"$in": value} if isinstance(value, list) else {"$eq": value}
            for field, value in filters.items()
        }

    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        for field, value in filters.items():
            if field not in metadata:
                return False
            if isinstance(value, list):
                if metadata[field] not in value:
                    return False
            elif metadata[field] != value:
                return False
        return True
