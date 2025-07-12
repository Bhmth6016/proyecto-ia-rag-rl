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
from src.core.config import settings  # Importa settings

logger = get_logger(__name__)


class Retriever:
    def __init__(
        self,
        index_path: Union[str, Path] = settings.VEC_DIR / settings.INDEX_NAME,
        embedding_model: str = settings.EMBEDDING_MODEL,
        vectorstore_type: str = settings.VECTOR_BACKEND,
        device: str = getattr(settings, "DEVICE", "cpu")  # fallback a 'cpu'
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

    def retrieve(self, query: str, k: int = 5, max_price: float = 30.0) -> List[Document]:
        """Enhanced retrieval with price filtering"""
        try:
            # First get category matches
            base_results = self.retriever.invoke(
                query,
                k=k * 3,
                filter={"category": {"$eq": "Beauty"}}
            )

            # Then filter by price
            filtered = []
            for doc in base_results:
                try:
                    price = float(doc.metadata.get("price", "0").replace("$", ""))
                    if price <= max_price:
                        filtered.append(doc)
                except (ValueError, AttributeError):
                    continue

            return filtered[:k]
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return []

    def _filter_by_price(self, docs: List[Document], max_price: float) -> List[Document]:
        filtered = []
        for doc in docs:
            try:
                price = float(doc.metadata.get("price", "0").replace("$", ""))
                if price <= max_price:
                    filtered.append(doc)
            except (ValueError, AttributeError):
                continue
        return filtered

    def _retrieve_faiss(self, query: str, k: int, filters: Optional[Dict]) -> List[Document]:
        query_embedding = np.array(self.embedder.embed_query(query)).astype("float32")
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

    def debug_index(self, category="Beauty", limit=5):
        """Debug what's actually in the index"""
        results = self.vectorstore.get(
            where={"category": category},
            limit=limit
        )
        for doc in results["documents"]:
            print(f"Title: {doc.metadata['title']}")
            print(f"Price: {doc.metadata.get('price', 'N/A')}")
            print(f"Content: {doc.page_content[:200]}...\n")
