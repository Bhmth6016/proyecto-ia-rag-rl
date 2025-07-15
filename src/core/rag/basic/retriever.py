"""
Multilingual, synonym-aware retriever.
- Uses paraphrase-multilingual-MiniLM-L12-v2
- Auto-expands Spanish ↔ English synonyms
- Boosts products that match tags / compatible_devices
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Union

import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma, FAISS  # Cambio importante aquí
from langchain_community.embeddings import HuggingFaceEmbeddings  # Y aquí

from src.core.utils.logger import get_logger
from src.core.data.product import Product
from src.core.config import settings

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# Synonyms & helpers
# ----------------------------------------------------------------------

_SYNONYMS: Dict[str, List[str]] = {
    "mochila": ["backpack", "bagpack"],
    "computador": ["laptop", "notebook", "pc"],
    "auriculares": ["headphones", "headset", "earbuds"],
    "altavoz": ["speaker", "bluetooth speaker"],
    "teclado": ["keyboard"],
    "ratón": ["mouse"],
    "monitor": ["screen", "display"],
}

def _normalize(text: str) -> str:
    """Lower-case + strip accents."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower().strip()

def _expand_query(query: str) -> List[str]:
    """Return normalized + synonym-expanded query strings."""
    base = _normalize(query)
    expansions = {base}
    for key, syns in _SYNONYMS.items():
        if key in base:
            expansions.update(syns)
        for s in syns:
            if s in base:
                expansions.add(key)
    return list(expansions)

# ----------------------------------------------------------------------
# Retriever
# ----------------------------------------------------------------------

class Retriever:
    def __init__(
        self,
        index_path: Union[str, Path] = settings.VEC_DIR / settings.INDEX_NAME,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        vectorstore_type: str = settings.VECTOR_BACKEND,
        device: str = getattr(settings, "DEVICE", "cpu"),
    ):
        self.index_path = Path(index_path)
        self.embedder_name = embedding_model
        self.backend = vectorstore_type.lower()
        self.device = device
        self._load_index()

    def _load_index(self) -> None:
        try:
            self.embedder = HuggingFaceEmbeddings(
                model_name=self.embedder_name,
                model_kwargs={"device": self.device},
            )

            if self.backend == "chroma":
                if not any(self.index_path.glob("*.parquet")):
                    raise ValueError(f"No Chroma index found at {self.index_path}")
                self.store = Chroma(
                    persist_directory=str(self.index_path),
                    embedding_function=self.embedder,
                )
            else:  # FAISS
                if not (self.index_path / "index.faiss").exists():
                    raise ValueError(f"No FAISS index found at {self.index_path}")
                
                # Cambio clave aquí - usa FAISS de langchain_community
                self.store = FAISS.load_local(
                    folder_path=str(self.index_path),
                    embeddings=self.embedder,
                    allow_dangerous_deserialization=True
                )
                
                # Cargar documentos FAISS si es necesario
                self._load_faiss_docs()

            logger.info("Loaded %s index from %s", self.backend.upper(), self.index_path)

        except Exception as e:
            logger.error("Error loading index: %s", e)
            raise


    def _load_faiss_docs(self) -> None:
        docs_path = self.index_path / "docs.json"
        with open(docs_path, "r", encoding="utf-8") as f:
            self.faiss_docs = [
                Document(page_content=d["content"], metadata=d["metadata"])
                for d in json.load(f)
            ]

    # ---------------- Retrieval ----------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Product]:
        docs = self._raw_retrieve(query, k * 3, filters)  # Over-fetch
        products = [self._doc_to_product(d) for d in docs]

        scored = [(p, self._score(query, p)) for p in products]
        top = sorted(scored, key=lambda t: t[1], reverse=True)[:k]
        return [p for p, _ in top]

    def _raw_retrieve(
        self,
        query: str,
        k: int,
        filters: Optional[Dict],
    ) -> List[Document]:
        if self.backend == "chroma":
            if filters:
                return self.store.similarity_search(query, k=k, filter=self._chroma_filter(filters))
            return self.store.similarity_search(query, k=k)

        # FAISS
        q_emb = np.array(self.embedder.embed_query(query)).astype(np.float32)
        _, idxs = self.store.index.search(np.array([q_emb]), k)
        docs = [self.faiss_docs[i] for i in idxs[0] if i < len(self.faiss_docs)]
        if filters:
            docs = [d for d in docs if self._matches(d.metadata, filters)]
        return docs

    # ---------------- Scoring & Filtering ----------------

    def _score(self, query: str, product: Product) -> float:
        """Heuristic boost for tags/compatible_devices overlap."""
        q_words = set(_expand_query(query))
        tag_hits = len(q_words.intersection({t.lower() for t in product.tags or []}))
        dev_hits = len(q_words.intersection({d.lower() for d in product.compatible_devices or []}))
        return tag_hits + dev_hits

    def _doc_to_product(self, doc: Document) -> Product:
        return Product.from_dict(doc.metadata | {"page_content": doc.page_content})

    def _chroma_filter(self, f: Optional[Dict]) -> Dict:
        return {
            k: {"$in": v} if isinstance(v, list) else {"$eq": v}
            for k, v in (f or {}).items()
        }

    def _matches(self, meta: Dict, f: Optional[Dict]) -> bool:
        if not f:
            return True
        return all(
            meta.get(k) == v or (isinstance(v, list) and meta.get(k) in v)
            for k, v in f.items()
        )

    # ---------------- Debug & Helpers ----------------

    def debug(self, category: str = "Beauty", limit: int = 3):
        if self.backend == "chroma":
            docs = self.store.get(where={"category": category}, limit=limit)["documents"]
            for d in docs:
                print(d.metadata["title"], d.metadata.get("price"))

    def _expand_query(self, query: str) -> List[str]:
        return _expand_query(query)

    def debug_search(self, query: str):
        print("Expansiones de query:", _expand_query(query))
        docs = self._raw_retrieve(query, k=5, filters=None)
        print(f"Encontrados {len(docs)} documentos para '{query}'")
        for doc in docs:
            print(doc.metadata.get("title"), doc.metadata.get("category"))
