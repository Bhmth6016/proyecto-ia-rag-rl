# retriever.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union

import unicodedata
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.core.utils.logger import get_logger
from src.core.data.product import Product
from src.core.utils import settings

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
    "monitor": ["screen", "display"]
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
        index_path: Union[str, Path] = settings.VECTOR_INDEX_PATH,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = getattr(settings, "DEVICE", "cpu"),
    ):
        """Initialize the Retriever with vector store configuration."""
        self.index_path = Path(index_path)
        self.embedder_name = embedding_model
        self.device = device
        
        # Initialize embedder but don't load index yet
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.embedder_name,
            model_kwargs={"device": self.device},
        )
        
        # Initialize store as None - will be set when building or loading index
        self.store = None

        # Load the index if it exists
        if self.index_exists():
            self._load_index()
        else:
            logger.warning("No index found at %s. Please build the index first.", self.index_path)

    def index_exists(self) -> bool:
        """Check if the vector index exists in the configured path."""
        return (self.index_path / "chroma.sqlite3").exists()

    def _load_index(self) -> None:
        """Load the existing vector index from disk."""
        try:
            if not self.index_exists():
                raise FileNotFoundError(f"No Chroma index found at {self.index_path}")

            self.store = Chroma(
                persist_directory=str(self.index_path),
                embedding_function=self.embedder,
            )
            logger.info("Loaded Chroma index from %s", self.index_path)

        except Exception as e:
            logger.error("Error loading index: %s", e)
            raise

    def build_index(self, products_file: Path) -> None:
        """Carga desde JSON unificado y construye índice"""
        logger.info("🛠 Building index from %s", products_file)
        
        with open(products_file, 'r', encoding='utf-8') as f:
            products = [Product.from_dict(item) for item in json.load(f)]
        
        documents = [
            Document(
                page_content=p.to_text(),
                metadata=p.to_metadata()
            ) for p in products
        ]
        
        self.store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedder,
            persist_directory=str(self.index_path)
        )
        logger.info("✅ Index built at %s with %d documents", self.index_path, len(documents))

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Product]:
        """Retrieve products matching the query.
        
        Args:
            query: Search query string
            k: Number of results to return
            filters: Dictionary of filters to apply
            
        Returns:
            List of matching Product objects
        """
        try:
            docs = self._raw_retrieve(query, k * 3, filters)  # Over-fetch
            products = [self._doc_to_product(d) for d in docs]

            scored = [(p, self._score(query, p)) for p in products]
            top = sorted(scored, key=lambda t: t[1], reverse=True)[:k]
            return [p for p, _ in top]
        except Exception as e:
            logger.error("Error during retrieval: %s", e)
            raise

    def _raw_retrieve(
        self,
        query: str,
        k: int,
        filters: Optional[Dict],
    ) -> List[Document]:
        """Internal method for raw document retrieval."""
        query_expanded = " ".join(_expand_query(query))

        if filters:
            return self.store.similarity_search(
                query_expanded, 
                k=k, 
                filter=self._chroma_filter(filters)
            )
        return self.store.similarity_search(query_expanded, k=k)

    def _score(self, query: str, product: Product) -> float:
        """Calculate relevance score between query and product."""
        q_words = set(_expand_query(query))
        tag_hits = len(q_words.intersection({t.lower() for t in (product.tags or [])}))
        dev_hits = len(q_words.intersection({d.lower() for d in (product.compatible_devices or [])}))
        return tag_hits + dev_hits

    def _doc_to_product(self, doc: Document) -> Product:
        """Convert Document back to Product."""
        try:
            return Product.from_dict(doc.metadata | {"page_content": doc.page_content})
        except Exception as e:
            logger.error("Error converting document to product: %s", e)
            raise

    def _chroma_filter(self, f: Optional[Dict]) -> Dict:
        """Format filters for ChromaDB."""
        return {
            k: {"$in": v} if isinstance(v, list) else {"$eq": v}
            for k, v in (f or {}).items()
        }

    def _matches(self, meta: Dict, f: Optional[Dict]) -> bool:
        """Check if document matches filters."""
        if not f:
            return True
        return all(
            meta.get(k) == v or (isinstance(v, list) and meta.get(k) in v)
            for k, v in f.items()
        )