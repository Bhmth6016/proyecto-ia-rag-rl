# src/core/retriever.py
from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Union

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata

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
        index_path: Union[str, Path] = settings.VECTOR_INDEX_PATH,
        embedding_model: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
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
            raise FileNotFoundError(
                f"Chroma index incomplete or missing at {self.index_path}.\n"
                "Run 'python main.py index' to build it.\n"
                f"Required files: chroma.sqlite3, index_metadata.pickle"
            )

    def index_exists(self) -> bool:
        """Check if Chroma index exists with all required files."""
        required_files = {"chroma.sqlite3", "index_metadata.pickle"}
        existing_files = {f.name for f in self.index_path.glob("*")}
        return required_files.issubset(existing_files)

    def _load_index(self) -> None:
        """Load the existing vector index from disk."""
        try:
            self.store = Chroma(
                persist_directory=str(self.index_path),
                embedding_function=self.embedder,
            )
            logger.info("Loaded Chroma index from %s", self.index_path)
        except Exception as e:
            logger.error("Error loading index: %s", e)
            raise

    def build_index(self, products: List[Product], force_rebuild: bool = False):
        if self.index_exists() and not force_rebuild:
            raise ValueError("Index already exists. Set force_rebuild=True to overwrite")
        """Build and save a new vector index from products."""
        try:
            logger.info("Building index at %s", self.index_path)
            self.index_path.mkdir(parents=True, exist_ok=True)
            
            if self.index_exists() and not force_rebuild:
                logger.warning("Index already exists. Use force_rebuild=True to overwrite.")
                return

            if not products:
                logger.warning("No products provided to build index")
                return

            documents = [
                Document(
                    page_content=prod.to_text(),
                    metadata=prod.to_metadata()
                )
                for prod in products
            ]

            # Filter complex metadata
            documents = filter_complex_metadata(documents)

            # Clear existing index if any
            if self.index_exists():
                import shutil
                shutil.rmtree(self.index_path)
                self.index_path.mkdir()
            
            # Create new Chroma index
            self.store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedder,
                persist_directory=str(self.index_path)
            )
            self.store.persist()
            logger.info(f"✅ Chroma index built at {self.index_path} with {len(documents)} documents")
        except Exception as e:
            logger.error("Failed to build index: %s", e)
            raise

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
            docs = self._raw_retrieve(query, k, filters)
            products = [self._doc_to_product(d) for d in docs]
            return products
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

    # ---------------------- Debug Methods ----------------------

    def debug(self, category: str = "Beauty", limit: int = 3) -> None:
        """Debug method to inspect indexed documents."""
        docs = self.store.get(where={"category": category}, limit=limit)["documents"]
        for d in docs:
            print(d.metadata["title"], d.metadata.get("price"))

    def debug_search(self, query: str) -> None:
        """Debug method to test search functionality."""
        print("Query expansions:", _expand_query(query))
        docs = self._raw_retrieve(query, k=5, filters=None)
        print(f"Found {len(docs)} documents for '{query}'")
        for doc in docs:
            print(doc.metadata.get("title"), doc.metadata.get("category"))