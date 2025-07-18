# src/core/retriever.py
from __future__ import annotations
import time 
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
        build_if_missing: bool = True
    ):
        """Initialize the Retriever with vector store configuration."""
        self.index_path = Path(index_path).resolve()  # Convertir a ruta absoluta
        logger.info(f"Initializing Retriever with index path: {self.index_path}")
        
        # Verificar si el directorio padre existe
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.embedder_name = embedding_model
        self.device = device

        # Inicializar el embedder
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.embedder_name,
            model_kwargs={"device": self.device},
        )

        self.store = None

        if build_if_missing:
            if self.index_exists():
                self._load_index()
            else:
                logger.warning(f"Index not found at {self.index_path}")
                logger.info("Attempting to build index...")
                # El sistema (por ejemplo, SystemInitializer) deberá manejar la construcción con productos


    def index_exists(self) -> bool:
        """Check if Chroma index exists with all required files."""
        try:
            required_files = {"chroma.sqlite3", "index_metadata.pickle"}
            existing_files = {f.name for f in self.index_path.glob("*") if f.is_file()}
            return required_files.issubset(existing_files)
        except Exception as e:
            logger.warning(f"Error checking index: {e}")
            return False

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

    def build_index(self, products: List[Product], force_rebuild: bool = False, batch_size: int = 4000):
        """Build and save a new vector index from products in safe batches."""
        if self.index_exists() and not force_rebuild:
            raise ValueError("Index already exists. Set force_rebuild=True to overwrite")

        try:
            logger.info("🚀 Starting index build at %s", self.index_path)
            self.index_path.mkdir(parents=True, exist_ok=True)

            if not products:
                logger.warning("No products provided to build index")
                return

            # Clear existing index
            if self.index_exists():
                logger.info("♻️ Removing existing index...")
                import shutil
                shutil.rmtree(self.index_path)
                self.index_path.mkdir()

            total_products = len(products)
            logger.info(f"📦 Total products to index: {total_products}")

            safe_batch_size = min(batch_size, 5000)

            # 🕒 Tracking time
            start_time = time.time()
            last_log_time = start_time

            for batch_start in range(0, total_products, safe_batch_size):
                batch_end = min(batch_start + safe_batch_size, total_products)
                batch = products[batch_start:batch_end]

                logger.info(f" Processing batch {batch_start+1}-{batch_end}/{total_products} (size: {len(batch)})")

                # Convert batch to documents
                documents = []
                for prod in batch:
                    try:
                        doc = Document(
                            page_content=prod.to_text(),
                            metadata=prod.to_metadata()
                        )
                        documents.append(doc)
                    except Exception as e:
                        logger.warning(f"Skipping product {prod.id}: {str(e)}")
                        continue

                documents = filter_complex_metadata(documents)

                if batch_start == 0:
                    self.store = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embedder,
                        persist_directory=str(self.index_path),
                        collection_metadata={"hnsw:space": "cosine"}
                    )
                    logger.info("✅ Created new Chroma index")
                else:
                    chunk_size = 1000
                    for i in range(0, len(documents), chunk_size):
                        chunk = documents[i:i + chunk_size]
                        self.store.add_documents(chunk)
                        logger.debug(f"➕ Added {len(chunk)} documents to index")

                # ⏱️ Log de progreso cada 30 segundos
                current_time = time.time()
                if current_time - last_log_time > 30:
                    elapsed = current_time - start_time
                    docs_per_sec = batch_start / elapsed if elapsed > 0 else 0
                    logger.info(f"⏱️ Progress: {batch_start}/{total_products} ({docs_per_sec:.1f} docs/sec)")
                    last_log_time = current_time

                del documents
                del batch

            logger.info(f"🎉 Successfully built index at {self.index_path}")
            logger.info(f"📊 Total documents indexed: {total_products}")

        except Exception as e:
            logger.error("❌ Failed to build index: %s", e)
            import shutil
            if self.index_path.exists():
                shutil.rmtree(self.index_path)
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

    def _doc_to_product(self, doc: Document) -> Optional[Product]:
        """Convert Document back to Product with robust error handling."""
        try:
            if not isinstance(doc.metadata, dict):
                logger.warning(f"Invalid metadata type: {type(doc.metadata)}")
                return None
                
            # Asegurar campos mínimos requeridos
            metadata = doc.metadata.copy()
            metadata.setdefault('title', 'Unknown Product')
            metadata.setdefault('price', 0.0)
            metadata.setdefault('average_rating', 0.0)
            
            # Si el contenido de la página tiene información útil, combinarlo
            if doc.page_content:
                metadata['description'] = doc.page_content[:500]  # Limitar tamaño
                
            return Product.from_dict(metadata)
        except Exception as e:
            logger.warning(f"Failed to convert document to product: {str(e)}")
            return None

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Product]:
        """Retrieve products matching the query with robust error handling."""
        try:
            docs = self._raw_retrieve(query, k, filters)
            products = []
            for doc in docs:
                product = self._doc_to_product(doc)
                if product is not None:
                    products.append(product)
                else:
                    logger.debug(f"Skipped invalid document: {doc.metadata.get('id', 'unknown')}")
            
            return products[:k]  # Asegurar no devolver más de k productos
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []  # Devolver lista vacía en caso de error

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