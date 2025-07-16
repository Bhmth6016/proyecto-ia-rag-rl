from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Union

import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from src.core.utils.logger import get_logger
from src.core.data.product import Product
from src.core.config import settings
from src.core.rag.basic.index_utils import build_faiss_index

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
        vectorstore_type: str = settings.VECTOR_BACKEND,
        device: str = getattr(settings, "DEVICE", "cpu"),
    ):
        """Initialize the Retriever with vector store configuration."""
        self.index_path = Path(index_path)
        self.embedder_name = embedding_model
        self.backend = vectorstore_type.lower()
        self.device = device
        
        # Initialize embedder but don't load index yet
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.embedder_name,
            model_kwargs={"device": self.device},
        )
        
        # Initialize store as None - will be set when building or loading index
        self.store = None
        self.faiss_docs = []

        # Load the index if it exists
        if self.index_exists():
            self._load_index()
        else:
            logger.warning("No index found at %s. Please build the index first.", self.index_path)

    def index_exists(self) -> bool:
        """Check if the vector index exists in the configured path."""
        if not self.index_path.exists():
            return False
            
        if self.backend == "chroma":
            return (self.index_path / "chroma.sqlite3").exists()
        else:  # FAISS
            return (self.index_path / "index.faiss").exists()

    def _load_index(self) -> None:
        """Load the existing vector index from disk."""
        try:
            if not self.index_exists():
                raise FileNotFoundError(f"No {self.backend} index found at {self.index_path}")

            self.embedder = HuggingFaceEmbeddings(
                model_name=self.embedder_name,
                model_kwargs={"device": self.device},
            )

            if self.backend == "chroma":
                self.store = Chroma(
                    persist_directory=str(self.index_path),
                    embedding_function=self.embedder,
                )
            else:  # FAISS
                self.store = FAISS.load_local(
                    folder_path=str(self.index_path),
                    embeddings=self.embedder,
                    allow_dangerous_deserialization=True
                )
                self._load_faiss_docs()

            logger.info("Loaded %s index from %s", self.backend.upper(), self.index_path)

        except Exception as e:
            logger.error("Error loading index: %s", e)
            raise

    def _load_faiss_docs(self) -> None:
        """Load the document metadata for FAISS index."""
        docs_path = self.index_path / "docs.json"
        try:
            with open(docs_path, "r", encoding="utf-8") as f:
                self.faiss_docs = [
                    Document(page_content=d["content"], metadata=d["metadata"])
                    for d in json.load(f)
                ]
            logger.info("Loaded FAISS documents from %s", docs_path)
        except Exception as e:
            logger.error("Error loading FAISS documents: %s", e)
            raise

    
    '''def build_index(self, products: List[Product]) -> None:
        logger.debug("🛠 build_index started with %d products", len(products))
        if not products:
            logger.warning("No products; exiting without creating index.")
            return

        # Ensure clean state
        self.store = None
        if self.index_exists():
            self._cleanup_existing_index()

        try:
            self.index_path.mkdir(parents=True, exist_ok=True)
            
            documents = []
            for prod in products:
                documents.append(
                    Document(
                        page_content=prod.to_text(),
                        metadata=prod.to_metadata()
                    )
                )

            if self.backend == "chroma":
                # Explicitly close any existing Chroma connection
                if hasattr(self, '_chroma_client'):
                    self._chroma_client.close()
                
                self.store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedder,
                    persist_directory=str(self.index_path)
                )
                self._chroma_client = self.store._client
                logger.info(f"✅ Chroma index built at {self.index_path}")
            else:
                self.store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embedder
                )
                self.store.save_local(str(self.index_path))

                # Save document metadata for reloading
                docs_path = self.index_path / "docs.json"
                with open(docs_path, "w", encoding="utf-8") as f:
                    json.dump(
                        [
                            {"content": d.page_content, "metadata": d.metadata}
                            for d in documents
                        ],
                        f,
                        ensure_ascii=False,
                        indent=2
                    )

            logger.info("✅ Index built with %d documents", len(documents))
        except Exception as e:
            logger.error("Index build error: %s", e, exc_info=True)
            raise'''
    
    def build_index(self, products: Union[List[Product], str, Path]) -> None:
        """Build index from Product list or unified JSON file"""
        logger.debug("🛠 build_index started")

        # Permitir cargar desde archivo JSON
        if isinstance(products, (str, Path)):
            products_path = Path(products)
            if not products_path.exists():
                raise FileNotFoundError(f"Products file not found: {products_path}")
            
            logger.info(f"📄 Loading products from JSON file: {products_path}")
            with open(products_path, 'r', encoding='utf-8') as f:
                products_data = json.load(f)
            products = [Product.from_dict(item) for item in products_data]

        # Validar que haya productos
        if not products:
            logger.warning("No products; exiting without creating index.")
            return

        # Limpiar índice anterior si existe
        self.store = None
        if self.index_exists():
            self._cleanup_existing_index()

        try:
            self.index_path.mkdir(parents=True, exist_ok=True)

            documents = [
                Document(
                    page_content=prod.to_text(),
                    metadata=prod.to_metadata()
                )
                for prod in products
            ]

            if self.backend == "chroma":
                from langchain_chroma import Chroma  # usar nuevo import si ya hiciste upgrade
                self.store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedder,
                    persist_directory=str(self.index_path)
                )
                self.store.persist()
                logger.info(f"✅ Chroma index built at {self.index_path} with {len(documents)} documents")

            else:  # FAISS
                from src.core.retrieval.index_utils import build_faiss_index
                self.store = build_faiss_index(self.embedder, documents, batch_size=128)
                self.store.save_local(str(self.index_path))

                # Guardar metadatos para recarga posterior
                docs_path = self.index_path / "docs.json"
                with open(docs_path, "w", encoding="utf-8") as f:
                    json.dump(
                        [{"content": d.page_content, "metadata": d.metadata} for d in documents],
                        f,
                        ensure_ascii=False,
                        indent=2
                    )
                logger.info(f"✅ FAISS index built at {self.index_path} with {len(documents)} documents")

        except Exception as e:
            logger.error("❌ Index build error: %s", e, exc_info=True)
            raise


    def _cleanup_existing_index(self) -> None:
        """Thoroughly clean up existing index files"""
        import shutil
        import time
        
        # Close any existing Chroma connection
        if self.backend == "chroma" and hasattr(self, '_chroma_client'):
            try:
                self._chroma_client.close()
            except:
                pass
        
        # Wait for resources to be released
        time.sleep(1)
        
        # Remove directory contents
        for item in self.index_path.glob('*'):
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                try:
                    item.unlink()
                except:
                    pass
        
        # Recreate directory
        shutil.rmtree(self.index_path, ignore_errors=True)
        self.index_path.mkdir(parents=True, exist_ok=True)

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

        if self.backend == "chroma":
            if filters:
                return self.store.similarity_search(
                    query_expanded, 
                    k=k, 
                    filter=self._chroma_filter(filters)
                )
            return self.store.similarity_search(query_expanded, k=k)

        # FAISS implementation
        q_emb = np.array(self.embedder.embed_query(query_expanded)).astype(np.float32)
        _, idxs = self.store.index.search(np.array([q_emb]), k)
        docs = [self.faiss_docs[i] for i in idxs[0] if i < len(self.faiss_docs)]
        if filters:
            docs = [d for d in docs if self._matches(d.metadata, filters)]
        return docs

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

    # ---------------------- Debug Methods ----------------------
'''
    def debug(self, category: str = "Beauty", limit: int = 3) -> None:
        """Debug method to inspect indexed documents."""
        if self.backend == "chroma":
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
            '''