# src/core/retriever.py
from __future__ import annotations
import json  # Añadir esto con las otras importaciones
import numpy as np
import time 
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Union
import re
from difflib import SequenceMatcher
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
    "mochila": ["backpack", "bagpack", "laptop bag", "daypack"],
    "computador": ["laptop", "notebook", "pc", "computer", "macbook"],
    "auriculares": ["headphones", "headset", "earbuds", "audífonos", "earphones"],
    "altavoz": ["speaker", "bluetooth speaker", "parlante", "soundbar"],
    "teclado": ["keyboard", "mechanical keyboard", "wireless keyboard"],
    "ratón": ["mouse", "trackpad", "wireless mouse"],
    "monitor": ["screen", "display", "pantalla", "monitor"],
    "cámara": ["camera", "webcam", "dslr", "videocámara", "camcorder", "mirrorless"],
    "instrumento": ["instrument", "guitar", "piano", "violin", "drum", "keyboard", "flauta", "bajo"],
    "software": ["app", "application", "program", "software", "apk", "license"],
    "herramienta": ["tool", "screwdriver", "drill", "hammer", "wrench", "toolkit"],
    "juguete": ["toy", "game", "juego", "puzzle", "board game", "lego", "doll"],
    "revista": ["magazine", "subscription", "suscripción", "issue", "editorial"],
    "película": ["movie", "film", "blu-ray", "dvd", "streaming", "show", "tv series"],
    "tarjeta": ["gift card", "voucher", "tarjeta regalo", "código", "redeem"],
    "deporte": ["sport", "ball", "bike", "outdoor", "fitness", "exercise", "yoga", "gym"],
    "jardín": ["garden", "patio", "outdoor", "grill", "barbecue", "mower", "plant"],
    "suscripción": ["subscription", "box", "plan", "monthly", "auto-renew"],
    "belleza": ["beauty", "makeup", "skincare", "cosmetics", "perfume", "lipstick", "serum", "cream"],
    "oficina": ["office", "printer", "stationery", "paper", "desk", "chair", "shredder"],
    "musica": ["music", "instrumentos", "instrumentos musicales", "audio"],
    "app": ["application", "software", "programa", "aplicación"],
    "videojuego": ["video game", "juego", "game", "pc game"]
}

def _normalize(text: str) -> str:
    """Lower-case + strip accents."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower().strip()

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
        logger.info(f"Initializing Retriever (store exists: {Path(index_path).exists()})")

        """Initialize the Retriever with vector store configuration."""
        self.index_path = Path(index_path).resolve()
        logger.info(f"Initializing Retriever with index path: {self.index_path}")
        
        # Ensure the parent directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.embedder_name = embedding_model
        self.device = device

        # Initialize the embedder
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.embedder_name,
            model_kwargs={"device": self.device},
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 32  # Reduce si hay problemas de memoria
            }
        )

        self.store = None


    def _expand_query(self, query: str) -> List[str]:
        """More robust query expansion"""
        def _normalize(text: str) -> str:
            """Lower-case + strip accents."""
            text = unicodedata.normalize("NFD", text)
            text = "".join(c for c in text if unicodedata.category(c) != "Mn")
            return text.lower().strip()
        
        base = _normalize(query)
        expansions = {base}
        
        # Add category-specific terms
        if "musica" in base:
            expansions.update(["guitar", "piano", "drums"])
        if "app" in base:
            expansions.update(["mobile", "android", "ios"])
            
        # Add general synonyms
        for key, syns in _SYNONYMS.items():
            if key in base:
                expansions.update(syns)
            for s in syns:
                if s in base:
                    expansions.add(key)
        return list(expansions)

    def build_index(self, products: List[Product], batch_size: int = 1000) -> None:
        """Build index with proper error handling and batching"""
        try:
            if not products:
                raise ValueError("No products provided to build index")

            # Pre-filtra productos inválidos
            valid_products = []
            for p in products:
                if not p.title or p.title == "Untitled Product":
                    continue
                if not hasattr(p, 'description') or not p.description:
                    p.description = ""  # Asigna valor por defecto
                valid_products.append(p)

            logger.info(f"Building index from {len(valid_products)} valid products")

            # Procesamiento por lotes
            documents = []
            for i in range(0, len(valid_products), batch_size):
                batch = valid_products[i:i + batch_size]
                batch_docs = []
                
                for product in batch:
                    try:
                        doc = Document(
                            page_content=product.to_text(),
                            metadata=product.to_metadata()
                        )
                        batch_docs.append(doc)
                    except Exception as e:
                        logger.warning(f"Skipping product {getattr(product, 'id', 'unknown')}: {str(e)}")
                        continue
                
                # Verifica embeddings antes de insertar
                if not batch_docs:
                    logger.warning("Empty batch, skipping...")
                    continue

                # Construye o actualiza el índice
                if not self.store:
                    self.store = Chroma.from_documents(
                        documents=batch_docs,
                        embedding=self.embedder,
                        persist_directory=str(self.index_path))
                else:
                    self.store.add_documents(batch_docs)

            logger.info(f"✅ Successfully built index with {len(documents)} documents")

        except Exception as e:
            logger.error(f"❌ Index build failed: {str(e)}")
            raise RuntimeError(f"Index construction failed: {str(e)}")


    def _raw_retrieve(
        self,
        query: str,
        k: int,
        filters: Optional[Dict],
    ) -> List[Document]:
        """Internal method for raw document retrieval."""
        query_expanded = " ".join(self._expand_query(query))

        if filters:
            return self.store.similarity_search(
                query_expanded, 
                k=k, 
                filter=self._chroma_filter(filters)
            )
        return self.store.similarity_search(query_expanded, k=k)

    def _doc_to_product(self, doc: Document) -> Optional[Product]:
        try:
            if not doc.metadata:
                return None

            # Reconstruye el diccionario del producto
            product_data = {
                "id": doc.metadata.get("id", str(uuid.uuid4())),
                "title": doc.metadata.get("title", "Untitled Product"),
                "main_category": doc.metadata.get("main_category", "Uncategorized"),
                "categories": json.loads(doc.metadata.get("categories", "[]")),
                "price": doc.metadata.get("price", 0.0),
                "average_rating": doc.metadata.get("average_rating", 0.0),
                "description": doc.metadata.get("description", ""),
                "details": {
                    "features": json.loads(doc.metadata.get("features", "[]"))
                }
            }
            
            return Product.from_dict(product_data)
        except Exception as e:
            logger.error(f"Failed to convert document to product: {str(e)}")
            return None

    def _text_similarity(self, query: str, text: str) -> float:
        """Calculate text similarity using SequenceMatcher."""
        return SequenceMatcher(None, query, text).ratio()

    def _score(self, query: str, product: Product) -> float:
        """Enhanced hybrid scoring with null checks"""
        # Text similarity
        text_content = f"{product.title or ''} {product.description or ''}"
        text_sim = self._text_similarity(query, text_content)
        
        # Category boost (with null check)
        category_boost = 1.5 if (product.main_category and 
                                query.lower() in product.main_category.lower()) else 1.0
        
        # Rating influence (with null check)
        rating_boost = (product.average_rating/5.0 
                    if product.average_rating is not None 
                    else 0.5)
        
        return text_sim * category_boost * rating_boost

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Product]:
        """Hybrid retrieval: semantic + keyword + filter search."""
        try:
            # 1. Semantic search
            docs = self._raw_retrieve(query, k=20, filters=filters)
            sem_products = []
            for doc in docs:
                p = self._doc_to_product(doc)
                if p:
                    sem_products.append(p)

            # 2. Keyword ranking
            for p in sem_products:
                p._keyword_score = self._score(query, p)

            # 3. Hybrid sorting (you can adjust weights)
            ranked = sorted(
                sem_products,
                key=lambda p: (p._keyword_score, p.average_rating),
                reverse=True
            )

            # 4. Return top-k
            return ranked[:k]
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return []

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
        print("Query expansions:", self._expand_query(query))
        docs = self._raw_retrieve(query, k=5, filters=None)
        print(f"Found {len(docs)} documents for '{query}'")
        for doc in docs:
            print(doc.metadata.get("title"), doc.metadata.get("category"))

    def index_exists(self) -> bool:
        """Verifica si el índice de Chroma existe, con logging detallado."""
        index_path = Path(self.index_path)
        exists = index_path.exists()
        
        # Log detallado
        logger.info(
            f"Verificando índice en: {self.index_path}\n"
            f"• Directorio existe: {exists}\n"
            f"• Contenido del directorio: {list(index_path.glob('*')) if exists else 'N/A'}"
        )
        
        # Verificación adicional de archivos críticos de Chroma
        if exists:
            required_files = ['chroma.sqlite3', 'chroma-collections.parquet', 'chroma-embeddings.parquet']
            missing_files = [f for f in required_files if not (index_path / f).exists()]
            
            if missing_files:
                logger.warning(f"Índice incompleto. Faltan archivos: {missing_files}")
                return False
        
        return exists