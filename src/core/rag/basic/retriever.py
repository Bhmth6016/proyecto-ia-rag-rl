from __future__ import annotations
# src/core/retriever.py
import json
import numpy as np
import uuid
import time
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
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
    "belleza": ["beauty", "makeup", "skincare", "cosmetics", "perfume", "lipstick", "serum", "cream", "all beauty"],
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
                'batch_size': 32  # Reduce if there are memory issues
            }
        )

        self.store = None

    def _expand_query(self, query: str) -> List[str]:
        """More robust query expansion"""
        base = _normalize(query)
        expansions = {base}
        
        # Add beauty-specific terms
        beauty_terms = ["belleza", "beauty", "maquillaje", "cosmeticos", "skincare", "cuidado facial"]
        if any(term in base for term in beauty_terms):
            expansions.update([
                "makeup", "skincare", "cosmetics", "perfume", "crema", "serum",
                "labial", "brocha", "paleta", "rubor", "base", "corrector"
            ])
        
        # Add general synonyms
        for key, syns in _SYNONYMS.items():
            if key in base:
                expansions.update(syns)
            for s in syns:
                if s in base:
                    expansions.add(key)
        
        # Ensure the original query is always included
        if base not in expansions:
            expansions.add(base)
        
        return list(expansions)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None,
        min_similarity: float = 0.3
    ) -> List[Product]:
        """Enhanced retrieval with better filter support"""
        try:
            # Parse filters from query if not provided
            if filters is None:
                filters = self._parse_filters_from_query(query)

            # Semantic search with expanded query
            expanded_query = self._expand_query(query)
            docs = self._raw_retrieve(" ".join(expanded_query), k=k*2, filters=filters)
            
            # Convert to products and apply filters
            products = []
            seen_ids = set()
            for doc in docs:
                p = self._doc_to_product(doc)
                if p and p.id not in seen_ids:
                    seen_ids.add(p.id)
                    products.append(p)
            
            # Apply additional filters that couldn't be handled by Chroma
            if filters:
                products = [p for p in products if self._matches_all_filters(p, filters)]
            
            # Score and filter by relevance
            scored_products = []
            for p in products:
                score = self._score(query, p)
                if score >= min_similarity:
                    scored_products.append((score, p))
            
            # Sort and return top-k
            scored_products.sort(key=lambda x: x[0], reverse=True)
            return [p for (score, p) in scored_products[:k]]
        
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def _parse_filters_from_query(self, query: str) -> Dict[str, Any]:
        """Extract filters from natural language query"""
        filters = {}
        
        # Price filters
        price_matches = re.findall(r'(?:precio|price)\s*(?:menor|less|below|under|<\s*)\s*(\d+)', query, re.IGNORECASE)
        if price_matches:
            filters["price_range"] = {"max": float(price_matches[0])}
        
        # Color filters
        color_matches = re.findall(r'(?:color|colou?r)\s+(rojo|azul|verde|negro|blanco|amarillo|rosado)', query, re.IGNORECASE)
        if color_matches:
            filters["color"] = color_matches[0].lower()
        
        # Wireless/Bluetooth
        if any(word in query.lower() for word in ["inalámbrico", "wireless", "bluetooth"]):
            filters["wireless"] = True
        
        # Weight filters
        weight_matches = re.findall(r'(?:peso|weight)\s*(?:menor|less|below|under|<\s*)\s*(\d+)\s*(?:g|gramos|grams|kg|kilos)', query, re.IGNORECASE)
        if weight_matches:
            filters["weight"] = {"max": float(weight_matches[0])}
        
        return filters

    def _matches_all_filters(self, product: Product, filters: Dict) -> bool:
        """Check if product matches all filters"""
        # Price filter
        if "price_range" in filters:
            price_range = filters["price_range"]
            if "max" in price_range and product.price and product.price > price_range["max"]:
                return False
            if "min" in price_range and product.price and product.price < price_range["min"]:
                return False
        
        # Color filter
        if "color" in filters:
            product_color = getattr(product, "color", "") or product.details.get("Color", "")
            if not product_color or str(product_color).lower() != filters["color"].lower():
                return False
        
        # Wireless filter
        if "wireless" in filters:
            is_wireless = any(word in (product.title + product.description).lower() 
                             for word in ["wireless", "inalámbrico", "bluetooth"])
            if not is_wireless:
                return False
        
        return True

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

            # Safely handle JSON parsing
            def safe_json_loads(json_str, default):
                try:
                    return json.loads(json_str)
                except:
                    return default

            # Reconstruct the product dictionary
            product_data = {
                "id": doc.metadata.get("id", str(uuid.uuid4())),
                "title": doc.metadata.get("title", "Untitled Product"),
                "main_category": doc.metadata.get("main_category", "Uncategorized"),
                "categories": safe_json_loads(doc.metadata.get("categories", "[]"), []),
                "price": doc.metadata.get("price", 0.0),
                "average_rating": doc.metadata.get("average_rating", 0.0),
                "description": doc.metadata.get("description", ""),
                "details": {
                    "features": safe_json_loads(doc.metadata.get("features", "[]"), [])
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
        """Enhanced hybrid scoring"""
        # Text similarity (title + description)
        text_content = f"{product.title or ''} {product.description or ''}"
        text_sim = self._text_similarity(query.lower(), text_content.lower())
        
        # Category boost
        category_boost = 1.5 if (product.main_category and 
                                any(cat.lower() in query.lower() 
                                    for cat in [product.main_category] + product.categories)) else 1.0
        
        # Rating influence (normalized)
        rating_boost = (product.average_rating/5.0 if product.average_rating else 0.5)
        
        # Price penalty for null prices
        price_penalty = 0.8 if product.price is None else 1.0
        
        return text_sim * category_boost * rating_boost * price_penalty

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
        """Verifica si el índice de Chroma existe de manera más flexible"""
        index_path = Path(self.index_path)
        if not index_path.exists():
            return False
            
        # Verificación mínima - solo que exista el directorio
        return True

    def build_index(self, products: List[Product], batch_size: int = 1000) -> None:
        """Build index with improved error handling"""
        try:
            if not products:
                raise ValueError("No products provided to build index")

            logger.info(f"Starting index build with {len(products)} products")
            
            # Convertir productos a documentos
            documents = [
                Document(
                    page_content=product.to_text(),
                    metadata=product.to_metadata()
                )
                for product in products
                if product.title and product.title != "Untitled Product"
            ]

            # Crear directorio si no existe
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            # Construir índice con persistencia
            self.store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedder,
                persist_directory=str(self.index_path),
                collection_metadata={"hnsw:space": "cosine"}
            )
            
            # Forzar persistencia inmediata
            self.store.persist()
            
            logger.info(f"✅ Successfully built index with {len(documents)} documents")

        except Exception as e:
            logger.error(f"❌ Index build failed: {str(e)}")
            # Limpiar índice incompleto
            if hasattr(self, 'store') and self.store:
                try:
                    self.store.delete_collection()
                except:
                    pass
            raise RuntimeError(f"Index construction failed: {str(e)}")