from __future__ import annotations
# src/core/rag/basic/retriever.py

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

# CAMBIO: Usar la nueva versión de Chroma
try:
    from langchain_chroma import Chroma
    CHROMA_NEW = True
except ImportError:
    from langchain_community.vectorstores import Chroma
    CHROMA_NEW = False

from langchain_huggingface import HuggingFaceEmbeddings

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
        logger.info(f"Using Chroma version: {'NEW' if CHROMA_NEW else 'OLD'}")

        self.index_path = Path(index_path).resolve()
        logger.info(f"Initializing Retriever with index path: {self.index_path}")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.embedder_name = embedding_model
        self.device = device

        self.embedder = HuggingFaceEmbeddings(
            model_name=self.embedder_name,
            model_kwargs={"device": self.device},
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32
            }
        )

        self.store = None
        self._ensure_store_loaded()
        self.feedback_weights = self._load_feedback_weights()

    # ------------------------------------------------------------
    # Ensure store loaded
    # ------------------------------------------------------------
    def _ensure_store_loaded(self):
        if not hasattr(self, "store") or self.store is None:
            try:
                if self.index_exists():
                    if CHROMA_NEW:
                        # NO pasar embedding_function
                        self.store = Chroma(
                            persist_directory=str(self.index_path)
                        )
                    else:
                        self.store = Chroma(
                            persist_directory=str(self.index_path),
                            embedding_function=self.embedder
                        )

                    logger.info("✅ Chroma store loaded successfully")
                else:
                    logger.warning("⚠️  No index found, need to build first")
            except Exception as e:
                logger.error(f"❌ Error loading Chroma store: {e}")
                self.store = None

    # Compatible con LangChain retriever
    def as_retriever(self, search_kwargs=None):
        self._ensure_store_loaded()
        return self.store.as_retriever(search_kwargs=search_kwargs) if self.store else None

    # ------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------
    def _expand_query(self, query: str) -> List[str]:
        base = _normalize(query)
        expansions = {base}

        beauty_terms = ["belleza", "beauty", "maquillaje", "cosmeticos", "skincare", "cuidado facial"]
        if any(term in base for term in beauty_terms):
            expansions.update([
                "makeup", "skincare", "cosmetics", "perfume", "crema", "serum",
                "labial", "brocha", "paleta", "rubor", "base", "corrector"
            ])

        for key, syns in _SYNONYMS.items():
            if key in base:
                expansions.update(syns)
            for s in syns:
                if s in base:
                    expansions.add(key)

        expansions.add(base)
        return list(expansions)
    
    def _load_feedback_weights(self) -> Dict[str, float]:
        """Carga pesos aprendidos de feedback positivo"""
        weights = {}
        try:
            success_log = Path("data/feedback/success_queries.log")
            if success_log.exists():
                with open(success_log, 'r', encoding='utf-8') as f:
                    for line in f:
                        record = json.loads(line)
                        product_id = record.get('selected_product_id')
                        if product_id:
                            weights[product_id] = weights.get(product_id, 0) + 1.0
        except:
            pass
        return weights

    # ------------------------------------------------------------
    # Retrieval principal
    # ------------------------------------------------------------
    def retrieve(self, query: str, k: int = 5, filters: Optional[Dict] = None, min_similarity: float = 0.3) -> List[Product]:
        try:
            self._ensure_store_loaded()

            if self.store is None:
                logger.error("❌ Store not available for retrieval")
                return []

            if filters is None:
                filters = self._parse_filters_from_query(query)

            expanded = self._expand_query(query)
            docs = self._raw_retrieve(" ".join(expanded), k=k * 2, filters=filters)

            products = []
            seen = set()

            for d in docs:
                p = self._doc_to_product(d)
                if p and p.id not in seen:
                    seen.add(p.id)
                    products.append(p)

            if filters:
                products = [p for p in products if self._matches_all_filters(p, filters)]

            scored = []
            for p in products:
                base_score = self._score(query, p)

                if base_score < min_similarity:
                    continue

                # 🔥 BOOST por feedback positivo — aprendizaje automático
                feedback_boost = self.feedback_weights.get(p.id, 0) * 0.1
                final_score = base_score + feedback_boost
                scored.append((final_score, p))

            # ✅ LÍNEAS CRÍTICAS QUE FALTABAN:
            scored.sort(key=lambda x: x[0], reverse=True)
            return [p for _, p in scored[:k]]

        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            return []
    # ------------------------------------------------------------
    # Raw chroma retrieve
    # ------------------------------------------------------------
    def _raw_retrieve(self, query: str, k: int, filters: Optional[Dict]) -> List[Document]:
        if self.store is None:
            logger.error("❌ Store not available for _raw_retrieve")
            return []

        query_expanded = " ".join(self._expand_query(query))

        try:
            if filters:
                return self.store.similarity_search(
                    query_expanded,
                    k=k,
                    filter=self._chroma_filter(filters)
                )
            else:
                return self.store.similarity_search(query_expanded, k=k)
        except Exception as e:
            logger.error(f"❌ Error in similarity_search: {e}")
            return []

    # ------------------------------------------------------------
    # Filters desde texto
    # ------------------------------------------------------------
    def _parse_filters_from_query(self, query: str) -> Dict[str, Any]:
        filters = {}

        price_matches = re.findall(r"(?:precio|price)\s*(?:menor|less|below|under|<\s*)\s*(\d+)", query, re.IGNORECASE)
        if price_matches:
            filters["price_range"] = {"max": float(price_matches[0])}

        color_matches = re.findall(r"(?:color|colou?r)\s+(rojo|azul|verde|negro|blanco|amarillo|rosado)", query, re.IGNORECASE)
        if color_matches:
            filters["color"] = color_matches[0].lower()

        if any(w in query.lower() for w in ["inalámbrico", "wireless", "bluetooth"]):
            filters["wireless"] = True

        weight_matches = re.findall(r"(?:peso|weight)\s*(?:menor|less|below|under|<\s*)\s*(\d+)", query, re.IGNORECASE)
        if weight_matches:
            filters["weight"] = {"max": float(weight_matches[0])}

        return filters

    def _matches_all_filters(self, product: Product, filters: Dict) -> bool:
        if "price_range" in filters:
            r = filters["price_range"]
            if "max" in r and product.price and product.price > r["max"]:
                return False

        if "color" in filters:
            product_color = getattr(product, "color", "") or product.details.get("Color", "")
            if not product_color or str(product_color).lower() != filters["color"].lower():
                return False

        if "wireless" in filters:
            text = (product.title + product.description).lower()
            is_wireless = any(w in text for w in ["wireless", "inalámbrico", "bluetooth"])
            if not is_wireless:
                return False

        return True

    def _doc_to_product(self, doc: Document) -> Optional[Product]:
        try:
            if not doc.metadata:
                return None

            def safe_json_load(x, default):
                try:
                    return json.loads(x)
                except:
                    return default

            product_data = {
                "id": doc.metadata.get("id", str(uuid.uuid4())),
                "title": doc.metadata.get("title", "Untitled Product"),
                "main_category": doc.metadata.get("main_category", "Uncategorized"),
                "categories": safe_json_load(doc.metadata.get("categories", "[]"), []),
                "price": doc.metadata.get("price", 0.0),
                "average_rating": doc.metadata.get("average_rating", 0.0),
                "description": doc.metadata.get("description", ""),
                "details": {
                    "features": safe_json_load(doc.metadata.get("features", "[]"), [])
                }
            }

            return Product.from_dict(product_data)

        except Exception as e:
            logger.error(f"❌ Failed to convert doc -> product: {e}")
            return None

    def _score(self, query: str, product: Product) -> float:
        try:
            if hasattr(product, "title") and product.title:
                text_sim = SequenceMatcher(None, query.lower(), product.title.lower()).ratio()
            else:
                text_sim = 0.3

            rating_boost = 1.0
            if hasattr(product, "average_rating") and product.average_rating:
                rating_boost += product.average_rating / 10.0

            return float(text_sim * rating_boost)

        except Exception:
            return 0.1

    def _text_similarity(self, q, t):
        return SequenceMatcher(None, q, t).ratio()

    def _chroma_filter(self, f: Optional[Dict]) -> Dict:
        return {
            k: {"$in": v} if isinstance(v, list) else {"$eq": v}
            for k, v in (f or {}).items()
        }

    def debug(self, category="Beauty", limit=3):
        if self.store:
            docs = self.store.get(where={"category": category}, limit=limit)["documents"]
            for d in docs:
                print(d.metadata["title"], d.metadata.get("price"))

    def debug_search(self, query: str):
        print("Query expansions:", self._expand_query(query))
        docs = self._raw_retrieve(query, k=5, filters=None)
        print(f"Found {len(docs)} docs for '{query}'")
        for doc in docs:
            print(doc.metadata.get("title"), doc.metadata.get("category"))

    def index_exists(self):
        return Path(self.index_path).exists()

    # ----------------------------------------------------------------------
    # MÉTODOS NUEVOS
    # ----------------------------------------------------------------------

    def _safe_clear_index(self):
        """Eliminar el índice de forma segura, manejando archivos bloqueados."""
        import time
        max_retries = 3
        retry_delay = 1  # segundos

        for attempt in range(max_retries):
            try:
                if self.index_path.exists():
                    import shutil
                    shutil.rmtree(self.index_path)
                    logger.info(f"✅ Index cleared on attempt {attempt + 1}")
                    return
            except PermissionError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️  File locked, retrying in {retry_delay}s... (attempt {attempt + 1})")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"❌ Failed to clear index after {max_retries} attempts: {e}")
                    self._force_clear_index()
            except Exception as e:
                logger.error(f"❌ Unexpected error clearing index: {e}")
                raise

    def _force_clear_index(self):
        """Método forzado para limpiar el índice cuando fallan los métodos normales."""
        try:
            import os

            if hasattr(self, 'store') and self.store:
                try:
                    self.store = None
                except:
                    pass

            if self.index_path.exists():
                for root, dirs, files in os.walk(self.index_path, topdown=False):
                    for name in files:
                        file_path = os.path.join(root, name)
                        try:
                            os.chmod(file_path, 0o777)
                            os.remove(file_path)
                        except Exception as e:
                            logger.warning(f"Could not remove {file_path}: {e}")

                    for name in dirs:
                        dir_path = os.path.join(root, name)
                        try:
                            os.chmod(dir_path, 0o777)
                            os.rmdir(dir_path)
                        except Exception as e:
                            logger.warning(f"Could not remove directory {dir_path}: {e}")

                try:
                    os.rmdir(self.index_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"❌ Force clear also failed: {e}")
            raise RuntimeError(f"Could not clear index directory: {e}")

    # ----------------------------------------------------------------------
    # build_index actualizado
    # ----------------------------------------------------------------------
    def build_index(self, products: List[Product], batch_size: int = 1000):
        try:
            if not products:
                raise ValueError("No products provided to build index")

            logger.info(f"Building index with {len(products)} products")

            # Limpiar directorio existente con manejo seguro
            if self.index_path.exists():
                self._safe_clear_index()
                logger.info("🗑️  Existing index cleared")

            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            documents = [
                Document(page_content=p.to_text(), metadata=p.to_metadata())
                for p in products if p.title and p.title.strip()
            ]

            logger.info(f"📝 Creating {len(documents)} documents")

            if CHROMA_NEW:
    # Versión nueva: NO usa embedding_function aquí
                self.store = Chroma.from_documents(
                    documents=documents,
                    persist_directory=str(self.index_path),
                    collection_metadata={"hnsw:space": "cosine"}
                )
            else:
                # Versión vieja: sí usa embedding_function
                self.store = Chroma.from_documents(
                    documents=documents,
                    embedding_function=self.embedder,
                    persist_directory=str(self.index_path),
                    collection_metadata={"hnsw:space": "cosine"}
                )

            logger.info("✅ Index built successfully")

        except Exception as e:
            logger.error(f"❌ Index build failed: {e}")
            raise RuntimeError(f"Index build failed: {e}")
