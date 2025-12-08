from __future__ import annotations
# src/core/rag/basic/retriever.py
import os
import json
import numpy as np  # ✅ Asegurar que numpy está importado
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

# 🔥 CAMBIO: Usar el embedder local más robusto
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HAS_LANGCHAIN_HF = True
except ImportError:
    # Fallback a sentence-transformers directamente
    from sentence_transformers import SentenceTransformer
    HAS_LANGCHAIN_HF = False

from src.core.utils.logger import get_logger
from src.core.data.product import Product
from src.core.config import settings

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# Synonyms & helpers
# ----------------------------------------------------------------------

_SYNONYMS: Dict[str, List[str]] = {
    # --- EXISTENTES ---
    "pelea": ["fight", "fighting", "combat", "battle", "versus", "vs", "lucha", "fighter"],
    "smash": ["super smash", "smash bros", "smash brothers", "nintendo smash"],
    "mario": ["super mario", "mario bros", "nintendo mario"],

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
    "videojuego": ["video game", "juego", "game", "pc game"],


    # --- 🔥 NUEVOS SINÓNIMOS REQUERIDOS para videojuegos ---
    "juego": ["game", "video game", "videogame", "gaming"],
    "nintendo": ["switch", "wii", "gamecube", "nes", "snes", "n64"],
    "playstation": ["ps4", "ps5", "sony playstation"],
    "xbox": ["xbox one", "xbox series", "microsoft xbox"],

    # 🔥 Géneros gaming
    "acción": ["action", "shooter", "fps", "third person"],
    "aventura": ["adventure", "rpg", "role playing"],
    "deportes": ["sports", "fifa", "nba", "madden"],
    "carreras": ["racing", "drive", "simulator"]
}


def _normalize(text: str) -> str:
    """Lower-case + strip accents."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.lower().strip()

# ----------------------------------------------------------------------
# 🔥 NUEVO: Custom Embedder para embeddings 100% locales
# ----------------------------------------------------------------------
class LocalEmbedder:
    """Embedder local que no requiere conexión a internet"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa el modelo localmente"""
        try:
            logger.info(f"🔧 Inicializando embedder local: {self.model_name}")
            
            if HAS_LANGCHAIN_HF:
                # Usar LangChain HuggingFaceEmbeddings
                self.model = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": self.device},
                    encode_kwargs={
                        "normalize_embeddings": True,
                        "batch_size": 32
                    }
                )
            else:
                # Fallback a SentenceTransformer directamente
                self.model = SentenceTransformer(self.model_name, device=self.device)
            
            logger.info(f"✅ Embedder local inicializado: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando embedder local {self.model_name}: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Genera embedding para una query"""
        try:
            if HAS_LANGCHAIN_HF:
                return self.model.embed_query(text)
            else:
                # Usar SentenceTransformer directamente
                embedding = self.model.encode(text, normalize_embeddings=True)
                return embedding.tolist()
        except Exception as e:
            logger.error(f"❌ Error generando embedding: {e}")
            # Fallback: embedding simple
            return [0.0] * 384
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings para múltiples documentos"""
        try:
            if HAS_LANGCHAIN_HF:
                return self.model.embed_documents(texts)
            else:
                embeddings = self.model.encode(texts, normalize_embeddings=True)
                return embeddings.tolist()
        except Exception as e:
            logger.error(f"❌ Error generando embeddings de documentos: {e}")
            # Fallback: embeddings simples
            return [[0.0] * 384 for _ in texts]

# ----------------------------------------------------------------------
# Retriever
# ----------------------------------------------------------------------
class Retriever:
    def __init__(
        self,
        index_path: Union[str, Path] = settings.VECTOR_INDEX_PATH,
        embedding_model: str = "all-MiniLM-L6-v2",  # Siempre local
        device: str = getattr(settings, "DEVICE", "cpu"),
        # 🔥 Ahora configurable globalmente — toma ML si está habilitado
        use_product_embeddings: bool = settings.ML_ENABLED
    ):
        logger.info(f"Initializing Retriever (store exists: {Path(index_path).exists()})")
        logger.info(f"Using Chroma version: {'NEW' if CHROMA_NEW else 'OLD'}")
        logger.info(f"ML Embeddings (initial flag): {'Enabled' if use_product_embeddings else 'Disabled'}")

        self.index_path = Path(index_path).resolve()
        logger.info(f"Initializing Retriever with index path: {self.index_path}")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.embedder_name = embedding_model
        self.device = device

        # 🔥 Config ML: bandera inicial pero puede cambiarse si el índice las soporta
        self.use_product_embeddings = use_product_embeddings  

        # Siempre embedder local
        self.embedder = LocalEmbedder(
            model_name=self.embedder_name,
            device=self.device
        )

        self.store = None
        self._ensure_store_loaded()

        # 🔥 NEW → Auto-switch a ML si el índice ya tiene embeddings especializados
        if self.index_exists():
            self._check_ml_capabilities()

        self.feedback_weights = self._load_feedback_weights()

    # ------------------------------------------------------------
    # 🔥 NUEVO MÉTODO: Similaridad con embeddings propios
    # ------------------------------------------------------------
    def _calculate_similarity_with_embeddings(self, query: str, product: Product) -> float:
        """Calcula similitud usando embeddings propios si están disponibles"""
        try:
            # Verificar si el producto tiene embedding y si está configurado para usarlo
            if self.use_product_embeddings and hasattr(product, 'embedding') and product.embedding:
                logger.debug(f"[Embedding Similarity] Product {product.id} tiene embedding propio")
                
                # Asegurar que el embedding sea un array numpy
                product_embedding = np.array(product.embedding, dtype=np.float32)
                
                # Generar embedding para la query
                query_embedding = np.array(self.embedder.embed_query(query), dtype=np.float32)
                
                # Verificar dimensiones
                if len(query_embedding) != len(product_embedding):
                    logger.warning(f"[Embedding Similarity] Dimension mismatch: query={len(query_embedding)}, product={len(product_embedding)}")
                    return 0.5
                
                # Normalizar vectores
                query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
                product_norm = product_embedding / (np.linalg.norm(product_embedding) + 1e-10)
                
                # Calcular similitud coseno
                similarity = np.dot(query_norm, product_norm)
                
                # Asegurar que esté en rango [0, 1]
                similarity = max(0.0, min(1.0, float(similarity)))
                
                logger.debug(f"[Embedding Similarity] Similitud: {similarity:.3f}")
                return similarity
                
        except Exception as e:
            logger.debug(f"[Embedding Similarity] Error usando embeddings propios: {e}")
        
        # Fallback al método original
        return self._score(query, product)
    
    def search(self, query: str, k: int = 5, **kwargs) -> List[Product]:
        """
        Método de compatibilidad para WorkingAdvancedRAGAgent.
        Llama a retrieve() y asegura que devuelve productos.
        """
        try:
            logger.info(f"[search] Buscando '{query[:50]}...' con k={k}")
            
            # Llama al método retrieve existente
            results = self.retrieve(query=query, k=k, **kwargs)
            
            # Filtrar solo objetos Product
            products = [item for item in results if isinstance(item, Product)]
            
            logger.info(f"[search] Devueltos {len(products)} productos (de {len(results)} resultados)")
            return products
            
        except Exception as e:
            logger.error(f"❌ Error en search(): {e}")
            return []
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
                        # 🔥 CAMBIO: Usar custom embedding function para compatibilidad
                        self.store = Chroma(
                            persist_directory=str(self.index_path),
                            embedding_function=self._get_langchain_embedding_function()
                        )

                    logger.info("✅ Chroma store loaded successfully")
                    
                    # 🔥 NUEVO: Verificar si el índice tiene metadata ML
                    self._check_ml_capabilities()
                    
                else:
                    logger.warning("⚠️  No index found, need to build first")
            except Exception as e:
                logger.error(f"❌ Error loading Chroma store: {e}")
                self.store = None
    
    def _get_langchain_embedding_function(self):
        """Crea una función de embedding compatible con LangChain"""
        class CustomEmbeddingFunction:
            def __init__(self, embedder):
                self.embedder = embedder
            
            def embed_query(self, text: str) -> List[float]:
                return self.embedder.embed_query(text)
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return self.embedder.embed_documents(texts)
        
        return CustomEmbeddingFunction(self.embedder)

    def _check_ml_capabilities(self):
        """Verifica si el índice Chroma tiene capacidades ML"""
        try:
            if hasattr(self.store, '_collection'):
                collection = self.store._collection
                metadata = collection.metadata or {}
                
                if metadata.get("ml_enhanced") == "true":
                    logger.info("✅ Índice Chroma tiene capacidades ML habilitadas")
                    self.use_product_embeddings = True
                    
                    # Contar documentos con embeddings ML
                    sample = collection.get(limit=100)
                    if sample and sample.get('metadatas'):
                        ml_count = sum(1 for m in sample['metadatas'] 
                                     if m.get('has_embedding', False))
                        logger.info(f"📊 {ml_count}/{len(sample['metadatas'])} documentos tienen embeddings ML")
                
        except Exception as e:
            logger.debug(f"No se pudo verificar capacidades ML: {e}")

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
        """Carga pesos aprendidos de feedback positivo - VERSIÓN COMPLETA"""
        weights = {}
        try:
            # 1. Cargar desde archivo persistente
            weights_file = Path("data/feedback/feedback_weights.json")
            if weights_file.exists():
                with open(weights_file, 'r', encoding='utf-8') as f:
                    weights = json.load(f)
            
            # 2. Enriquecer con datos de success_queries.log
            success_log = Path("data/feedback/success_queries.log")
            if success_log.exists():
                with open(success_log, 'r', encoding='utf-8') as f:
                    for line in f:
                        record = json.loads(line)
                        product_id = record.get('selected_product_id')
                        rating = record.get('feedback', 0)
                        if product_id and rating >= 4:  # Solo feedback positivo
                            weights[product_id] = weights.get(product_id, 0) + 1.0
            
            logger.info(f"✅ Cargados {len(weights)} pesos de feedback positivo")
            
        except Exception as e:
            logger.error(f"Error cargando pesos de feedback: {e}")
        
        return weights

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None,
        min_similarity: float = 0.3,
        top_k: Optional[int] = None,
        use_ml_embeddings: Optional[bool] = None
    ):
        """Compatibilidad con deepeval — ahora con deduplicación avanzada."""
        try:
            # --- compatibilidad ---
            if top_k is not None:
                k = top_k
            
            # 🔥 Nuevo → uso de embeddings ML
            use_ml = use_ml_embeddings if use_ml_embeddings is not None else self.use_product_embeddings

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

            # 🔥 Aplicar filtros si existen
            if filters:
                products = [p for p in products if self._matches_all_filters(p, filters)]

            # =================================================================================
            # 🔥🔥 **A) MEJORA SOLICITADA — DEDUPLICAR USANDO content_hash**
            # =================================================================================
            seen_hashes = set()
            unique_products = []

            for p in products:
                if hasattr(p, "content_hash") and p.content_hash:
                    if p.content_hash in seen_hashes:
                        continue  # ← duplicado → descartar
                    seen_hashes.add(p.content_hash)
                unique_products.append(p)

            products = unique_products  # ← nueva lista limpia
            # =================================================================================

            scored = []
            for p in products:
                # 🔥 Si hay embeddings ML activos usa similitud vectorial
                base_score = (
                    self._calculate_similarity_with_embeddings(query, p)
                    if use_ml else self._score(query, p)
                )

                if base_score < min_similarity:
                    continue

                feedback_boost = self.feedback_weights.get(p.id, 0) * 0.1
                scored.append((base_score + feedback_boost, p))

            scored.sort(key=lambda x: x[0], reverse=True)

            if scored:
                method = "ML Embeddings" if use_ml else "Text Similarity"
                logger.info(f"[Retriever] Returning {min(k,len(scored))} objects using {method}")

                for i, (score, p) in enumerate(scored[:3]):
                    logger.debug(f" {i+1}. {p.title[:50]}... score={score:.3f}")

                return [p for _, p in scored[:k]]

            logger.warning("[Retriever] No products passed scoring threshold")
            return []

        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            import traceback
            logger.error(traceback.format_exc())
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

            # 🔥 NUEVO: Extraer información ML de metadata
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

            # 🔥 NUEVO: Recuperar embeddings ML si existen
            if doc.metadata.get("has_embedding"):
                product_data["ml_processed"] = True
                
                # Intentar recuperar embedding serializado
                if doc.metadata.get("product_embedding"):
                    try:
                        import base64
                        import pickle
                        embedding_str = doc.metadata.get("product_embedding")
                        serialized = base64.b64decode(embedding_str.encode('utf-8'))
                        embedding = pickle.loads(serialized)
                        product_data["embedding"] = embedding.tolist()
                        product_data["embedding_model"] = doc.metadata.get("embedding_model", "unknown")
                    except Exception as e:
                        logger.debug(f"No se pudo recuperar embedding ML: {e}")
            
            # 🔥 NUEVO: Recuperar otras informaciones ML
            if doc.metadata.get("predicted_category"):
                product_data["predicted_category"] = doc.metadata.get("predicted_category")
            
            if doc.metadata.get("extracted_entities"):
                try:
                    entities = safe_json_load(doc.metadata.get("extracted_entities"), {})
                    product_data["extracted_entities"] = entities
                except:
                    pass

            # 🔥 AÑADE ESTA LÍNEA DE LOGGING PARA DEBUG:
            logger.debug(f"[_doc_to_product] Creando Product con id: {product_data['id']}")
            
            return Product.from_dict(product_data, ml_enrich=False)  # Ya tiene ML procesado

        except Exception as e:
            logger.error(f"❌ Failed to convert doc -> product: {e}")
            return None

    def _score(self, query: str, product: Product) -> float:
        try:
            # VERIFICA que product tenga los atributos necesarios
            if not hasattr(product, 'title') or not product.title:
                logger.debug(f"[_score] Product {getattr(product, 'id', 'unknown')} no tiene título")
                return 0.1
            
            # Verificar que realmente sea un Product
            product_type = type(product).__name__
            if product_type != 'Product':
                logger.warning(f"[_score] Object is not Product but {product_type}")
            
            text_sim = SequenceMatcher(None, query.lower(), product.title.lower()).ratio()
            
            rating_boost = 1.0
            if hasattr(product, "average_rating") and product.average_rating:
                rating_boost += product.average_rating / 10.0

            final_score = float(text_sim * rating_boost)
            logger.debug(f"[_score] Score for '{product.id}': {final_score:.3f} (text_sim: {text_sim:.3f})")
            
            return final_score

        except Exception as e:
            logger.error(f"[_score] Error scoring product: {e}")
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
    def _force_clear_index_safe(self):
        """Método ultra-seguro para limpiar índice bloqueado."""
        import time
        import shutil
        
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                if not self.index_path.exists():
                    return True
                
                # Intentar usar handle_windows para Windows
                if os.name == 'nt':  # Windows
                    import subprocess
                    
                    # Usar handle.exe para encontrar y cerrar handles
                    try:
                        subprocess.run(
                            ['handle.exe', str(self.index_path)], 
                            capture_output=True, 
                            text=True
                        )
                    except:
                        pass
                
                # Esperar más tiempo
                time.sleep(retry_delay)
                
                # Limpiar con shutil ignorando errores
                shutil.rmtree(self.index_path, ignore_errors=True)
                
                # Verificar si realmente se eliminó
                if not self.index_path.exists():
                    logger.info(f"✅ Índice eliminado en intento {attempt + 1}")
                    return True
                    
            except Exception as e:
                logger.warning(f"Intento {attempt + 1} falló: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Esperar más cada vez
        
        # Último recurso: mover en lugar de eliminar
        try:
            import datetime
            backup_name = f"{self.index_path}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(self.index_path, backup_name)
            logger.info(f"📦 Índice movido a backup: {backup_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Falló incluso el backup: {e}")
            raise RuntimeError(f"No se pudo limpiar índice bloqueado: {e}")
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
    def __del__(self):
        """Destructor para asegurar que Chroma se cierre correctamente."""
        try:
            if hasattr(self, 'store') and self.store:
                # Forzar cierre de Chroma
                if hasattr(self.store, '_client'):
                    try:
                        self.store._client = None
                    except:
                        pass
                if hasattr(self.store, '_collection'):
                    try:
                        self.store._collection = None
                    except:
                        pass
                self.store = None
        except:
            pass

    def close(self):
        """Cierra explícitamente la conexión Chroma."""
        try:
            self.__del__()
            logger.info("✅ Conexión Chroma cerrada explícitamente")
        except Exception as e:
            logger.warning(f"⚠️ Error cerrando Chroma: {e}")
    # ----------------------------------------------------------------------
    # build_index actualizado
    # ----------------------------------------------------------------------
    def build_index(self, products: List[Product], batch_size: int = 1000):
        try:
            if not products:
                raise ValueError("No products provided to build index")

            logger.info(f"Building index with {len(products)} products")

            # 🔥 NUEVO: Cerrar conexión existente si hay
            self.close()

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

            # 🔥 NUEVO: Crear Chroma en un contexto separado
            if CHROMA_NEW:
                # Versión nueva: NO usa embedding_function aquí
                self.store = Chroma.from_documents(
                    documents=documents,
                    persist_directory=str(self.index_path),
                    collection_metadata={"hnsw:space": "cosine", "ml_enhanced": "true"}
                )
            else:
                # Versión vieja: usar embedding function personalizada
                self.store = Chroma.from_documents(
                    documents=documents,
                    embedding_function=self._get_langchain_embedding_function(),
                    persist_directory=str(self.index_path),
                    collection_metadata={"hnsw:space": "cosine", "ml_enhanced": "true"}
                )

            logger.info("✅ Index built successfully")

        except Exception as e:
            logger.error(f"❌ Index build failed: {e}")
            # Asegurar cierre incluso en error
            self.close()
            raise RuntimeError(f"Index build failed: {e}")
        
    def update_feedback_weights_immediately(self, selected_product_id: str, rating: int, all_shown_products: List[str] = None):
        """Actualiza pesos de feedback con soft negative filtering"""
        try:
            if rating >= 4:  # ✅ Feedback positivo
                current_weight = self.feedback_weights.get(selected_product_id, 0)
                self.feedback_weights[selected_product_id] = current_weight + 1.0
                
                # Ligero boost a productos mostrados
                if all_shown_products:
                    for product_id in all_shown_products:
                        if product_id != selected_product_id:
                            self.feedback_weights[product_id] = self.feedback_weights.get(product_id, 0) + 0.1
                
            else:  # ❌ Feedback negativo - SOFT NEGATIVE FILTERING
                current_weight = self.feedback_weights.get(selected_product_id, 0)
                # ✅ Penalización suave con clipping
                new_weight = current_weight - 1.0
                self.feedback_weights[selected_product_id] = max(new_weight, -3.0)  # Clipping en -3
            
            # ✅ Aplicar decay temporal periódicamente
            self._apply_temporal_decay()
            
            # Guardar pesos actualizados
            self._save_feedback_weights()
            
        except Exception as e:
            logger.error(f"Error actualizando pesos de feedback: {e}")

    def _apply_temporal_decay(self):
        """Aplica decay temporal a pesos antiguos (half-life de 30 días)"""
        try:
            current_time = time.time()
            if not hasattr(self, '_last_decay_time'):
                self._last_decay_time = current_time
            
            # Aplicar decay cada 24 horas
            if current_time - self._last_decay_time < 86400:
                return
            
            decay_factor = 0.95  # 5% de decay cada día
            for product_id in list(self.feedback_weights.keys()):
                self.feedback_weights[product_id] *= decay_factor
            
            # Filtrar pesos muy bajos
            self.feedback_weights = {k: v for k, v in self.feedback_weights.items() 
                                if abs(v) > 0.01}
            
            self._last_decay_time = current_time
            logger.debug("🕒 Aplicado decay temporal a pesos de feedback")
            
        except Exception as e:
            logger.error(f"Error aplicando decay temporal: {e}")

    def _save_feedback_weights(self):
        """Guarda pesos con límite de tamaño"""
        try:
            weights_file = Path("data/feedback/feedback_weights.json")
            weights_file.parent.mkdir(parents=True, exist_ok=True)
            
            # ✅ LIMITAR TAMAÑO: mantener solo top 1000 productos
            sorted_weights = sorted(self.feedback_weights.items(), 
                                key=lambda x: abs(x[1]), reverse=True)
            limited_weights = dict(sorted_weights[:1000])
            
            with open(weights_file, 'w', encoding='utf-8') as f:
                json.dump(limited_weights, f, ensure_ascii=False, indent=2)
                
            logger.debug(f"💾 Pesos guardados: {len(limited_weights)} productos (limitado a 1000)")
                
        except Exception as e:
            logger.error(f"Error guardando pesos de feedback: {e}")

    def _update_feedback_weights(self, doc_ids: List[str], positive: bool = True):
        """Actualiza pesos de feedback de forma segura y con clipping"""
        try:
            weight_change = 0.1 if positive else -0.1

            for doc_id in doc_ids:
                # Asegurar que doc_id es string
                doc_id_str = str(doc_id)

                # Obtener peso actual y asegurar que es numérico
                current_weight = self.feedback_weights.get(doc_id_str, 0.0)
                if not isinstance(current_weight, (int, float)):
                    current_weight = 0.0

                # Actualizar con clipping entre 0 y 1
                new_weight = max(0.0, min(1.0, current_weight + weight_change))
                self.feedback_weights[doc_id_str] = new_weight

            # 🔹 Guardar inmediatamente con limitación top 1000
            self._save_feedback_weights()

        except Exception as e:
            logger.error(f"❌ Error actualizando pesos de feedback: {e}")
            # No romper el flujo