# src/core/rag/basic/retriever.py
from __future__ import annotations
import os
import json
import uuid
import time
import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from difflib import SequenceMatcher
import numpy as np 

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Intentar importar Chroma con diferentes versiones
try:
    from langchain_chroma import Chroma
    CHROMA_NEW = True
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
        CHROMA_NEW = False
    except ImportError as e:
        raise ImportError(
            "No se pudo importar Chroma. "
            "Instala `langchain-chroma` o `langchain-community`."
        ) from e

# Importar modelos de embeddings
from sentence_transformers import SentenceTransformer

# Intentar importar HuggingFaceEmbeddings pero manejar si no está disponible
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    HAS_HUGGINGFACE = True
except ImportError:
    HAS_HUGGINGFACE = False
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.core.rag.basic.retriever import Retriever
    from langchain_huggingface import HuggingFaceEmbeddings

# Y actualizar la importación condicional:
try:
    from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddings
    HAS_HUGGINGFACE = True
    HuggingFaceEmbeddings = HFEmbeddings  # Para type checking
except ImportError:
    HAS_HUGGINGFACE = False
    HuggingFaceEmbeddings = None  #
from src.core.utils.logger import get_logger
from src.core.data.product import Product, ProductDetails
from src.core.config import settings

logger = get_logger(__name__)

# ----------------------------------------------------------------------
# Synonyms & helpers
# ----------------------------------------------------------------------

_SYNONYMS: Dict[str, List[str]] = {
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
    "juego": ["game", "video game", "videogame", "gaming"],
    "nintendo": ["switch", "wii", "gamecube", "nes", "snes", "n64"],
    "playstation": ["ps4", "ps5", "sony playstation"],
    "xbox": ["xbox one", "xbox series", "microsoft xbox"],
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
class LocalEmbedder(Embeddings):
    """Embedder local que implementa la interfaz Embeddings de LangChain"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.model: Optional[Union[SentenceTransformer, Any, DummyEmbedder]] = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa el modelo localmente"""
        try:
            logger.info(f"🔧 Inicializando embedder local: {self.model_name}")
            
            if HAS_HUGGINGFACE:
                # Importar aquí para evitar variable unbound
                from langchain_huggingface import HuggingFaceEmbeddings
                self.model = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": self.device},
                    encode_kwargs={
                        "normalize_embeddings": True,
                        "batch_size": 32
                    }
                )
                logger.info(f"✅ Embedder local inicializado con HuggingFaceEmbeddings: {self.model_name}")
            else:
                # Usar SentenceTransformer directamente
                self.model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"✅ Embedder local inicializado con SentenceTransformer: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando embedder local {self.model_name}: {e}")
            # Crear un modelo dummy como último recurso
            self.model = DummyEmbedder()
    
    def embed_query(self, text: str) -> List[float]:
        """Genera embedding para una query"""
        try:
            if self.model is None:
                self._initialize_model()
            
            if self.model is None:
                logger.error("❌ Modelo aún es None después de inicialización")
                return [0.0] * 384
            
            if HAS_HUGGINGFACE and hasattr(self.model, 'embed_query'):
                # Type safe para HuggingFaceEmbeddings
                return self.model.embed_query(text)
            elif isinstance(self.model, SentenceTransformer):
                # Usar SentenceTransformer directamente
                embedding = self.model.encode(text, normalize_embeddings=True)
                
                # Convertir a lista de floats de forma segura
                return self._convert_to_float_list(embedding)
            else:
                # Si es DummyEmbedder u otro tipo
                return self.model.embed_query(text)
                
        except Exception as e:
            logger.error(f"❌ Error generando embedding: {e}")
            return [0.0] * 384
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings para múltiples documentos"""
        try:
            if self.model is None:
                self._initialize_model()
            
            if self.model is None:
                logger.error("❌ Modelo aún es None después de inicialización")
                return [[0.0] * 384 for _ in texts]
            
            if HAS_HUGGINGFACE and hasattr(self.model, 'embed_documents'):
                return self.model.embed_documents(texts)
            elif isinstance(self.model, SentenceTransformer):
                embeddings = self.model.encode(texts, normalize_embeddings=True)
                
                # Convertir a List[List[float]]
                return self._convert_batch_to_float_lists(embeddings)
            else:
                return self.model.embed_documents(texts)
                
        except Exception as e:
            logger.error(f"❌ Error generando embeddings de documentos: {e}")
            return [[0.0] * 384 for _ in texts]
    
    def _convert_to_float_list(self, embedding: Any) -> List[float]:
        """Convierte cualquier tipo de embedding a List[float]."""
        try:
            if isinstance(embedding, list):
                return [float(x) for x in embedding]
            elif isinstance(embedding, np.ndarray):
                return embedding.tolist()
            elif hasattr(embedding, 'cuda'):
                # Tensor de PyTorch
                return embedding.cuda().numpy().tolist()
            elif hasattr(embedding, 'tolist'):
                return embedding.tolist()
            else:
                # Intentar conversión directa
                return [float(x) for x in embedding]
        except Exception as e:
            logger.warning(f"⚠️ Error convirtiendo embedding a lista: {e}")
            return [0.0] * 384
    
    def _convert_batch_to_float_lists(self, embeddings: Any) -> List[List[float]]:
        """Convierte un batch de embeddings a List[List[float]]."""
        try:
            if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
                return embeddings.tolist()
            elif isinstance(embeddings, list):
                # Ya es una lista, convertir cada elemento
                result = []
                for emb in embeddings:
                    result.append(self._convert_to_float_list(emb))
                return result
            elif hasattr(embeddings, 'tolist'):
                # Intentar tolist primero
                emb_list = embeddings.tolist()
                if isinstance(emb_list, list) and all(isinstance(x, list) for x in emb_list):
                    return emb_list
                else:
                    # Wrap en lista adicional si es necesario
                    return [self._convert_to_float_list(embeddings)]
            else:
                # Último recurso
                return [[0.0] * 384 for _ in range(len(embeddings) if hasattr(embeddings, '__len__') else 1)]
        except Exception as e:
            logger.warning(f"⚠️ Error convirtiendo batch de embeddings: {e}")
            return [[0.0] * 384 for _ in range(10)]

# ----------------------------------------------------------------------
# Dummy Embedder como fallback
# ----------------------------------------------------------------------
class DummyEmbedder:
    """Embedder dummy como último recurso"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def embed_query(self, text: str) -> List[float]:
        """Embedding dummy para query"""
        import random
        random.seed(hash(text) % 1000000)
        embedding = [random.gauss(0, 1) for _ in range(self.dimension)]
        # Normalizar
        norm = sum(x**2 for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeddings dummy para documentos"""
        return [self.embed_query(text) for text in texts]

# ----------------------------------------------------------------------
# Retriever
# ----------------------------------------------------------------------
class Retriever:
    def __init__(
        self,
        index_path: Union[str, Path] = settings.VECTOR_INDEX_PATH,
        embedding_model: str = "all-MiniLM-L6-v2",  # Siempre local
        device: str = getattr(settings, "DEVICE", "cuda"),
        # 🔥 Ahora configurable globalmente — toma ML si está habilitado
        use_product_embeddings: bool = getattr(settings, "ML_ENABLED", False)
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
        self._last_decay_time = time.time()

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
                if isinstance(product.embedding, list):
                    product_embedding = np.array(product.embedding, dtype=np.float32)
                elif isinstance(product.embedding, np.ndarray):
                    product_embedding = product.embedding.astype(np.float32)
                else:
                    return self._score(query, product)
                
                # Generar embedding para la query
                query_embedding_array = self.embedder.embed_query(query)
                if isinstance(query_embedding_array, list):
                    query_embedding = np.array(query_embedding_array, dtype=np.float32)
                else:
                    query_embedding = query_embedding_array.astype(np.float32)
                
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
                    # Tanto la versión nueva como vieja usan 'embedding_function'
                    self.store = Chroma(
                        persist_directory=str(self.index_path),
                        embedding_function=self.embedder  # Usar directamente el embedder
                    )
                    logger.info("✅ Chroma store loaded successfully")
                    
                    # 🔥 NUEVO: Verificar si el índice tiene metadata ML
                    self._check_ml_capabilities()
                    
                else:
                    logger.warning("⚠️  No index found, need to build first")
            except Exception as e:
                logger.error(f"❌ Error loading Chroma store: {e}")
                self.store = None

    def _check_ml_capabilities(self):
        """Verifica si el índice Chroma tiene capacidades ML"""
        try:
            if self.store and hasattr(self.store, '_collection'):
                collection = self.store._collection
                metadata = collection.metadata or {}
                
                if metadata.get("ml_enhanced") == "true":
                    logger.info("✅ Índice Chroma tiene capacidades ML habilitadas")
                    self.use_product_embeddings = True
                    
                    # Contar documentos con embeddings ML
                    sample = collection.get(limit=100)
                    if sample and sample.get('metadatas'):
                        metadatas = sample['metadatas'] or []
                        ml_count = sum(1 for m in metadatas 
                                     if m and isinstance(m, dict) and m.get('has_embedding', False))
                        logger.info(f"📊 {ml_count}/{len(metadatas)} documentos tienen embeddings ML")
                
        except Exception as e:
            logger.debug(f"No se pudo verificar capacidades ML: {e}")

    # Compatible con LangChain retriever
    def as_retriever(self, search_kwargs=None):
        self._ensure_store_loaded()
        if self.store:
            return self.store.as_retriever(search_kwargs=search_kwargs)
        return None

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
                        try:
                            record = json.loads(line)
                            product_id = record.get('selected_product_id')
                            rating = record.get('feedback', 0)
                            if product_id and rating >= 4:  # Solo feedback positivo
                                weights[str(product_id)] = weights.get(str(product_id), 0) + 1.0
                        except json.JSONDecodeError:
                            continue
            
            logger.info(f"✅ Cargados {len(weights)} pesos de feedback positivo")
            
        except Exception as e:
            logger.error(f"Error cargando pesos de feedback: {e}")
        
        return weights

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None,
        min_similarity: float = 0.15,  # 🔥 Aumentado de 0.1 a 0.15
        top_k: Optional[int] = None,
        use_ml_embeddings: Optional[bool] = None
    ) -> List[Product]:
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

            # 🔥 MEJORA: Extraer categoría esperada de la consulta
            expected_category = self._extract_expected_category(query)
            if expected_category:
                if filters is None:
                    filters = {}
                filters['main_category'] = expected_category
                logger.debug(f"[Retrieve] Filtro por categoría esperada: {expected_category}")
            
            if filters is None:
                filters = self._parse_filters_from_query(query)

            expanded = self._expand_query(query)
            docs = self._raw_retrieve(" ".join(expanded), k=k * 3, filters=filters)  # 🔥 Buscar más inicialmente

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

            # 🔥 ORDENAR por relevancia antes de deduplicar
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
            
            # Ordenar por score
            scored.sort(key=lambda x: x[0], reverse=True)
            
            # 🔥 NUEVO: Filtrar duplicados después de ordenar
            seen_hashes = set()
            unique_scored = []
            
            for score, p in scored:
                if hasattr(p, "content_hash") and p.content_hash:
                    if p.content_hash in seen_hashes:
                        continue
                    seen_hashes.add(p.content_hash)
                unique_scored.append((score, p))
            
            scored = unique_scored

            if scored:
                method = "ML Embeddings" if use_ml else "Text Similarity"
                logger.info(f"[Retriever] Returning {min(k,len(scored))} objects using {method}")

                # 🔥 MEJORA: Logging más informativo
                for i, (score, p) in enumerate(scored[:5]):
                    category = getattr(p, 'main_category', 'Unknown')
                    predicted = getattr(p, 'predicted_category', 'None')
                    logger.debug(f" {i+1}. {p.title[:50]}... score={score:.3f}, "
                            f"cat={category}, pred={predicted}")

                return [p for _, p in scored[:k]]

            logger.warning("[Retriever] No products passed scoring threshold")
            return []

        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    # 🔥 AÑADIR NUEVO MÉTODO PARA EXTRACCIÓN DE CATEGORÍA
    def _extract_expected_category(self, query: str) -> Optional[str]:
        """Extrae la categoría esperada basada en palabras clave en la consulta"""
        query_lower = query.lower()
        
        # Mapeo de palabras clave a categorías
        keyword_to_category = {
            # Video Games
            'juego': 'Video Games',
            'nintendo': 'Video Games',
            'playstation': 'Video Games',
            'xbox': 'Video Games',
            'consola': 'Video Games',
            'videojuego': 'Video Games',
            
            # Electronics
            'laptop': 'Electronics',
            'computador': 'Electronics',
            'auricular': 'Electronics',
            'auriculares': 'Electronics',
            'altavoz': 'Electronics',
            'cámara': 'Electronics',
            'smartphone': 'Electronics',
            'teléfono': 'Electronics',
            
            # Clothing
            'zapato': 'Clothing',
            'ropa': 'Clothing',
            'vestido': 'Clothing',
            'camisa': 'Clothing',
            'pantalón': 'Clothing',
            
            # Home & Kitchen
            'sofá': 'Home & Kitchen',
            'mueble': 'Home & Kitchen',
            'cocina': 'Home & Kitchen',
            'electrodoméstico': 'Home & Kitchen',
            'nevera': 'Home & Kitchen',
            
            # Books
            'libro': 'Books',
            'novela': 'Books',
            'cuento': 'Books',
            
            # Beauty
            'crema': 'Beauty',
            'maquillaje': 'Beauty',
            'cosmético': 'Beauty',
            'perfume': 'Beauty',
            
            # Sports & Outdoors
            'bicicleta': 'Sports & Outdoors',
            'balón': 'Sports & Outdoors',
            'deporte': 'Sports & Outdoors',
            'gimnasio': 'Sports & Outdoors',
            
            # Toys & Games
            'juguete': 'Toys & Games',
            'juego de mesa': 'Toys & Games',
            
            # Office Products
            'impresora': 'Office Products',
            'papel': 'Office Products',
            'oficina': 'Office Products',
            
            # Automotive
            'herramienta': 'Automotive',
            'coche': 'Automotive',
            'auto': 'Automotive',
            
            # Health
            'vitamina': 'Health & Personal Care',
            'medicina': 'Health & Personal Care',
            'suplemento': 'Health & Personal Care'
        }
        
        # Buscar palabras clave
        for keyword, category in keyword_to_category.items():
            if keyword in query_lower:
                return category
        
        return None


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
            # SIMPLIFICACIÓN: Buscar solo en el texto del producto
            search_text = ""
            if product.title:
                search_text += product.title.lower() + " "
            if product.description:
                search_text += product.description.lower() + " "
            
            # También buscar en detalles si están disponibles
            if hasattr(product, "details") and product.details:
                if isinstance(product.details, dict):
                    search_text += " ".join(str(v).lower() for v in product.details.values() if v) + " "
                elif hasattr(product.details, "__dict__"):
                    search_text += " ".join(str(v).lower() for v in product.details.__dict__.values() if v) + " "
            
            filter_color = filters["color"].lower()
            
            # Mapear sinónimos de colores
            color_synonyms = {
                "rojo": ["rojo", "red", "colorado"],
                "azul": ["azul", "blue"],
                "verde": ["verde", "green"],
                "negro": ["negro", "black"],
                "blanco": ["blanco", "white", "blanca"],
                "amarillo": ["amarillo", "yellow"],
                "rosa": ["rosa", "pink", "rosado"],
                "gris": ["gris", "gray", "grey"]
            }
            
            # Verificar si el color o sus sinónimos están en el texto
            color_found = False
            if filter_color in color_synonyms:
                for synonym in color_synonyms[filter_color]:
                    if synonym in search_text:
                        color_found = True
                        break
            else:
                # Color no mapeado, buscar directamente
                color_found = filter_color in search_text
            
            if not color_found:
                return False

        if "wireless" in filters:
            text = ""
            if product.title:
                text += product.title.lower()
            if product.description:
                text += product.description.lower()
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
                    if isinstance(x, str):
                        return json.loads(x)
                    elif isinstance(x, (list, dict)):
                        return x
                    else:
                        return default
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
                embedding_str = doc.metadata.get("product_embedding")
                if embedding_str:  # CORRECCIÓN: Verificar que no sea None
                    try:
                        import base64
                        import pickle
                        # CORRECCIÓN: embedding_str ya es string
                        serialized = base64.b64decode(embedding_str)
                        embedding = pickle.loads(serialized)
                        
                        # Convertir a lista de floats de forma segura
                        product_data["embedding"] = self._convert_any_to_float_list(embedding)
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

            return Product.from_dict(product_data, ml_enrich=False)
            
        except Exception as e:
            logger.error(f"❌ Failed to convert doc -> product: {e}")
            return None
    def _convert_any_to_float_list(self, embedding: Any) -> List[float]:
        """Convierte cualquier tipo de embedding a lista de floats."""
        try:
            if isinstance(embedding, list):
                return [float(x) for x in embedding]
            elif isinstance(embedding, np.ndarray):
                return embedding.tolist()
            elif hasattr(embedding, 'cuda'):
                # Tensor de PyTorch
                return embedding.cuda().numpy().tolist()
            elif hasattr(embedding, 'tolist'):
                return embedding.tolist()
            else:
                # Intentar conversión directa
                return [float(x) for x in embedding]
        except Exception as e:
            logger.warning(f"Error convirtiendo embedding: {e}")
            return []
    def _score(self, query: str, product: Product) -> float:
        try:
            if not hasattr(product, 'title') or not product.title:
                return 0.1
            
            query_lower = query.lower()
            title_lower = product.title.lower()
            desc_lower = (product.description or "").lower()
            
            # 🔥 MEJORA: Ponderaciones para diferentes factores
            weights = {
                'title_match': 0.6,
                'category_match': 0.3,
                'description_match': 0.1
            }
            
            # 1. Similitud en título
            title_score = SequenceMatcher(None, query_lower, title_lower).ratio()
            
            # Bonus por palabras exactas en título
            exact_title_bonus = 0
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 3 and word in title_lower:
                    exact_title_bonus += 0.2
            
            # 2. Coincidencia de categoría
            category_score = 0
            if hasattr(product, 'main_category') and product.main_category:
                category_lower = product.main_category.lower()
                # Verificar si palabras de la consulta están en la categoría
                category_words = set(category_lower.split())
                query_words_set = set(query_words)
                common_words = category_words.intersection(query_words_set)
                if common_words:
                    category_score = len(common_words) / len(query_words_set)
            
            # 3. Similitud en descripción (limitada)
            desc_score = 0
            if desc_lower and len(desc_lower) > 50:
                desc_match = SequenceMatcher(None, query_lower, desc_lower[:200]).ratio()
                desc_score = min(desc_match, 0.5)  # Limitar influencia
            
            # 🔥 NUEVO: Bonus por categorías específicas
            category_bonus = 1.0
            if hasattr(product, 'predicted_category') and product.predicted_category:
                pred_cat = product.predicted_category.lower()
                # Mapeo de consultas a categorías esperadas
                query_to_category = {
                    'juego': ['video games', 'toys & games'],
                    'consola': ['video games', 'electronics'],
                    'laptop': ['electronics', 'computers'],
                    'zapato': ['clothing', 'fashion'],
                    'libro': ['books'],
                    'crema': ['beauty', 'health & personal care'],
                    'sofá': ['home & kitchen'],
                    'bicicleta': ['sports & outdoors'],
                    'herramienta': ['tools & home improvement', 'automotive'],
                    'impresora': ['office products', 'electronics'],
                    'vitamina': ['health & personal care', 'health'],
                    'auricular': ['electronics'],
                    'vestido': ['clothing', 'fashion'],
                    'cocina': ['home & kitchen'],
                    'balón': ['sports & outdoors']
                }
                
                for keyword, expected_categories in query_to_category.items():
                    if keyword in query_lower:
                        expected_lower = [ec.lower() for ec in expected_categories]
                        if pred_cat in expected_lower:
                            category_bonus += 0.5
                        break
            
            # Calcular puntuación final
            base_score = (
                (title_score + min(exact_title_bonus, 0.5)) * weights['title_match'] +
                category_score * weights['category_match'] +
                desc_score * weights['description_match']
            )
            
            final_score = base_score * category_bonus
            
            # Bonus por rating
            if hasattr(product, "average_rating") and product.average_rating:
                rating_boost = 1.0 + (product.average_rating * 0.1)
                final_score *= rating_boost
            
            # Asegurar que esté entre 0 y 1
            final_score = max(0.0, min(1.0, final_score))
            
            logger.debug(f"[_score] {product.title[:30]}...: title={title_score:.3f}, "
                    f"category={category_score:.3f}, final={final_score:.3f}")
            
            return final_score

        except Exception as e:
            logger.debug(f"[_score] Error: {e}")
            return 0.1

    def _chroma_filter(self, f: Optional[Dict]) -> Dict:
        if not f:
            return {}
        
        result = {}
        for k, v in f.items():
            if isinstance(v, list):
                result[k] = {"$in": v}
            else:
                result[k] = {"$eq": v}
        return result

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
                    self._force_clear_index_safe()
            except Exception as e:
                logger.error(f"❌ Unexpected error clearing index: {e}")
                raise

    def __del__(self):
        """Destructor para asegurar que Chroma se cierre correctamente."""
        try:
            if hasattr(self, 'store') and self.store:
                # Limpiar referencias
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

            # Cerrar conexión existente
            self.close()

            # Limpiar directorio existente
            if self.index_path.exists():
                self._safe_clear_index()
                logger.info("🗑️  Existing index cleared")

            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            # Crear documentos
            documents = []
            for p in products:
                if p.title and p.title.strip():
                    try:
                        doc = Document(
                            page_content=p.to_text(),
                            metadata=p.to_metadata()
                        )
                        documents.append(doc)
                    except Exception as e:
                        logger.warning(f"Error creating document for product {p.id}: {e}")

            logger.info(f"📝 Created {len(documents)} documents")

            # 🔥 CORRECCIÓN: Usar self.embedder directamente
            # El embedder ya implementa la interfaz Embeddings de LangChain
            self.store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedder,  # Usar embedder directamente
                persist_directory=str(self.index_path),
                collection_metadata={
                    "hnsw:space": "cosine", 
                    "ml_enhanced": "true",
                    "builder": "retriever_build_index"
                }
            )

            logger.info("✅ Index built successfully")

        except Exception as e:
            logger.error(f"❌ Index build failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.close()
            raise RuntimeError(f"Index build failed: {e}")
    
    def update_feedback_weights_immediately(self, selected_product_id: str, rating: int, all_shown_products: Optional[List[str]] = None):
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