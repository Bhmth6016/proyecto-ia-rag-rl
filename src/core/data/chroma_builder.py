# src/core/data/chroma_builder.py - VERSIÓN FINAL CORREGIDA

import os
import json
import logging
import shutil
import time
import pickle
import base64
import threading
import psutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Sequence, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import gc

from tqdm import tqdm
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

from src.core.data.product import Product
from src.core.config import settings
from src.core.utils.logger import get_logger
from src.core.data.loader import FastDataLoader

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Dummy Embedder para fallback
# ------------------------------------------------------------------
class DummyEmbedder:
    """Embedder dummy como último recurso cuando fallan todos los modelos."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.model_name = "dummy_embedder"
        logger.warning(f"⚠️  Usando DummyEmbedder con dimensión {dimension}")
    
    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Genera embeddings dummy (aleatorios normalizados)."""
        import random
        
        embeddings = []
        for text in texts:
            # Embedding aleatorio pero reproducible basado en el texto
            random.seed(hash(text) % 1000000)
            embedding = [random.gauss(0, 1) for _ in range(self.dimension)]
            
            # Normalizar
            norm = sum(x**2 for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
            
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compatibilidad con langchain."""
        return self.encode(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Para queries individuales."""
        return self.encode([text])[0]

# ------------------------------------------------------------------
# Embedding Serializer
# ------------------------------------------------------------------
class EmbeddingSerializer:
    """Serializa/deserializa embeddings para almacenamiento en metadata"""
    
    @staticmethod
    def serialize_embedding(embedding: List[float]) -> str:
        """Convierte embedding a string base64 optimizado."""
        try:
            # Convertir a numpy array de tipo float32 para reducir tamaño
            arr = np.array(embedding, dtype=np.float32)
            
            # Comprimir con compresión simple
            serialized = pickle.dumps(arr, protocol=4)  # Protocolo 4 es más eficiente
            
            # Codificar base64
            return base64.b64encode(serialized).decode('utf-8')
        except Exception as e:
            logger.warning(f"Error serializando embedding: {e}")
            return ""
    
    @staticmethod
    def deserialize_embedding(embedding_str: str) -> Optional[np.ndarray]:
        """Convierte string base64 a embedding numpy array."""
        try:
            if not embedding_str:
                return None
            
            # Decodificar base64
            serialized = base64.b64decode(embedding_str.encode('utf-8'))
            
            # Deserializar
            arr = pickle.loads(serialized)
            
            # Asegurar tipo correcto
            if isinstance(arr, list):
                arr = np.array(arr, dtype=np.float32)
            elif isinstance(arr, np.ndarray):
                arr = arr.astype(np.float32)
            
            return arr
            
        except Exception as e:
            logger.warning(f"Error deserializando embedding: {e}")
            return None
    
    @staticmethod
    def embedding_to_json(embedding: List[float]) -> str:
        """Convierte embedding a JSON string (para debugging)."""
        try:
            # Solo primeros 5 valores para debugging
            truncated = embedding[:5] if len(embedding) > 5 else embedding
            return json.dumps([float(x) for x in truncated], separators=(',', ':'))
        except Exception as e:
            logger.warning(f"Error convirtiendo embedding a JSON: {e}")
            return "[]"
    
    @staticmethod
    def validate_embedding(embedding: Any, expected_dim: Optional[int] = None) -> bool:
        """Valida que un embedding tenga formato correcto."""
        if embedding is None:
            return False
        
        try:
            # Convertir a numpy array
            if isinstance(embedding, list):
                arr = np.array(embedding, dtype=np.float32)
            elif isinstance(embedding, np.ndarray):
                arr = embedding.astype(np.float32)
            elif torch.is_tensor(embedding):
                arr = embedding.cpu().numpy().astype(np.float32)
            else:
                return False
            
            # Validar dimensiones
            if expected_dim and arr.shape[0] != expected_dim:
                logger.warning(f"Embedding dimension mismatch: {arr.shape[0]} != {expected_dim}")
                return False
            
            # Validar que no sea todo ceros o NaNs
            if np.all(arr == 0) or np.any(np.isnan(arr)):
                logger.warning("Embedding contains all zeros or NaNs")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validando embedding: {e}")
            return False
    
    @staticmethod
    def to_list(embedding: Any) -> List[float]:
        """Convierte cualquier tipo de embedding a lista de floats."""
        try:
            if isinstance(embedding, list):
                return [float(x) for x in embedding]
            elif isinstance(embedding, np.ndarray):
                return embedding.tolist()
            elif torch.is_tensor(embedding):
                return embedding.cpu().numpy().tolist()
            else:
                # Intentar convertir
                return list(map(float, embedding))
        except Exception as e:
            logger.warning(f"Error convirtiendo embedding a lista: {e}")
            return []

# ------------------------------------------------------------------
# Clase EmbeddingsWrapper para compatibilidad
# ------------------------------------------------------------------
class EmbeddingsWrapper(Embeddings):
    """Wrapper para cualquier modelo de embeddings para compatibilidad con LangChain."""
    
    def __init__(self, model: Any):
        self.model = model
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documentos."""
        try:
            if hasattr(self.model, 'embed_documents'):
                return self.model.embed_documents(texts)
            elif hasattr(self.model, 'encode'):
                # Para SentenceTransformer
                embeddings = self.model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                if isinstance(embeddings, np.ndarray):
                    return embeddings.tolist()
                return embeddings
            else:
                # Fallback
                dummy = DummyEmbedder(dimension=384)
                return dummy.encode(texts)
        except Exception as e:
            logger.error(f"Error en embed_documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed una query."""
        try:
            if hasattr(self.model, 'embed_query'):
                return self.model.embed_query(text)
            elif hasattr(self.model, 'encode'):
                # Para SentenceTransformer
                embedding = self.model.encode(
                    [text],
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                if isinstance(embedding, np.ndarray):
                    return embedding[0].tolist()
                return embedding[0] if embedding else []
            else:
                # Fallback
                dummy = DummyEmbedder(dimension=384)
                return dummy.encode([text])[0]
        except Exception as e:
            logger.error(f"Error en embed_query: {e}")
            raise

    def __repr__(self) -> str:
        """Representación del wrapper."""
        if hasattr(self.model, 'model_name'):
            return f"EmbeddingsWrapper(model={self.model.model_name})"
        return f"EmbeddingsWrapper(model={type(self.model).__name__})"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
class ChromaBuilderConfig:
    """Configuración para el constructor de Chroma optimizado"""
    
    # Optimizaciones de rendimiento
    DEFAULT_BATCH_SIZE = 1000
    MAX_CONCURRENT_WORKERS = 4
    EMBEDDING_BATCH_SIZE = 64
    DOCUMENT_CACHE_SIZE = 1000
    
    # Configuración de embeddings
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DEVICE = "cuda"
    
    # Límites de memoria
    MAX_DOCUMENTS_PER_BATCH = 500000
    MEMORY_CHECK_INTERVAL = 1000000
    
    # Configuración ML unificada
    @staticmethod
    def get_ml_config() -> Dict[str, Any]:
        """Obtiene configuración ML desde settings global."""
        return {
            'enabled': getattr(settings, 'ML_ENABLED', False),
            'features': list(getattr(settings, 'ML_FEATURES', [])),
            'embedding_model': getattr(settings, 'ML_EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
            'use_gpu': getattr(settings, 'ML_USE_GPU', False),
            'categories': getattr(settings, 'ML_CATEGORIES', []),
            'confidence_threshold': getattr(settings, 'ML_CONFIDENCE_THRESHOLD', 0.7)
        }

# ------------------------------------------------------------------
# Optimized Chroma Builder con Fallbacks Mejorados
# ------------------------------------------------------------------
class OptimizedChromaBuilder:
    def __init__(
        self,
        *,
        processed_json_path: Optional[Path] = None,
        chroma_db_path: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = ChromaBuilderConfig.DEFAULT_BATCH_SIZE,
        max_workers: int = ChromaBuilderConfig.MAX_CONCURRENT_WORKERS,
        enable_cache: bool = True,
        use_product_embeddings: Optional[bool] = None,
        ml_logging: bool = True
    ):
        """
        Constructor optimizado para ChromaDB con soporte ML completo.
        
        Args:
            use_product_embeddings: Si None, usa settings.ML_ENABLED
        """
        from src.core.config import settings
        
        self.processed_json_path = processed_json_path or settings.PROC_DIR / "products.json"
        self.chroma_db_path = chroma_db_path or Path(settings.CHROMA_DB_PATH)
        
        # Configuración ML
        ml_config = ChromaBuilderConfig.get_ml_config()
        self.embedding_model = embedding_model or ml_config['embedding_model']
        
        self.device = device or ChromaBuilderConfig.EMBEDDING_DEVICE
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.enable_cache = enable_cache
        
        # Configuración ML unificada
        self.ml_config = ml_config
        if use_product_embeddings is None:
            self.use_product_embeddings = ml_config['enabled']
        else:
            self.use_product_embeddings = use_product_embeddings
        
        self.ml_logging = ml_logging
        
        # Cache para documentos procesados
        self._document_cache: Dict[str, bool] = {}
        
        # Lazy loading para el modelo
        self._embedding_model: Optional[Any] = None
        self._embedding_model_lock = threading.Lock()
        
        # 🔥 MEJORA: Configurar opciones de fallback
        self._fallback_config = {
            'max_retries': 3,
            'retry_delay': 2,
            'use_dummy_on_failure': True,
            'dummy_dimension': 384
        }
        
        # 🔥 MEJORA: Cache de modelos fallback
        self._model_cache: Dict[str, Any] = {}
        self._model_cache_lock = threading.Lock()
        
        # 🔥 MEJORA: Registro de fallos
        self._model_failures: List[Tuple[str, str]] = []
        
        # Configurar logging ML
        if self.ml_logging:
            self._setup_ml_logging()
        
        # Estadísticas mejoradas
        self._stats: Dict[str, Any] = {
            'total_products': 0,
            'processed_documents': 0,
            'skipped_documents': 0,
            'total_time': 0,
            'embedding_time': 0,
            'products_with_ml': 0,
            'products_with_embedding': 0,
            'ml_embeddings_used': 0,
            'chroma_embeddings_computed': 0,
            'valid_embeddings': 0,
            'invalid_embeddings': 0,
            'model_load_time': 0,
            'model_type': 'unknown',
            'dummy_fallback_used': False
        }
        
        # Metadata del índice
        self._index_metadata = {
            'builder_version': 'fallback_enhanced_v1',
            'ml_enabled': self.use_product_embeddings,
            'embedding_model': self.embedding_model,
            'created_at': time.time(),
            'ml_config': ml_config
        }

    def _get_memory_usage(self) -> float:
        """Obtiene el uso de memoria del proceso actual en MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def cleanup_memory(self):
        """Libera memoria de modelos grandes."""
        self._log_ml_info("🧹 Limpiando memoria del MLProcessor...")
        
        # Limpiar modelo de embeddings
        if hasattr(self, '_embedding_model') and self._embedding_model is not None:
            del self._embedding_model
            self._embedding_model = None
        
        # Limpiar cache de modelos
        with self._model_cache_lock:
            self._model_cache.clear()
        
        # Limpiar cache de documentos
        self._document_cache.clear()
        
        # Limpiar estadísticas
        self._model_failures.clear()
        
        # Forzar garbage collection
        gc.collect()
        
        self._log_ml_info("✅ Memoria liberada")
        
        # Verificar memoria después de limpiar
        self._log_ml_info(f"📊 Memoria RSS después de limpiar: {self._get_memory_usage():.1f}MB")

    def _setup_ml_logging(self):
        """Configura logging específico para ML."""
        try:
            # Crear logger específico para ML
            self.ml_logger = logging.getLogger('ml_chroma_builder')
            if not self.ml_logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s - ML-Chroma - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S'
                ))
                self.ml_logger.addHandler(handler)
                self.ml_logger.setLevel(logging.INFO)
            
        except Exception as e:
            logger.warning(f"Error configurando logging ML: {e}")

    def _log_ml_info(self, message: str):
        """Log message específico para ML."""
        if self.ml_logging:
            if hasattr(self, 'ml_logger'):
                self.ml_logger.info(message)
            else:
                logger.info(f"[ML] {message}")

    # ------------------------------------------------------------------
    # 🔥 MEJORA: Métodos de fallback mejorados
    # ------------------------------------------------------------------
    
    def _load_sentence_transformer(self) -> SentenceTransformer:
        """Carga SentenceTransformer con manejo de errores."""
        try:
            self._log_ml_info(f"🔧 Intentando cargar SentenceTransformer: {self.embedding_model}")
            
            # Opciones de configuración
            model_kwargs = {
                'device': self.device,
                'cache_folder': str(settings.MODELS_DIR / 'sentence_transformers')
            }
            
            # Crear directorio de cache si no existe
            cache_folder = Path(model_kwargs['cache_folder'])
            cache_folder.mkdir(parents=True, exist_ok=True)
            
            model = SentenceTransformer(
                self.embedding_model,
                device=model_kwargs.get('device', 'cpu'),
                cache_folder=str(cache_folder)
            )
            
            # Verificar que el modelo funcione
            test_embedding = model.encode(["test"], convert_to_numpy=True)
            if test_embedding is None or len(test_embedding) == 0:
                raise ValueError("Modelo devolvió embedding vacío")
            
            self._log_ml_info(f"✅ SentenceTransformer cargado: {model}")
            return model
            
        except Exception as e:
            error_msg = f"Error cargando SentenceTransformer: {e}"
            self._model_failures.append(('sentence_transformer', error_msg))
            self._log_ml_info(f"❌ {error_msg}")
            raise
    
    def _load_huggingface_embeddings(self) -> HuggingFaceEmbeddings:
        """Carga HuggingFaceEmbeddings con manejo de errores."""
        try:
            self._log_ml_info(f"🔧 Intentando cargar HuggingFaceEmbeddings: {self.embedding_model}")
            
            model_kwargs = {
                "device": self.device,
                "trust_remote_code": True,
                "cache_dir": str(settings.MODELS_DIR / 'huggingface')
            }
            
            encode_kwargs = {
                "batch_size": 32,
                "normalize_embeddings": True,
                "show_progress_bar": False
            }
            
            model = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            # Verificar que el modelo funcione
            test_embeddings = model.embed_documents(["test"])
            if not test_embeddings or len(test_embeddings[0]) == 0:
                raise ValueError("Modelo devolvió embedding vacío")
            
            self._log_ml_info(f"✅ HuggingFaceEmbeddings cargado")
            return model
            
        except Exception as e:
            error_msg = f"Error cargando HuggingFaceEmbeddings: {e}"
            self._model_failures.append(('huggingface', error_msg))
            self._log_ml_info(f"❌ {error_msg}")
            raise
    
    def _create_dummy_embedder(self) -> DummyEmbedder:
        """Crea un embedder dummy como último recurso."""
        self._log_ml_info("🔄 Creando DummyEmbedder como último recurso...")
        
        # Intentar determinar dimensión basada en el nombre del modelo
        dimension = self._fallback_config['dummy_dimension']
        
        # Algunos modelos comunes y sus dimensiones
        model_dimensions = {
            'all-MiniLM-L6-v2': 384,
            'all-MiniLM-L12-v2': 384,
            'all-mpnet-base-v2': 768,
            'paraphrase-multilingual-MiniLM-L12-v2': 384,
            'distilbert-base-nli-stsb-mean-tokens': 768,
        }
        
        for model_name, model_dim in model_dimensions.items():
            if model_name in self.embedding_model:
                dimension = model_dim
                break
        
        dummy_model = DummyEmbedder(dimension=dimension)
        
        # Registrar uso de dummy
        self._log_ml_info(f"⚠️  Usando DummyEmbedder con dimensión {dimension}")
        logger.warning(f"DummyEmbedder activado después de {len(self._model_failures)} fallos")
        
        return dummy_model
    
    def _load_model_with_fallback(self) -> Any:
        """🔥 MEJORA: Intenta diferentes métodos de carga con reintentos."""
        
        fallback_strategies = [
            ('sentence_transformers', self._load_sentence_transformer),
            ('huggingface', self._load_huggingface_embeddings),
            ('dummy', self._create_dummy_embedder)
        ]
        
        last_error = None
        
        for strategy_name, loader in fallback_strategies:
            # 🔥 MEJORA: Reintentos para cada estrategia
            for attempt in range(self._fallback_config['max_retries']):
                try:
                    if attempt > 0:
                        self._log_ml_info(f"🔄 Reintento {attempt + 1}/{self._fallback_config['max_retries']} para {strategy_name}")
                        time.sleep(self._fallback_config['retry_delay'] * attempt)
                    
                    model = loader()
                    
                    # 🔥 MEJORA: Verificar dimensiones del modelo
                    if strategy_name != 'dummy':
                        self._validate_model_dimensions(model, strategy_name)
                    
                    self._log_ml_info(f"✅ Modelo cargado con {strategy_name} (intento {attempt + 1})")
                    
                    # 🔥 MEJORA: Cachear modelo exitoso
                    with self._model_cache_lock:
                        self._model_cache[strategy_name] = model
                    
                    return model
                    
                except Exception as e:
                    last_error = e
                    error_msg = f"Intento {attempt + 1} con {strategy_name} falló: {str(e)}"
                    self._log_ml_info(f"⚠️  {error_msg}")
                    
                    # No reintentar si es el dummy embedder
                    if strategy_name == 'dummy':
                        break
            
            # Si llegamos aquí, esta estrategia falló completamente
            self._log_ml_info(f"❌ Estrategia {strategy_name} falló después de {self._fallback_config['max_retries']} intentos")
        
        # Si todo falla y dummy no está habilitado
        if not self._fallback_config['use_dummy_on_failure']:
            error_summary = "\n".join([f"{name}: {msg}" for name, msg in self._model_failures])
            raise RuntimeError(
                f"No se pudo cargar ningún modelo de embeddings. Fallos:\n{error_summary}\n"
                f"Último error: {last_error}"
            )
        
        # Último recurso: dummy embedder
        return self._create_dummy_embedder()
    
    def _validate_model_dimensions(self, model: Any, strategy_name: str) -> None:
        """Valida que el modelo tenga dimensiones razonables."""
        try:
            # Obtener dimensión del embedding
            if strategy_name == 'sentence_transformers':
                test_embedding = model.encode(["test"], convert_to_numpy=True)
                dimension = len(test_embedding[0]) if len(test_embedding) > 0 else 0
            elif strategy_name == 'huggingface':
                test_embedding = model.embed_documents(["test"])
                dimension = len(test_embedding[0]) if len(test_embedding) > 0 else 0
            else:
                return
            
            # Validar dimensión
            if dimension < 10 or dimension > 10000:
                raise ValueError(f"Dimensión de embedding inválida: {dimension}")
            
            self._log_ml_info(f"📐 Modelo tiene dimensión {dimension}")
            
        except Exception as e:
            self._log_ml_info(f"⚠️  Error validando dimensiones del modelo: {e}")

    def _get_model_info(self, model: Any) -> str:
        """Obtiene información legible del modelo."""
        try:
            if isinstance(model, SentenceTransformer):
                info = f"SentenceTransformer: {model.model_name if hasattr(model, 'model_name') else 'unknown'}"
                if hasattr(model, 'get_sentence_embedding_dimension'):
                    info += f", dim={model.get_sentence_embedding_dimension()}"
                return info
            elif isinstance(model, HuggingFaceEmbeddings):
                return f"HuggingFaceEmbeddings: {model.model_name}"
            elif isinstance(model, DummyEmbedder):
                return f"DummyEmbedder: dim={model.dimension}"
            elif isinstance(model, EmbeddingsWrapper):
                return f"EmbeddingsWrapper: {self._get_model_info(model.model)}"
            else:
                return f"Modelo tipo: {type(model).__name__}"
        except Exception:
            return "Información del modelo no disponible"
    
    def _get_model_type(self, model: Any) -> str:
        """Obtiene el tipo de modelo."""
        if isinstance(model, SentenceTransformer):
            return 'sentence_transformer'
        elif isinstance(model, HuggingFaceEmbeddings):
            return 'huggingface'
        elif isinstance(model, DummyEmbedder):
            return 'dummy'
        elif isinstance(model, EmbeddingsWrapper):
            # Obtener el tipo del modelo envuelto
            if hasattr(model, 'model'):
                return self._get_model_type(model.model)
            return 'wrapped'
        else:
            return 'unknown'
    
    def _get_embedding_model(self) -> Any:
        """🔥 VERSIÓN MEJORADA: Obtiene el modelo con lazy loading y fallbacks robustos."""
        if self._embedding_model is None:
            with self._embedding_model_lock:
                if self._embedding_model is None:  # Double-check locking
                    logger.info(f"🔄 Iniciando carga de modelo: {self.embedding_model}")
                    
                    start_time = time.time()
                    
                    try:
                        # 🔥 USAR EL SISTEMA DE FALLBACKS MEJORADO
                        raw_model = self._load_model_with_fallback()
                        
                        load_time = time.time() - start_time
                        
                        # 🔥 MEJORA: Crear wrapper para compatibilidad
                        if isinstance(raw_model, SentenceTransformer):
                            # Crear wrapper para SentenceTransformer
                            self._embedding_model = EmbeddingsWrapper(raw_model)
                            self._log_ml_info(f"✅ SentenceTransformer cargado y envuelto en {load_time:.2f}s")
                        elif isinstance(raw_model, (HuggingFaceEmbeddings, DummyEmbedder)):
                            # HuggingFaceEmbeddings y DummyEmbedder ya son compatibles
                            self._embedding_model = raw_model
                            self._log_ml_info(f"✅ Modelo compatible cargado en {load_time:.2f}s")
                        else:
                            # Para cualquier otro tipo, crear wrapper
                            self._embedding_model = EmbeddingsWrapper(raw_model)
                            self._log_ml_info(f"✅ Modelo envuelto cargado en {load_time:.2f}s")
                        
                        # 🔥 MEJORA: Información adicional del modelo
                        model_info = self._get_model_info(raw_model)
                        self._log_ml_info(f"📋 Info modelo: {model_info}")
                        
                        # 🔥 MEJORA: Registrar uso en estadísticas
                        self._stats['model_load_time'] = load_time
                        self._stats['model_type'] = self._get_model_type(raw_model)
                        
                    except Exception as e:
                        logger.error(f"❌ Error fatal cargando modelo: {e}")
                        
                        # 🔥 MEJORA: Intentar crear dummy como último recurso
                        if self._fallback_config['use_dummy_on_failure']:
                            logger.warning("🔄 Intentando crear dummy embedder como último recurso...")
                            try:
                                self._embedding_model = self._create_dummy_embedder()
                                logger.warning("✅ DummyEmbedder creado como último recurso")
                            except Exception as dummy_error:
                                logger.error(f"❌ Error incluso creando dummy: {dummy_error}")
                                raise RuntimeError(
                                    f"No se pudo cargar ningún modelo de embeddings: {e}\n"
                                    f"Y falló al crear dummy: {dummy_error}"
                                )
                        else:
                            raise
        
        return self._embedding_model

    # ------------------------------------------------------------------
    # Métodos existentes mejorados
    # ------------------------------------------------------------------
    
    def _ensure_data_loaded(self):
        """Asegura que los datos estén cargados."""
        if not self.processed_json_path.exists():
            logger.warning("📦 Archivo procesado no encontrado. Ejecutando FastDataLoader...")
            
            # Usar parámetros correctos para FastDataLoader
            loader = FastDataLoader(
                use_progress_bar=True
                # ml_enabled y ml_features se manejan internamente
            )
            loader.load_data(self.processed_json_path)
            
            if not self.processed_json_path.exists():
                raise FileNotFoundError(f"No se pudo crear: {self.processed_json_path}")

    def load_products_optimized(self) -> List[Product]:
        """Carga optimizada de productos con tracking ML."""
        self._log_ml_info("🔵 CHROMA BUILDER STARTED - FALLBACK ENHANCED")
        logger.info("📦 Paso 1: Cargando productos...")
        
        # Asegurar que los datos existan
        self._ensure_data_loaded()
        
        start_time = time.time()
        
        try:
            with open(self.processed_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Modo DEBUG con manejo de errores
            if os.getenv("DEBUG"):
                original_count = len(data)
                data = data[:1000]
                logger.info(f"🔧 MODO DEBUG: Limitado a {len(data)} de {original_count} productos")
            
            # Procesamiento con tracking ML y manejo robusto de errores
            products = []
            ml_products_count = 0
            embedding_products_count = 0
            error_count = 0
            
            for i, item in enumerate(tqdm(data, desc="Cargando productos")):
                try:
                    product = Product.from_dict(item)
                    
                    if product:
                        products.append(product)
                        
                        # Track ML statistics
                        if getattr(product, 'ml_processed', False):
                            ml_products_count += 1
                        if hasattr(product, 'embedding') and product.embedding is not None:
                            embedding_products_count += 1
                    else:
                        error_count += 1
                        
                except Exception as e:
                    error_count += 1
                    logger.debug(f"Error creando producto {i}: {e}")
                    continue
            
            load_time = time.time() - start_time
            
            # Log detallado
            self._log_ml_info(f"📊 ML Statistics:")
            self._log_ml_info(f"   • Total productos: {len(products)}")
            self._log_ml_info(f"   • Con ML procesado: {ml_products_count}")
            self._log_ml_info(f"   • Con embeddings: {embedding_products_count}")
            self._log_ml_info(f"   • Errores: {error_count}")
            self._log_ml_info(f"   • Tiempo carga: {load_time:.2f}s")
            
            # ADVERTENCIA si hay muchos errores
            if error_count > len(products) * 0.5:  # Más del 50% errores
                logger.warning(f"⚠️  ADVERTENCIA: {error_count} errores de {len(data)} productos!")
            
            # Actualizar estadísticas
            self._stats['total_products'] = len(products)
            self._stats['products_with_ml'] = ml_products_count
            self._stats['products_with_embedding'] = embedding_products_count
            
            # ERROR si no hay productos
            if len(products) == 0:
                raise ValueError(f"No se pudo cargar ningún producto. Errores: {error_count}/{len(data)}")
            
            return products
            
        except Exception as e:
            logger.error(f"❌ Error cargando productos: {e}")
            raise

    def create_documents_optimized(self, products: List[Product]) -> List[Document]:
        """Crea documentos optimizados con información ML completa."""
        self._log_ml_info("📝 Paso 2: Creando documentos con ML...")
        
        start_time = time.time()
        documents = []
        skipped_count = 0
        
        # Estadísticas por tipo de producto
        ml_document_count = 0
        embedding_document_count = 0
        
        for product in tqdm(products, desc="Creando documentos"):
            try:
                document = self._product_to_optimized_document(product)
                if document:
                    documents.append(document)
                    
                    # Track ML documents
                    if hasattr(product, 'ml_processed') and product.ml_processed:
                        ml_document_count += 1
                    if hasattr(product, 'embedding') and product.embedding:
                        embedding_document_count += 1
                else:
                    skipped_count += 1
                    
            except Exception as e:
                logger.debug(f"Error creando documento para producto {product.id}: {e}")
                skipped_count += 1
            
            # Liberar memoria periódicamente
            if len(documents) % ChromaBuilderConfig.MEMORY_CHECK_INTERVAL == 0:
                gc.collect()
        
        process_time = time.time() - start_time
        
        # Log detallado ML
        self._log_ml_info(f"📊 Documentos creados:")
        self._log_ml_info(f"   • Total: {len(documents)}")
        self._log_ml_info(f"   • Con ML: {ml_document_count}")
        self._log_ml_info(f"   • Con embeddings: {embedding_document_count}")
        self._log_ml_info(f"   • Omitidos: {skipped_count}")
        self._log_ml_info(f"   • Tiempo: {process_time:.2f}s")
        
        self._stats['processed_documents'] = len(documents)
        self._stats['skipped_documents'] = skipped_count
        
        return documents

    def _product_to_optimized_document(self, product: Product) -> Optional[Document]:
        """Convierte producto a documento con toda la información ML."""
        try:
            # Usar métodos nativos de Product
            content = product.to_text()
            if not content or len(content.strip()) < 10:
                return None
            
            # Validación adicional para productos basura
            if self._is_low_quality_product(product):
                return None
            
            # Cache para detección de duplicados
            if self.enable_cache:
                h = str(product.id)  # Usar ID como hash
                if h in self._document_cache:
                    return None
                self._document_cache[h] = True
            
            # Metadata con información ML completa y consistente
            metadata = product.to_metadata()
            
            # Añadir información ML específica
            metadata["ml_processed"] = getattr(product, 'ml_processed', False)
            
            # Añadir categoría predicha si existe
            if hasattr(product, 'predicted_category') and product.predicted_category:
                metadata["predicted_category"] = product.predicted_category
            
            # Manejo de embeddings del producto
            if hasattr(product, 'embedding') and product.embedding:
                # Validar embedding
                is_valid = EmbeddingSerializer.validate_embedding(product.embedding)
                metadata["has_embedding"] = is_valid
                metadata["embedding_dim"] = len(product.embedding) if is_valid else 0
                metadata["embedding_model"] = getattr(product, 'embedding_model', 'unknown')
                
                # Serializar embedding si vamos a usarlo
                if self.use_product_embeddings and is_valid:
                    embedding_str = EmbeddingSerializer.serialize_embedding(product.embedding)
                    if embedding_str:
                        metadata["product_embedding"] = embedding_str
                        self._stats['valid_embeddings'] += 1
                    else:
                        self._stats['invalid_embeddings'] += 1
                else:
                    self._stats['invalid_embeddings'] += 1
            else:
                metadata["has_embedding"] = False
            
            # Añadir metadatos de procesamiento
            metadata["processing_timestamp"] = time.time()
            metadata["chroma_builder_version"] = self._index_metadata['builder_version']
            
            return Document(page_content=content, metadata=metadata)
            
        except Exception as e:
            logger.warning(f"Error convirtiendo producto {product.id} a documento: {e}")
            return None

    def _is_low_quality_product(self, product: Product) -> bool:
        """Valida si un producto es de baja calidad con consideraciones ML."""
        # Productos sin descripción Y sin features Y sin precio
        has_no_description = (not product.description or 
                            product.description.startswith("No description"))
        has_no_features = (not product.details.features or 
                          len(product.details.features) == 0)
        has_no_price = not product.price
        
        # Considerar si tiene información ML valiosa
        has_ml_info = (
            getattr(product, 'ml_processed', False) or
            getattr(product, 'predicted_category', None) or
            getattr(product, 'extracted_entities', None)
        )
        
        # Si no tiene nada y tampoco tiene ML, es basura
        if has_no_description and has_no_features and has_no_price and not has_ml_info:
            return True
        
        # Productos con metadata extremadamente pobre
        metadata = product.to_metadata()
        if (metadata.get("price", 0) == 0 and 
            metadata.get("average_rating", 0) == 0 and 
            not metadata.get("features")):
            return True
        
        return False

    def _embed_documents_batch(self, documents: List[Document]) -> Tuple[List[List[float]], int, int]:
        """Embeddings por lote con manejo de errores mejorado."""
        if not documents:
            return [], 0, 0
        
        self._log_ml_info(f"⚡ Computando embeddings para {len(documents)} documentos...")
        
        try:
            # 🔥 MEJORA: Verificar que el modelo esté disponible
            model = self._get_embedding_model()
            
            # 🔥 MEJORA: Manejar diferentes tipos de modelos
            if isinstance(model, DummyEmbedder):
                self._log_ml_info("⚠️  Usando DummyEmbedder - calidad limitada")
                # Procesar en lotes pequeños para evitar memory issues
                batch_size = min(100, len(documents))
                embeddings = []
                
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    batch_texts = [doc.page_content for doc in batch]
                    
                    if hasattr(model, 'encode'):
                        batch_embeddings = model.encode(batch_texts)
                    else:
                        batch_embeddings = model.embed_documents(batch_texts)
                    
                    embeddings.extend(batch_embeddings)
                    
                    # 🔥 MEJORA: Progress bar para dummy embedder
                    if i % 500 == 0:
                        self._log_ml_info(f"📊 DummyEmbedder: {i}/{len(documents)} documentos procesados")
                
                return embeddings, 0, len(documents)
            
            # Procesamiento normal para modelos reales
            return self._embed_documents_batch_normal(model, documents)
            
        except Exception as e:
            self._log_ml_info(f"❌ Error en embeddings: {e}")
            
            # 🔥 MEJORA: Fallback a dummy embeddings
            if self._fallback_config['use_dummy_on_failure']:
                self._log_ml_info("🔄 Fallback a DummyEmbedder debido a error...")
                dummy_model = self._create_dummy_embedder()
                
                # Usar contenido de documentos para embeddings dummy
                texts = [doc.page_content for doc in documents]
                dummy_embeddings = dummy_model.encode(texts)
                
                self._stats['dummy_fallback_used'] = True
                self._log_ml_info(f"✅ DummyEmbedder completó {len(documents)} documentos")
                
                return dummy_embeddings, 0, len(documents)
            else:
                raise
    
    def _embed_documents_batch_normal(self, model: Any, documents: List[Document]) -> Tuple[List[List[float]], int, int]:
        """Procesamiento normal de embeddings con soporte para embeddings preexistentes."""
        if not documents:
            return [], 0, 0
        
        product_embeddings_used = 0
        chroma_embeddings_computed = 0
        
        if self.use_product_embeddings:
            # Primera pasada: recolectar embeddings de productos válidos
            product_embeddings = []
            need_computation = []
            need_computation_indices = []
            
            for i, doc in enumerate(documents):
                if doc.metadata.get('has_embedding') and doc.metadata.get('product_embedding'):
                    # Intentar usar embedding del producto
                    embedding_str = doc.metadata['product_embedding']
                    embedding = EmbeddingSerializer.deserialize_embedding(embedding_str)
                    
                    if embedding is not None and EmbeddingSerializer.validate_embedding(embedding):
                        product_embeddings.append(embedding.tolist())
                        product_embeddings_used += 1
                        continue
                
                # Necesita embedding de Chroma
                need_computation.append(doc.page_content)
                need_computation_indices.append(i)
            
            # Si todos tienen embeddings válidos del producto, usarlos
            if product_embeddings_used == len(documents):
                self._log_ml_info(f"✅ Usando {product_embeddings_used} embeddings de productos")
                return product_embeddings, product_embeddings_used, 0
            
            # Mezclar embeddings correctamente
            if product_embeddings_used > 0:
                self._log_ml_info(f"🔀 Usando {product_embeddings_used} embeddings de producto y calculando {len(need_computation)}")
                
                # Computar embeddings para los que faltan
                if need_computation:
                    self._log_ml_info(f"🔧 Usando {self._get_model_type(model)} para {len(need_computation)} documentos")
                    
                    try:
                        chroma_embeddings = self._compute_embeddings_with_model(model, need_computation)
                        chroma_embeddings_computed = len(chroma_embeddings)
                        
                    except Exception as e:
                        self._log_ml_info(f"⚠️  Error en embed_documents: {e}")
                        raise
                    
                    # Combinar embeddings en el orden correcto
                    all_embeddings: List[Optional[List[float]]] = [None] * len(documents)
                    
                    # Poner embeddings de producto
                    product_idx = 0
                    for i, doc in enumerate(documents):
                        if i not in need_computation_indices:
                            all_embeddings[i] = product_embeddings[product_idx]
                            product_idx += 1
                    
                    # Poner embeddings de Chroma
                    chroma_idx = 0
                    for idx in need_computation_indices:
                        all_embeddings[idx] = chroma_embeddings[chroma_idx]
                        chroma_idx += 1
                    
                    # Asegurar que no haya None
                    final_embeddings = [emb for emb in all_embeddings if emb is not None]
                    
                    self._stats['ml_embeddings_used'] = product_embeddings_used
                    self._stats['chroma_embeddings_computed'] = chroma_embeddings_computed
                    
                    return final_embeddings, product_embeddings_used, chroma_embeddings_computed
        
        # Computar todos los embeddings con Chroma
        self._log_ml_info(f"🔄 Computando todos los embeddings con Chroma...")
        
        self._log_ml_info(f"🔧 Usando {self._get_model_type(model)} para {len(documents)} documentos")
        
        contents = [doc.page_content for doc in documents]
        embeddings_list = self._compute_embeddings_with_model(model, contents)
        chroma_embeddings_computed = len(documents)
        
        self._stats['chroma_embeddings_computed'] = chroma_embeddings_computed
        
        return embeddings_list, 0, chroma_embeddings_computed
    
    def _compute_embeddings_with_model(self, model: Any, texts: List[str]) -> List[List[float]]:
        """Computa embeddings usando cualquier tipo de modelo y convierte a List[List[float]]."""
        try:
            # Si es un EmbeddingsWrapper, obtener el modelo interno
            if isinstance(model, EmbeddingsWrapper):
                model = model.model
            
            if isinstance(model, SentenceTransformer):
                # SentenceTransformer devuelve numpy arrays por defecto
                embeddings_array = model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                # Convertir a lista de listas de floats
                if isinstance(embeddings_array, np.ndarray):
                    return embeddings_array.tolist()
                elif isinstance(embeddings_array, list):
                    return [EmbeddingSerializer.to_list(emb) for emb in embeddings_array]
                else:
                    # Manejar otros tipos
                    embeddings_list = []
                    for emb in embeddings_array:
                        if hasattr(emb, 'tolist'):
                            embeddings_list.append(emb.tolist())
                        else:
                            embeddings_list.append(EmbeddingSerializer.to_list(emb))
                    return embeddings_list
                    
            else:
                # Para HuggingFaceEmbeddings u otros
                embeddings_raw = model.embed_documents(texts)
                
                # Convertir a formato estándar
                embeddings_list = []
                for emb in embeddings_raw:
                    embeddings_list.append(EmbeddingSerializer.to_list(emb))
                return embeddings_list
                
        except Exception as e:
            self._log_ml_info(f"⚠️  Error computando embeddings: {e}")
            # Fallback: embeddings dummy
            dummy_model = DummyEmbedder(dimension=384)
            return dummy_model.encode(texts)

    def _create_chroma_with_embeddings(self, 
                                      documents: List[Document], 
                                      embeddings: List[List[float]], 
                                      persist_directory: Optional[str] = None) -> Chroma:
        """
        Crea índice Chroma con embeddings precomputados.
        Versión compatible con langchain-chroma 1.1.0
        """
        try:
            # Configuración de Chroma
            collection_metadata = {
                "hnsw:space": "cosine",
                "ml_enabled": str(self.use_product_embeddings),
                "builder_version": self._index_metadata['builder_version']
            }
            
            self._log_ml_info("🏗️  Creando índice Chroma...")
            
            # 🔥 MEJORA: Obtener el modelo de embeddings
            embedding_model = self._get_embedding_model()
            
            # 🔥 FIX: Asegurar que el modelo tenga los métodos requeridos
            if not hasattr(embedding_model, 'embed_documents'):
                # Si no tiene embed_documents, crear un wrapper
                embedding_model = EmbeddingsWrapper(embedding_model)
            
            # 🔥 FIX: Usar from_documents con el modelo de embeddings
            chroma_index = Chroma.from_documents(
                documents=documents,
                embedding=embedding_model,  # 🔥 Usar el modelo de embeddings
                persist_directory=persist_directory,
                collection_metadata=collection_metadata
            )
            
            # Log de éxito
            self._log_ml_info(f"✅ Índice Chroma creado con {len(documents)} documentos")
            
            return chroma_index
            
        except Exception as e:
            logger.error(f"❌ Error creando Chroma: {e}")
            raise

    # ------------------------------------------------------------------
    # 🔥 NUEVO: Métodos para diagnóstico
    # ------------------------------------------------------------------
    
    def diagnose_model_loading(self) -> Dict[str, Any]:
        """Realiza diagnóstico del sistema de carga de modelos."""
        diagnosis: Dict[str, Any] = {
            'embedding_model': self.embedding_model,
            'device': self.device,
            'ml_enabled': self.use_product_embeddings,
            'model_loaded': self._embedding_model is not None,
            'model_type': None,
            'model_failures': self._model_failures.copy(),
            'model_cache_size': len(self._model_cache),
            'fallback_config': self._fallback_config.copy()
        }
        
        if self._embedding_model is not None:
            diagnosis['model_type'] = self._get_model_type(self._embedding_model)
            diagnosis['model_info'] = self._get_model_info(self._embedding_model)
        
        # Verificar dependencias
        import importlib.util
        
        dependencies = {
            'sentence_transformers': importlib.util.find_spec("sentence_transformers") is not None,
            'langchain_huggingface': importlib.util.find_spec("langchain_huggingface") is not None,
            'torch': importlib.util.find_spec("torch") is not None,
            'transformers': importlib.util.find_spec("transformers") is not None,
        }
        
        diagnosis['dependencies'] = dependencies
        
        # Verificar acceso a modelos
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            model_info = api.model_info(self.embedding_model, timeout=5)
            diagnosis['model_accessible'] = True
            diagnosis['model_id'] = model_info.id
        except Exception as e:
            diagnosis['model_accessible'] = False
            diagnosis['model_error'] = str(e)
        
        return diagnosis
    
    def reset_model_cache(self) -> None:
        """Reinicia la cache del modelo y libera memoria."""
        with self._model_cache_lock:
            self._model_cache.clear()
        
        if self._embedding_model is not None:
            self._log_ml_info("🧹 Reiniciando cache del modelo...")
            del self._embedding_model
            self._embedding_model = None
            gc.collect()
        
        self._model_failures.clear()
        self._log_ml_info("✅ Cache del modelo reiniciada")

    def build_index(self, persist: bool = True) -> Chroma:
        """Construye el índice con soporte ML completo y fallbacks mejorados."""
        total_start_time = time.time()
        
        try:
            # 🔥 Log de configuración ML
            self._log_ml_info("=" * 50)
            self._log_ml_info("🏗️  CONSTRUYENDO ÍNDICE CHROMA CON FALLBACKS MEJORADOS")
            self._log_ml_info("=" * 50)
            self._log_ml_info(f"• Usar embeddings de producto: {self.use_product_embeddings}")
            self._log_ml_info(f"• Config ML global: {self.ml_config['enabled']}")
            self._log_ml_info(f"• Modelo: {self.embedding_model}")
            self._log_ml_info(f"• Dispositivo: {self.device}")
            self._log_ml_info(f"• Fallback config: {self._fallback_config}")
            
            # 1. Cargar productos
            products = self.load_products_optimized()
            
            # 2. Crear documentos
            documents = self.create_documents_optimized(products)
            
            if not documents:
                raise ValueError("No se pudieron crear documentos válidos")
            
            # 3. Computar embeddings (pero no los usaremos directamente en Chroma)
            embedding_start_time = time.time()
            embeddings, product_used, chroma_computed = self._embed_documents_batch(documents)
            self._stats['embedding_time'] = time.time() - embedding_start_time
            self._stats['ml_embeddings_used'] = product_used
            self._stats['chroma_embeddings_computed'] = chroma_computed
            
            # 4. Limpiar índice existente
            if persist and self.chroma_db_path.exists():
                self._log_ml_info("♻️  Eliminando índice Chroma existente")
                shutil.rmtree(self.chroma_db_path)
            
            # 5. Construir índice
            self._log_ml_info("🏗️  Construyendo índice Chroma...")
            
            persist_dir = str(self.chroma_db_path) if persist else None
            chroma_index = self._create_chroma_with_embeddings(
                documents=documents,
                embeddings=embeddings,  # No se usan directamente
                persist_directory=persist_dir
            )
            
            # 6. Estadísticas finales
            total_time = time.time() - total_start_time
            self._stats['total_time'] = total_time
            
            self._log_final_stats()
            
            return chroma_index
            
        except Exception as e:
            logger.error(f"❌ Error en la indexación: {e}")
            raise

    def _log_final_stats(self):
        """Registra estadísticas finales con información ML."""
        stats = self._stats
        
        self._log_ml_info("=" * 50)
        self._log_ml_info("📊 ESTADÍSTICAS FINALES - FALLBACK EDITION")
        self._log_ml_info("=" * 50)
        
        # Estadísticas básicas
        self._log_ml_info(f"📦 Productos cargados: {stats['total_products']}")
        self._log_ml_info(f"   • Con ML procesado: {stats['products_with_ml']}")
        self._log_ml_info(f"   • Con embeddings: {stats['products_with_embedding']}")
        
        self._log_ml_info(f"📝 Documentos creados: {stats['processed_documents']}")
        self._log_ml_info(f"⏭️  Documentos omitidos: {stats['skipped_documents']}")
        
        # Estadísticas de embeddings
        self._log_ml_info(f"🔢 Embeddings:")
        self._log_ml_info(f"   • Embeddings ML usados: {stats['ml_embeddings_used']}")
        self._log_ml_info(f"   • Embeddings Chroma calculados: {stats['chroma_embeddings_computed']}")
        self._log_ml_info(f"   • Embeddings válidos: {stats.get('valid_embeddings', 0)}")
        self._log_ml_info(f"   • Embeddings inválidos: {stats.get('invalid_embeddings', 0)}")
        
        if stats['embedding_time'] > 0:
            self._log_ml_info(f"⚡ Tiempo embeddings: {stats['embedding_time']:.1f}s")
            total_embeddings = stats['ml_embeddings_used'] + stats['chroma_embeddings_computed']
            if total_embeddings > 0:
                rate = total_embeddings / stats['embedding_time']
                self._log_ml_info(f"📈 Tasa embeddings: {rate:.1f} emb/s")
        
        # Información del modelo
        if stats.get('model_load_time'):
            self._log_ml_info(f"🤖 Modelo: {stats.get('model_type', 'unknown')}")
            self._log_ml_info(f"   • Tiempo carga: {stats['model_load_time']:.2f}s")
            if stats.get('dummy_fallback_used'):
                self._log_ml_info(f"   • ⚠️  DUMMY EMBEDDER ACTIVADO")
        
        # Tiempo total
        if stats['total_time'] > 0:
            self._log_ml_info(f"⏱️  Tiempo total: {stats['total_time']:.1f}s")
            if stats['processed_documents'] > 0:
                rate = stats['processed_documents'] / stats['total_time']
                self._log_ml_info(f"🚀 Tasa total: {rate:.1f} doc/s")
        
        self._log_ml_info(f"💾 Índice guardado → {self.chroma_db_path}")
        
        # También log normal
        logger.info(f"✅ Indexación completada: {stats['processed_documents']} documentos")

    def get_index_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del índice construido."""
        if not self.chroma_db_path.exists():
            return {"error": "Índice no existe"}
        
        try:
            # Cargar índice para obtener estadísticas
            embeddings = self._get_embedding_model()
            chroma_index = Chroma(
                persist_directory=str(self.chroma_db_path),
                embedding_function=embeddings
            )
            
            collection = chroma_index._collection
            count = collection.count()
            
            # Obtener metadata ML
            collection_metadata = collection.metadata or {}
            
            # Intentar obtener muestras para estadísticas ML
            sample_results = collection.get(limit=10)
            ml_stats = {
                "ml_enhanced": collection_metadata.get("ml_enabled", "false") == "true",
                "embedding_source": collection_metadata.get("embedding_source", "chroma"),
                "samples_with_ml": 0,
                "samples_with_embedding": 0
            }
            
            if sample_results and 'metadatas' in sample_results and sample_results['metadatas']:
                for metadata in sample_results['metadatas'][:10]:
                    if metadata and metadata.get('ml_processed'):
                        ml_stats["samples_with_ml"] += 1
                    if metadata and metadata.get('has_embedding'):
                        ml_stats["samples_with_embedding"] += 1
            
            return {
                "document_count": count,
                "index_path": str(self.chroma_db_path),
                "embedding_model": self.embedding_model,
                "ml_enabled": self.use_product_embeddings,
                "build_stats": self._stats,
                "ml_info": ml_stats,
                "collection_metadata": collection_metadata
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {"error": str(e)}

    # Alias para compatibilidad
    def build_index_optimized(self) -> Chroma:
        return self.build_index(persist=True)

    def build_index_batch_optimized(self, batch_size: int = 5000) -> Chroma:
        return self.build_index(persist=True)

    # Método para liberar memoria
    def cleanup(self):
        """Libera recursos del modelo de embeddings."""
        if self._embedding_model is not None:
            self._log_ml_info("🧹 Liberando memoria del modelo de embeddings...")
            del self._embedding_model
            self._embedding_model = None
            gc.collect()


# Alias para compatibilidad
ChromaBuilder = OptimizedChromaBuilder


# ------------------------------------------------------------------
# 🔥 NUEVO: Funciones utilitarias
# ------------------------------------------------------------------

def test_embedding_system():
    """Prueba el sistema de embeddings con diferentes configuraciones."""
    import tempfile
    
    print("🧪 Probando sistema de embeddings...")
    
    test_configs = [
        {"model": "all-MiniLM-L6-v2", "device": "cuda", "name": "Modelo estándar"},
        {"model": "sentence-transformers/all-MiniLM-L6-v2", "device": "cuda", "name": "Modelo HF"},
        {"model": "invalid-model-name", "device": "cuda", "name": "Modelo inválido (test fallback)"},
    ]
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Test: {config['name']}")
        print(f"Modelo: {config['model']}")
        print(f"Dispositivo: {config['device']}")
        print(f"{'='*60}")
        
        try:
            # Crear builder temporal
            with tempfile.TemporaryDirectory() as tmpdir:
                builder = OptimizedChromaBuilder(
                    chroma_db_path=Path(tmpdir) / "test_chroma",
                    embedding_model=config['model'],
                    device=config['device'],
                    use_product_embeddings=False,
                    ml_logging=True
                )
                
                # Diagnosticar carga de modelo
                diagnosis = builder.diagnose_model_loading()
                
                print(f"✅ Diagnosis completada")
                print(f"   • Modelo cargado: {diagnosis.get('model_loaded', False)}")
                print(f"   • Tipo de modelo: {diagnosis.get('model_type', 'N/A')}")
                print(f"   • Fallos: {len(diagnosis.get('model_failures', []))}")
                
                # Probar embeddings
                test_texts = ["Este es un texto de prueba", "Otro texto para embedding"]
                
                model = builder._get_embedding_model()
                if hasattr(model, 'encode'):
                    embeddings = model.encode(test_texts)
                else:
                    embeddings = model.embed_documents(test_texts)
                
                print(f"   • Embeddings generados: {len(embeddings)}")
                print(f"   • Dimensión: {len(embeddings[0]) if embeddings else 0}")
                
                builder.cleanup()
                
        except Exception as e:
            print(f"❌ Error en test: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("✅ Todos los tests completados")
    print(f"{'='*60}")


def validate_chroma_index(index_path: Path) -> Dict[str, Any]:
    """Valida un índice Chroma existente."""
    try:
        from langchain_chroma import Chroma
        
        print(f"🔍 Validando índice en: {index_path}")
        
        # Cargar índice
        index = Chroma(
            persist_directory=str(index_path),
            embedding_function=None
        )
        
        # Obtener información
        collection = index._collection
        count = collection.count()
        
        print(f"📊 Documentos en índice: {count}")
        
        if count > 0:
            # Obtener muestras con manejo seguro de None
            results = collection.get(limit=min(10, count))
            
            documents = results.get('documents', []) or []
            metadatas = results.get('metadatas', []) or []
            
            sample_count = min(len(documents), len(metadatas), 10)
            print(f"\n📋 Muestra de documentos (primeros {sample_count}):")
            
            for i in range(sample_count):
                doc = documents[i] if i < len(documents) else ""
                metadata = metadatas[i] if i < len(metadatas) else {}
                
                print(f"\n  Documento {i+1}:")
                print(f"    Texto: {doc[:100]}..." if doc else "    Texto: [Vacío]")
                print(f"    ML procesado: {metadata.get('ml_processed', False)}")
                print(f"    Tiene embedding: {metadata.get('has_embedding', False)}")
        
        # Obtener metadata de colección
        collection_metadata = collection.metadata or {}
        
        return {
            "valid": True,
            "document_count": count,
            "collection_metadata": collection_metadata,
            "index_path": str(index_path)
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "index_path": str(index_path)
        }


def build_chroma_from_cli():
    """Función para ejecutar desde CLI con opciones ML."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Constructor de índice Chroma con ML y fallbacks mejorados")
    parser.add_argument("--input", type=Path, help="Ruta al JSON procesado")
    parser.add_argument("--output", type=Path, help="Ruta para guardar ChromaDB")
    parser.add_argument("--model", type=str, help="Modelo de embeddings")
    parser.add_argument("--device", type=str, help="Dispositivo (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=ChromaBuilderConfig.DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=ChromaBuilderConfig.MAX_CONCURRENT_WORKERS)
    parser.add_argument("--no-cache", action="store_true", help="Deshabilitar cache")
    parser.add_argument("--in-memory", action="store_true", help="Modo in-memory")
    
    # Opciones ML unificadas
    parser.add_argument("--use-product-embeddings", action="store_true", 
                       help="Usar embeddings de Product (sobreescribe settings.ML_ENABLED)")
    parser.add_argument("--no-ml", action="store_true", 
                       help="Deshabilitar ML (sobreescribe settings.ML_ENABLED)")
    parser.add_argument("--ml-logging", action="store_true", 
                       help="Habilitar logging específico para ML")
    parser.add_argument("--debug", action="store_true", 
                       help="Modo debug (limitar productos)")
    
    # 🔥 NUEVO: Opciones para diagnóstico
    parser.add_argument("--diagnose", action="store_true", help="Ejecutar diagnóstico del sistema")
    parser.add_argument("--test", action="store_true", help="Ejecutar tests del sistema")
    parser.add_argument("--validate", type=Path, help="Validar un índice Chroma existente")
    
    args = parser.parse_args()
    
    if args.debug:
        os.environ["DEBUG"] = "1"
    
    # Determinar configuración ML
    use_ml = getattr(settings, 'ML_ENABLED', False)
    if args.use_product_embeddings:
        use_ml = True
    if args.no_ml:
        use_ml = False
    
    # Ejecutar diagnóstico si se solicita
    if args.diagnose:
        builder = OptimizedChromaBuilder(
            processed_json_path=args.input,
            chroma_db_path=args.output,
            embedding_model=args.model,
            device=args.device,
            use_product_embeddings=use_ml,
            ml_logging=True
        )
        diagnosis = builder.diagnose_model_loading()
        print("\n" + "="*60)
        print("🧪 DIAGNÓSTICO DEL SISTEMA DE EMBEDDINGS")
        print("="*60)
        for key, value in diagnosis.items():
            if key != 'dependencies' and key != 'fallback_config':
                print(f"  {key}: {value}")
        print(f"  dependencias: {diagnosis['dependencies']}")
        print(f"  fallback_config: {diagnosis['fallback_config']}")
        return
    
    # Ejecutar test si se solicita
    if args.test:
        test_embedding_system()
        return
    
    # Validar índice si se solicita
    if args.validate:
        result = validate_chroma_index(args.validate)
        print("\n" + "="*60)
        print("🔍 VALIDACIÓN DE ÍNDICE")
        print("="*60)
        for key, value in result.items():
            print(f"  {key}: {value}")
        return
    
    # Construcción normal del índice
    builder = OptimizedChromaBuilder(
        processed_json_path=args.input,
        chroma_db_path=args.output,
        embedding_model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_workers=args.workers,
        enable_cache=not args.no_cache,
        use_product_embeddings=use_ml,
        ml_logging=args.ml_logging
    )
    
    try:
        index = builder.build_index(persist=not args.in_memory)
        stats = builder.get_index_stats()
        
        print(f"\n{'='*60}")
        print("✅ ÍNDICE CHROMA CONSTRUIDO EXITOSAMENTE")
        print(f"{'='*60}")
        print(f"📊 Documentos: {stats.get('document_count', 'N/A')}")
        print(f"📁 Ubicación: {stats.get('index_path', 'N/A')}")
        print(f"🤖 Modelo: {stats.get('embedding_model', 'N/A')}")
        print(f"🔬 ML habilitado: {stats.get('ml_enabled', 'N/A')}")
        
        # Mostrar info ML
        ml_info = stats.get('ml_info', {})
        if ml_info:
            print(f"\n📈 INFORMACIÓN ML:")
            print(f"   • ML Enhanced: {'✅ Sí' if ml_info.get('ml_enhanced') else '❌ No'}")
            print(f"   • Fuente embeddings: {ml_info.get('embedding_source', 'chroma')}")
            if 'samples_with_ml' in ml_info:
                print(f"   • Muestras con ML: {ml_info['samples_with_ml']}/10")
            if 'samples_with_embedding' in ml_info:
                print(f"   • Muestras con embedding: {ml_info['samples_with_embedding']}/10")
        
        build_stats = stats.get('build_stats', {})
        if build_stats.get('total_time'):
            print(f"\n⏱️  Tiempo total: {build_stats['total_time']:.1f}s")
            if build_stats.get('processed_documents'):
                rate = build_stats['processed_documents'] / build_stats['total_time']
                print(f"🚀 Velocidad: {rate:.1f} documentos/segundo")
        
        # Mostrar información de fallbacks si se usaron
        if build_stats.get('model_type') == 'dummy':
            print(f"\n⚠️  ADVERTENCIA: Se usó DummyEmbedder - calidad de embeddings limitada")
        if build_stats.get('dummy_fallback_used'):
            print(f"⚠️  Se activó fallback a DummyEmbedder durante el procesamiento")
        
        print(f"\n🎉 ¡Índice listo para usar!")
        
        # Limpiar memoria al finalizar
        builder.cleanup()
        
    except Exception as e:
        print(f"❌ Error construyendo índice: {e}")
        # Limpiar memoria incluso en caso de error
        if 'builder' in locals():
            builder.cleanup()
        raise


if __name__ == "__main__":
    build_chroma_from_cli()