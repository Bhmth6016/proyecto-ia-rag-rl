#!/usr/bin/env python3
"""
ML Processor con gestión de memoria - VERSIÓN CORREGIDA
"""
# src/core/data/ml_processor.py
import logging
import time
import gc
import threading
from typing import Optional, List, Dict, Any
from pathlib import Path
from functools import lru_cache
import psutil
import numpy as np

from src.core.config import settings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configurar logging para desactivar progress bars
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("tqdm").setLevel(logging.WARNING)
logging.getLogger("tqdm.auto").setLevel(logging.WARNING)
logging.getLogger("tqdm.std").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Desactivar progress bars de transformers
try:
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()
    # No usar disable_progress_bars si no existe
    if hasattr(transformers_logging, 'disable_progress_bar'):
        transformers_logging.disable_progress_bar()
except (ImportError, AttributeError):
    pass

# Desactivar progress bars de datasets
try:
    import datasets
    if hasattr(datasets, 'disable_progress_bar'):
        datasets.disable_progress_bar()
except ImportError:
    pass

# ------------------------------------------------------------------
# SINGLETON CLASSES para evitar problemas de atributos en funciones
# ------------------------------------------------------------------

class EmbeddingModelSingleton:
    """Singleton para el modelo de embeddings."""
    _instance = None
    _model = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, model_name: Optional[str] = None):
        """Obtiene la instancia singleton del modelo de embeddings."""
        if model_name is None:
            model_name = settings.ML_EMBEDDING_MODEL
        
        with cls._lock:
            if cls._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    logger.info(f"🔧 Cargando modelo de embeddings: {model_name}")
                    
                    # Configurar para desactivar progress bars de manera diferente
                    import os
                    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
                    os.environ["TOKENIZERS_PARALLELISM"] = "false"
                    
                    cls._model = SentenceTransformer(
                        model_name,
                        device='cuda'  # ← OPCIÓN: especificar dispositivo
                    )
                    logger.info(f"✅ Modelo de embeddings cargado")
                except ImportError:
                    logger.warning("⚠️ SentenceTransformer no disponible")
                    cls._model = None
                except Exception as e:
                    logger.error(f"❌ Error cargando modelo: {e}")
                    cls._model = None
        
        return cls._model
    
    @classmethod
    def clear(cls):
        """Limpia el singleton."""
        with cls._lock:
            if cls._model is not None:
                del cls._model
                cls._model = None
                logger.info("✅ Singleton de embeddings limpiado")


class NLPEnricherSingleton:
    """Singleton para el NLP enricher."""
    _enricher = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, enable_nlp: bool = True, device: str = "cuda"):
        """Obtiene la instancia singleton del NLP enricher."""
        if not enable_nlp:
            return None
        
        with cls._lock:
            if cls._enricher is None:
                try:
                    # 🔥 CORRECCIÓN: Importar correctamente NLPEnricher
                    from src.core.nlp.enrichment import NLPEnricher
                    logger.info(f"🔧 Cargando NLP enricher")
                    
                    # 🔥 CORRECCIÓN: Pasar solo parámetros válidos para NLPEnricher
                    # NLPEnricher no tiene el parámetro disable_progress_bars
                    cls._enricher = NLPEnricher(
                        device=device,
                        use_small_models=True  # Usar modelos pequeños por defecto
                    )
                    
                    # Inicializar el enricher
                    cls._enricher.initialize()
                    logger.info(f"✅ NLP enricher cargado")
                except ImportError as e:
                    logger.warning(f"⚠️ NLPEnricher no disponible: {e}")
                    cls._enricher = None
                except Exception as e:
                    logger.error(f"❌ Error cargando NLP enricher: {e}")
                    cls._enricher = None
        
        return cls._enricher
    
    @classmethod
    def clear(cls):
        """Limpia el singleton."""
        with cls._lock:
            if cls._enricher is not None:
                cls._enricher.cleanup_memory()
                del cls._enricher
                cls._enricher = None
                logger.info("✅ Singleton de NLP enricher limpiado")


# ------------------------------------------------------------------
# Helper functions usando las clases singleton
# ------------------------------------------------------------------

def _get_embedding_model_singleton(model_name: Optional[str] = None):
    """Función helper para compatibilidad con código existente."""
    return EmbeddingModelSingleton.get_instance(model_name)


def _get_nlp_enricher_singleton(enable_nlp: bool = True, device: str = "cuda"):
    """Función helper para compatibilidad con código existente."""
    return NLPEnricherSingleton.get_instance(enable_nlp, device)


def _create_dummy_embedder(dimension: int = 384):
    """Crea un embedder dummy como fallback."""
    class DummyEmbedder:
        def __init__(self, dimension: int = 384):
            self.dimension = dimension
            self.model_name = "dummy_embedder"
        
        def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
            import random
            embeddings = []
            for text in texts:
                random.seed(hash(text) % 1000000)
                embedding = [random.gauss(0, 1) for _ in range(self.dimension)]
                norm = sum(x**2 for x in embedding) ** 0.5
                if norm > 0:
                    embedding = [x / norm for x in embedding]
                embeddings.append(embedding)
            return embeddings
        
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return self.encode(texts)
    
    return DummyEmbedder(dimension)


# ------------------------------------------------------------------
# Clase principal
# ------------------------------------------------------------------

class ProductDataPreprocessor:
    """Preprocesador de datos de productos con gestión de memoria."""
    
    def __init__(self, 
                 verbose: bool = False,
                 max_memory_mb: int = 2048,
                 memory_monitoring: bool = True,
                 enable_nlp: bool = True):
        self.verbose = verbose
        self.max_memory_mb = max_memory_mb
        self.memory_monitoring = memory_monitoring
        self.enable_nlp = enable_nlp
        
        # 🔥 CORRECCIÓN: Añadir atributo device
        self.device = "cuda" if settings.ML_USE_GPU else "cuda"
        
        # Modelos (lazy loading)
        self._embedding_model = None
        self._zero_shot_classifier = None
        self._model_lock = threading.Lock()
        
        # Cache para embeddings frecuentes
        self._embedding_cache = {}
        self._cache_lock = threading.Lock()
        
        self._nlp_enricher = None
        # Estadísticas
        self._stats = {
            'total_processed': 0,
            'memory_usage_peak': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'start_time': time.time()
        }
        
        if self.verbose:
            logger.info(f"🔧 ProductDataPreprocessor inicializado (límite memoria: {max_memory_mb}MB)")
    
    def _get_nlp_enricher(self):
        """Obtiene enriquecedor NLP (lazy loading)"""
        if self.enable_nlp and self._nlp_enricher is None:
            try:
                # Usar singleton para evitar problemas
                self._nlp_enricher = NLPEnricherSingleton.get_instance(
                    enable_nlp=True,
                    device=self.device
                )
                if self._nlp_enricher:
                    logger.debug("✅ NLP enricher obtenido del singleton")
            except Exception as e:
                logger.warning(f"⚠️ NLPEnricher no disponible: {e}")
                self._nlp_enricher = None
        return self._nlp_enricher
    
    def _log(self, message: str):
        """Log condicional basado en verbose."""
        if self.verbose:
            logger.info(f"[ML Processor] {message}")
    
    def warm_up_models(self):
        """Pre-carga los modelos de ML."""
        self._log("🔧 Pre-calentando modelos...")
        
        # Cargar modelo de embeddings
        try:
            self._get_embedding_model()
            self._log("✅ Modelo de embeddings pre-cargado")
        except Exception as e:
            self._log(f"⚠️ Error pre-cargando modelo de embeddings: {e}")
        
        # Podrías agregar más modelos aquí
        self._log("✅ Pre-calentamiento completado")
    
    def _get_embedding_model(self):
        """Obtiene el modelo de embeddings con lazy loading."""
        if self._embedding_model is None:
            with self._model_lock:
                if self._embedding_model is None:
                    try:
                        # Intentar con singleton primero
                        model = EmbeddingModelSingleton.get_instance()
                        if model is not None:
                            self._embedding_model = model
                            self._log(f"✅ Modelo de embeddings obtenido del singleton")
                            return self._embedding_model
                        
                        # Fallback a carga directa
                        from sentence_transformers import SentenceTransformer
                        model_name = settings.ML_EMBEDDING_MODEL
                        self._log(f"🔧 Cargando modelo de embeddings: {model_name}")
                        
                        # Configurar para desactivar progress bars
                        import os
                        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
                        
                        self._embedding_model = SentenceTransformer(model_name)
                        self._log(f"✅ Modelo de embeddings cargado directamente")
                    except ImportError:
                        self._log("⚠️ SentenceTransformer no disponible, usando dummy")
                        self._embedding_model = _create_dummy_embedder()
                    except Exception as e:
                        self._log(f"❌ Error cargando modelo: {e}, usando dummy")
                        self._embedding_model = _create_dummy_embedder()
        
        return self._embedding_model
    
    def preprocess_product(self, 
                          product_data: Dict[str, Any], 
                          enable_ml: bool = True) -> Dict[str, Any]:
        """Preprocesa un producto individual."""
        self._stats['total_processed'] += 1
        
        # Copiar datos para no modificar el original
        processed = product_data.copy()
        
        # Procesamiento básico
        processed = self._basic_preprocessing(processed)
        
        # Procesamiento ML si está habilitado
        if enable_ml and settings.ML_ENABLED:
            try:
                processed = self._ml_processing(processed)
                processed['ml_processed'] = True
            except Exception as e:
                self._log(f"⚠️ Error en procesamiento ML: {e}")
                processed['ml_processed'] = False
        
        # Monitorear memoria periódicamente
        if self.memory_monitoring and self._stats['total_processed'] % 100 == 0:
            self._check_memory_usage()
        
        return processed
    
    def _basic_preprocessing(self, product_data: Dict) -> Dict:
        """Preprocesamiento básico del producto."""
        processed = product_data.copy()
        
        # Limpieza de texto básica
        if 'title' in processed and processed['title']:
            processed['title'] = str(processed['title']).strip()[:200]
        
        if 'description' in processed and processed['description']:
            processed['description'] = str(processed['description']).strip()[:1000]
        
        # Normalizar precio
        if 'price' in processed:
            try:
                if isinstance(processed['price'], str):
                    # Extraer números de strings como "$29.99"
                    import re
                    match = re.search(r'(\d+\.?\d*)', processed['price'])
                    if match:
                        processed['price'] = float(match.group(1))
                else:
                    processed['price'] = float(processed['price'])
            except (ValueError, TypeError):
                processed['price'] = 0.0
        
        return processed
    
    def _ml_processing(self, product_data: Dict) -> Dict:
        """Procesamiento ML del producto con NLP."""
        processed = product_data.copy()
        
        # 🔥 NUEVO: Procesamiento NLP si está habilitado
        if self.enable_nlp and settings.NLP_ENABLED:
            nlp_enricher = self._get_nlp_enricher()
            if nlp_enricher:
                # Usar categorías del sistema
                categories = settings.ML_CATEGORIES if hasattr(settings, 'ML_CATEGORIES') else None
                
                processed = nlp_enricher.enrich_product(processed, categories)
                processed['nlp_processed'] = True
        
        # Procesamiento ML existente (embedding, etc.)
        if 'embedding' in settings.ML_FEATURES:
            text = self._get_text_for_embedding(processed)
            if text:
                embedding = self._get_or_create_embedding(text)
                if embedding is not None:
                    processed['embedding'] = embedding
                    processed['embedding_model'] = settings.ML_EMBEDDING_MODEL
        
        return processed
    
    def _get_text_for_embedding(self, product_data: Dict) -> str:
        """Obtiene texto para embeddings."""
        parts = []
        
        if product_data.get('title'):
            parts.append(str(product_data['title']))
        
        if product_data.get('description'):
            parts.append(str(product_data['description']))
        
        if product_data.get('brand'):
            parts.append(str(product_data['brand']))
        
        return " ".join(parts[:3])  # Limitar a 3 partes para eficiencia
    
    def _get_or_create_embedding(self, text: str) -> Optional[List[float]]:
        # Generar clave de cache
        cache_key = hash(text) % 1000000
        
        with self._cache_lock:
            # Verificar cache
            if cache_key in self._embedding_cache:
                self._stats['cache_hits'] += 1
                return self._embedding_cache[cache_key]
            
            self._stats['cache_misses'] += 1
        
        # Generar nuevo embedding
        try:
            model = self._get_embedding_model()
            if model is None:
                return None
            
            # 🔥 CORRECCIÓN: Configurar para desactivar progress bars en encoding
            import os
            original_verbosity = os.environ.get("TRANSFORMERS_VERBOSITY", "error")
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            
            try:
                if hasattr(model, 'encode'):
                    # Desactivar tqdm para encoding
                    embedding_result = model.encode(
                        [text], 
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        batch_size=32,
                        show_progress_bar=False  # 🔥 Desactivar progress bar
                    )
                    
                    # 🔥 CORRECCIÓN: Asegurar que sea un numpy array y convertir a lista
                    if isinstance(embedding_result, np.ndarray):
                        embedding = embedding_result[0]
                        embedding_list = embedding.tolist()
                    else:
                        # Si ya es una lista o tupla
                        embedding = embedding_result[0] if isinstance(embedding_result, (list, tuple)) else embedding_result
                        embedding_list = embedding if isinstance(embedding, list) else embedding.tolist()
                else:
                    embedding_result = model.embed_documents([text])
                    embedding = embedding_result[0] if isinstance(embedding_result, (list, tuple)) else embedding_result
                    embedding_list = embedding if isinstance(embedding, list) else embedding.tolist()
            finally:
                # Restaurar verbosidad original
                os.environ["TRANSFORMERS_VERBOSITY"] = original_verbosity
            
            # Almacenar en cache (si hay espacio)
            with self._cache_lock:
                if len(self._embedding_cache) < 1000:  # Límite de cache
                    self._embedding_cache[cache_key] = embedding_list
            
            return embedding_list
            
        except Exception as e:
            self._log(f"⚠️ Error generando embedding: {e}")
            return None
    
    def preprocess_batch(self, 
                    products_data: List[Dict[str, Any]], 
                    enable_ml: bool = True,
                    batch_size: int = 100) -> List[Dict[str, Any]]:
        """Preprocesa un batch de productos - VERSIÓN SIN RECURSIÓN"""
        self._log(f"🔧 Procesando batch de {len(products_data)} productos")
        
        results = []
        
        # 🔥 EVITAR LLAMADAS RECURSIVAS: Procesar directamente
        for i in range(0, len(products_data), batch_size):
            batch = products_data[i:i + batch_size]
            batch_results = []
            
            for product_data in batch:
                try:
                    # Usar el método de instancia directamente
                    result = self.preprocess_product(product_data, enable_ml)
                    batch_results.append(result)
                except Exception as e:
                    self._log(f"⚠️ Error procesando producto: {e}")
                    batch_results.append(product_data)
            
            results.extend(batch_results)
            
            # Limpieza periódica
            if self.memory_monitoring and i % (batch_size * 5) == 0:
                self._cleanup_resources()
        
        self._log(f"✅ Batch procesado: {len(results)} productos")
        return results
    
    def _check_memory_usage(self) -> Dict[str, float]:
        """Verifica el uso de memoria."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        rss_mb = memory_info.rss / 1024 / 1024
        vms_mb = memory_info.vms / 1024 / 1024
        
        self._stats['memory_usage_peak'] = max(
            self._stats['memory_usage_peak'], rss_mb
        )
        
        if rss_mb > self.max_memory_mb * 0.9:  # 90% del límite
            self._log(f"⚠️  Memoria alta: {rss_mb:.1f}MB, limpiando...")
            self._cleanup_resources()
        
        return {
            'rss_mb': rss_mb,
            'vms_mb': vms_mb,
            'peak_mb': self._stats['memory_usage_peak']
        }
    
    def _cleanup_resources(self):
        """Limpia recursos para liberar memoria."""
        self._log("🧹 Limpiando recursos...")
        
        # Limpiar cache de embeddings
        with self._cache_lock:
            self._embedding_cache.clear()
        
        # Forzar garbage collection
        gc.collect()
        
        # Verificar memoria después de limpiar
        memory = self._check_memory_usage()
        self._log(f"📊 Memoria después de limpieza: {memory['rss_mb']:.1f}MB")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache."""
        with self._cache_lock:
            cache_size = len(self._embedding_cache)
        
        uptime = time.time() - self._stats['start_time']
        
        return {
            'total_processed': self._stats['total_processed'],
            'cache_size': cache_size,
            'cache_hits': self._stats['cache_hits'],
            'cache_misses': self._stats['cache_misses'],
            'cache_hit_ratio': (
                self._stats['cache_hits'] / max(1, self._stats['cache_hits'] + self._stats['cache_misses'])
            ),
            'memory_usage_peak_mb': self._stats['memory_usage_peak'],
            'uptime_seconds': uptime,
            'processing_rate': self._stats['total_processed'] / max(1, uptime)
        }
    
    def check_memory_usage(self) -> Dict[str, float]:
        """Interfaz pública para verificar memoria."""
        return self._check_memory_usage()
    
    def cleanup_memory(self):
        """⚠️ IMPORTANTE: Método para liberar memoria - SOLUCIÓN PROBLEMA 4"""
        self._log("🧹 Liberando memoria de modelos grandes...")
        
        with self._model_lock:
            if self._embedding_model is not None:
                del self._embedding_model
                self._embedding_model = None
                self._log("✅ Modelo de embeddings liberado")
        
        # Limpiar cache
        with self._cache_lock:
            self._embedding_cache.clear()
            self._log("✅ Cache de embeddings limpiado")
        
        # Forzar garbage collection
        gc.collect()
        
        self._log("✅ Memoria liberada")
    
    def auto_cleanup_if_needed(self) -> bool:
        """Limpia automáticamente si es necesario."""
        memory = self.check_memory_usage()
        
        if memory['rss_mb'] > self.max_memory_mb * 0.8:
            self._log("⚡ Limpieza automática activada por alto uso de memoria")
            self._cleanup_resources()
            return True
        
        return False
    
    def reset_to_initial_state(self):
        """Reinicia al estado inicial."""
        self._log("🔄 Reiniciando a estado inicial...")
        
        self.cleanup_memory()
        
        # Reiniciar estadísticas
        self._stats = {
            'total_processed': 0,
            'memory_usage_peak': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'start_time': time.time()
        }
        
        self._log("✅ Reinicio completado")


# ------------------------------------------------------------------
# Context Managers y funciones utilitarias
# ------------------------------------------------------------------

class MLProcessorContextManager:
    """Context manager para ProductDataPreprocessor."""
    
    def __init__(self, verbose: bool = False, **kwargs):
        self.verbose = verbose
        self.kwargs = kwargs
        self.preprocessor = None
    
    def __enter__(self):
        self.preprocessor = ProductDataPreprocessor(
            verbose=self.verbose,
            **self.kwargs
        )
        return self.preprocessor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.preprocessor is not None:
            self.preprocessor.cleanup_memory()
            if self.verbose:
                logger.info("[Context Manager] Recursos liberados automáticamente")


def create_ml_preprocessor_with_context(verbose: bool = False, **kwargs):
    """Crea un ProductDataPreprocessor con context manager."""
    return MLProcessorContextManager(verbose=verbose, **kwargs)


def process_with_memory_management(products_data: List[Dict[str, Any]], 
                                 use_gpu: bool = False,
                                 batch_size: int = 100,
                                 verbose: bool = True) -> List[Dict[str, Any]]:
    """Función de alto nivel con gestión automática de memoria - VERSIÓN SIN RECURSIÓN"""
    logger.info(f"🚀 Procesando {len(products_data)} productos con gestión de memoria")
    
    results = []
    
    # 🔥 CORRECCIÓN: Usar try-except-finally para manejar preprocessor
    preprocessor = None
    try:
        # Crear preprocessor directamente
        preprocessor = ProductDataPreprocessor(
            verbose=verbose,
            max_memory_mb=2048,
            memory_monitoring=True,
            enable_nlp=True
        )
        
        # Pre-calentar modelos
        if verbose:
            logger.info("🔧 Pre-calentando modelos...")
        preprocessor.warm_up_models()
        
        # Procesar en batches MANUALMENTE para evitar recursión
        total_batches = (len(products_data) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(products_data))
            batch = products_data[start_idx:end_idx]
            
            if verbose and batch_idx % 10 == 0:
                logger.info(f"📦 Procesando batch {batch_idx + 1}/{total_batches}")
            
            # Procesar cada producto individualmente
            batch_results = []
            for product_data in batch:
                try:
                    result = preprocessor.preprocess_product(product_data, enable_ml=True)
                    batch_results.append(result)
                except Exception as e:
                    logger.warning(f"⚠️ Error procesando producto: {e}")
                    batch_results.append(product_data)  # Mantener original
            
            results.extend(batch_results)
            
            # Limpiar periódicamente
            if batch_idx % 5 == 0:
                preprocessor.auto_cleanup_if_needed()
    
    except Exception as e:
        logger.error(f"❌ Error en procesamiento: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Asegurar limpieza
        if preprocessor is not None:
            preprocessor.cleanup_memory()
    
    logger.info(f"✅ Procesamiento completado: {len(results)} productos")
    return results


# ------------------------------------------------------------------
# Funciones de conveniencia para importación
# ------------------------------------------------------------------

def get_ml_preprocessor(verbose: bool = False, **kwargs) -> ProductDataPreprocessor:
    """Obtiene un preprocesador ML."""
    return ProductDataPreprocessor(verbose=verbose, **kwargs)


def cleanup_global_resources():
    """Limpia recursos globales del módulo."""
    logger.info("🧹 Limpiando recursos globales ML...")
    
    # Limpiar singletons usando las clases
    EmbeddingModelSingleton.clear()
    NLPEnricherSingleton.clear()
    
    # Forzar garbage collection
    gc.collect()
    
    logger.info("✅ Recursos globales liberados")


# ------------------------------------------------------------------
# Exportaciones
# ------------------------------------------------------------------

__all__ = [
    'ProductDataPreprocessor',
    'create_ml_preprocessor_with_context',
    'process_with_memory_management',
    'get_ml_preprocessor',
    'cleanup_global_resources',
]

# Alias para compatibilidad
cleanup_memory = cleanup_global_resources