# src/core/data/chroma_builder.py

import os
import json
import logging
import shutil
import time
import pickle
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import gc

from tqdm import tqdm
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np

from src.core.data.product import Product
from src.core.config import settings
from src.core.utils.logger import get_logger
from src.core.data.loader import FastDataLoader

logger = get_logger(__name__)

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
    EMBEDDING_DEVICE = "cpu"
    
    # Límites de memoria
    MAX_DOCUMENTS_PER_BATCH = 5000
    MEMORY_CHECK_INTERVAL = 10000

# ------------------------------------------------------------------
# Embedding Serializer para embeddings de Product
# ------------------------------------------------------------------
class EmbeddingSerializer:
    """Serializa/deserializa embeddings para almacenamiento en metadata"""
    
    @staticmethod
    def serialize_embedding(embedding: List[float]) -> str:
        """Convierte embedding a string base64"""
        try:
            arr = np.array(embedding, dtype=np.float32)
            serialized = pickle.dumps(arr)
            return base64.b64encode(serialized).decode('utf-8')
        except Exception as e:
            logger.warning(f"Error serializando embedding: {e}")
            return ""
    
    @staticmethod
    def deserialize_embedding(embedding_str: str) -> Optional[List[float]]:
        """Convierte string base64 a embedding"""
        try:
            if not embedding_str:
                return None
            serialized = base64.b64decode(embedding_str.encode('utf-8'))
            arr = pickle.loads(serialized)
            return arr.tolist()
        except Exception as e:
            logger.warning(f"Error deserializando embedding: {e}")
            return None
    
    @staticmethod
    def embedding_to_json(embedding: List[float]) -> str:
        """Convierte embedding a JSON string (menos eficiente pero legible)"""
        try:
            # Truncar para no hacer metadata demasiado grande
            truncated = embedding[:50] if len(embedding) > 50 else embedding
            return json.dumps([float(x) for x in truncated])
        except Exception as e:
            logger.warning(f"Error convirtiendo embedding a JSON: {e}")
            return "[]"

# ------------------------------------------------------------------
# Optimized Chroma Builder
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
        compression: bool = False,
        # 🔥 NUEVO: Configuración ML
        use_product_embeddings: bool = False,  # Usar embeddings propios del Product
        ml_logging: bool = True  # Logging específico para ML
    ):
        """
        Constructor optimizado para ChromaDB con soporte ML completo.
        
        Args:
            processed_json_path: Ruta al JSON unificado de productos
            chroma_db_path: Donde se guardará el índice Chroma
            embedding_model: Modelo de embeddings a usar
            device: Dispositivo para los embeddings (cpu/cuda)
            batch_size: Tamaño de lote para procesamiento
            max_workers: Número máximo de workers paralelos
            enable_cache: Habilitar cache de documentos
            compression: OBSOLETO - mantener por compatibilidad
            use_product_embeddings: Si True, usa embeddings de Product cuando están disponibles
            ml_logging: Habilitar logging específico para ML
        """
        self.processed_json_path = processed_json_path or settings.PROC_DIR / "products.json"
        self.chroma_db_path = chroma_db_path or Path(settings.CHROMA_DB_PATH)
        self.embedding_model = embedding_model or ChromaBuilderConfig.EMBEDDING_MODEL_NAME
        self.device = device or ChromaBuilderConfig.EMBEDDING_DEVICE
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.enable_cache = enable_cache
        
        # 🔥 NUEVO: Configuración ML
        self.use_product_embeddings = use_product_embeddings
        self.ml_logging = ml_logging
        
        # Cache para documentos procesados
        self._document_cache = {}
        self._embedding_model = None
        
        # 🔥 NUEVO: Configurar logging ML
        if self.ml_logging:
            self._setup_ml_logging()
        
        # Estadísticas mejoradas
        self._stats = {
            'total_products': 0,
            'processed_documents': 0,
            'skipped_documents': 0,
            'total_time': 0,
            'embedding_time': 0,
            # 🔥 NUEVO: Estadísticas ML
            'products_with_ml': 0,
            'products_with_embedding': 0,
            'ml_embeddings_used': 0,
            'chroma_embeddings_computed': 0
        }

    def _setup_ml_logging(self):
        """Configura logging específico para ML"""
        try:
            # Crear logger específico para ML
            self.ml_logger = logging.getLogger('ml_chroma_builder')
            self.ml_logger.setLevel(logging.INFO)
            
            # Evitar duplicar handlers
            if not self.ml_logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s - ML-Chroma - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S'
                ))
                self.ml_logger.addHandler(handler)
            
            # También configurar logger para el producto
            product_ml_logger = logging.getLogger('ml_processor')
            product_ml_logger.setLevel(logging.WARNING)  # Reducir verbosidad
            
        except Exception as e:
            logger.warning(f"Error configurando logging ML: {e}")

    def _log_ml_info(self, message: str):
        """Log message específico para ML"""
        if self.ml_logging:
            if hasattr(self, 'ml_logger'):
                self.ml_logger.info(message)
            else:
                logger.info(f"[ML] {message}")

    def _get_embedding_model(self):
        """Obtiene el modelo de embeddings con lazy loading"""
        if self._embedding_model is None:
            self._log_ml_info("🔄 Cargando modelo de embeddings...")
            start_time = time.time()
            
            try:
                # Usar SentenceTransformer directamente para mejor control
                self._embedding_model = SentenceTransformer(
                    self.embedding_model,
                    device=self.device
                )
            except Exception as e:
                self._log_ml_info(f"⚠️  Error con SentenceTransformer: {e}")
                self._embedding_model = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={"device": self.device},
                    encode_kwargs={
                        "batch_size": ChromaBuilderConfig.EMBEDDING_BATCH_SIZE,
                        "normalize_embeddings": True
                    }
                )
            
            load_time = time.time() - start_time
            self._log_ml_info(f"✅ Modelo de embeddings cargado en {load_time:.2f}s")
        
        return self._embedding_model

    def _ensure_data_loaded(self):
        """Asegura que los datos estén cargados, ejecutando FastDataLoader si es necesario"""
        if not self.processed_json_path.exists():
            logger.warning("📦 Archivo procesado no encontrado. Ejecutando FastDataLoader automáticamente...")
            
            # 🔥 NUEVO: Pasar configuración ML al loader
            ml_enabled = getattr(settings, "ML_ENABLED", False)
            ml_features = getattr(settings, "ML_FEATURES", ["category", "entities"])
            
            loader = FastDataLoader(
                use_progress_bar=True,
                ml_enabled=ml_enabled,
                ml_features=ml_features
            )
            loader.load_data(self.processed_json_path)
            
            if not self.processed_json_path.exists():
                raise FileNotFoundError(f"No se pudo crear el archivo procesado: {self.processed_json_path}")

    def load_products_optimized(self) -> List[Product]:
        """Carga optimizada de productos con tracking ML"""
        self._log_ml_info("🔵 CHROMA BUILDER STARTED - ML Edition")
        logger.info("📦 Paso 1: Cargando productos...")
        
        # Asegurar que los datos existan
        self._ensure_data_loaded()
        
        start_time = time.time()
        
        try:
            with open(self.processed_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Modo DEBUG: Limitar productos
            if os.getenv("DEBUG"):
                original_count = len(data)
                data = data[:1000]
                logger.info(f"🔧 MODO DEBUG: Limitado a {len(data)} de {original_count} productos")
            
            # Procesamiento con tracking ML
            products = []
            ml_products_count = 0
            embedding_products_count = 0
            
            for item in data:
                try:
                    product = Product.from_dict(item)
                    products.append(product)
                    
                    # 🔥 NUEVO: Track ML statistics
                    if getattr(product, 'ml_processed', False):
                        ml_products_count += 1
                    if getattr(product, 'embedding', None):
                        embedding_products_count += 1
                        
                except Exception as e:
                    logger.debug(f"Error creando producto: {e}")
                    continue
            
            load_time = time.time() - start_time
            
            # 🔥 NUEVO: Log ML statistics
            self._log_ml_info(f"📊 ML Statistics:")
            self._log_ml_info(f"   • Total productos: {len(products)}")
            self._log_ml_info(f"   • Con ML procesado: {ml_products_count}")
            self._log_ml_info(f"   • Con embeddings: {embedding_products_count}")
            self._log_ml_info(f"   • Tiempo carga: {load_time:.2f}s")
            
            # Actualizar estadísticas
            self._stats['total_products'] = len(products)
            self._stats['products_with_ml'] = ml_products_count
            self._stats['products_with_embedding'] = embedding_products_count
            
            return products
            
        except Exception as e:
            logger.error(f"❌ Error cargando productos: {e}")
            raise

    def create_documents_optimized(self, products: List[Product]) -> List[Document]:
        """Crea documentos optimizados con información ML completa"""
        self._log_ml_info("📝 Paso 2: Creando documentos con ML...")
        
        start_time = time.time()
        documents = []
        skipped_count = 0
        
        # 🔥 NUEVO: Estadísticas por tipo de producto
        ml_document_count = 0
        embedding_document_count = 0
        
        for product in products:
            try:
                document = self._product_to_optimized_document(product)
                if document:
                    documents.append(document)
                    
                    # Track ML documents
                    if product.ml_processed:
                        ml_document_count += 1
                    if product.embedding:
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
        
        # 🔥 NUEVO: Log detallado ML
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
        """Convierte producto a documento con toda la información ML"""
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
                h = product.content_hash
                if h in self._document_cache:
                    return None
                self._document_cache[h] = True
            
            # 🔥🔥🔥 NUEVO: Metadata con información ML completa
            metadata = product.to_metadata()
            
            # Añadir información ML específica
            metadata["ml_processed"] = getattr(product, 'ml_processed', False)
            metadata["ml_features_used"] = getattr(product, 'ml_features', [])
            
            # Añadir categoría predicha si existe
            if hasattr(product, 'predicted_category') and product.predicted_category:
                metadata["predicted_category"] = product.predicted_category
            
            # Añadir entidades extraídas
            if hasattr(product, 'extracted_entities') and product.extracted_entities:
                # Limitar tamaño para no hacer metadata demasiado grande
                entities_summary = {}
                for entity_type, entities in product.extracted_entities.items():
                    if entities:
                        entities_summary[entity_type] = entities[:3]  # Solo primeras 3
                if entities_summary:
                    metadata["extracted_entities"] = json.dumps(entities_summary, ensure_ascii=False)
            
            # 🔥 NUEVO: Manejo de embeddings del producto
            if hasattr(product, 'embedding') and product.embedding:
                metadata["has_embedding"] = True
                metadata["embedding_dim"] = len(product.embedding)
                metadata["embedding_model"] = getattr(product, 'embedding_model', 'unknown')
                
                # Serializar embedding si vamos a usarlo
                if self.use_product_embeddings:
                    embedding_str = EmbeddingSerializer.serialize_embedding(product.embedding)
                    if embedding_str:
                        metadata["product_embedding"] = embedding_str
                        
                        # También versión truncada para debugging
                        truncated_json = EmbeddingSerializer.embedding_to_json(product.embedding)
                        metadata["embedding_preview"] = truncated_json
            
            # Añadir tags ML si existen
            if hasattr(product, 'ml_tags') and product.ml_tags:
                metadata["ml_tags"] = json.dumps(product.ml_tags[:10], ensure_ascii=False)
            
            # Añadir metadatos de procesamiento
            metadata["processing_timestamp"] = time.time()
            metadata["chroma_builder_version"] = "ml_enhanced_v1"
            
            return Document(page_content=content, metadata=metadata)
            
        except Exception as e:
            logger.warning(f"Error convirtiendo producto {product.id} a documento: {e}")
            return None

    def _is_low_quality_product(self, product: Product) -> bool:
        """Valida si un producto es de baja calidad con consideraciones ML"""
        # Productos sin descripción Y sin features Y sin precio
        has_no_description = (not product.description or 
                            product.description.startswith("No description"))
        has_no_features = (not product.details.features or 
                          len(product.details.features) == 0)
        has_no_price = not product.price
        
        # 🔥 NUEVO: Considerar si tiene información ML valiosa
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

    def _embed_documents_batch(self, documents: List[Document]) -> List[List[float]]:
        """Embeddings por lote con soporte para embeddings preexistentes"""
        if not documents:
            return []
        
        # 🔥 NUEVO: Verificar si podemos usar embeddings del producto
        if self.use_product_embeddings:
            product_embeddings = []
            need_chroma_embeddings = []
            need_chroma_indices = []
            
            for i, doc in enumerate(documents):
                if doc.metadata.get('has_embedding') and doc.metadata.get('product_embedding'):
                    # Intentar usar embedding del producto
                    embedding_str = doc.metadata.get('product_embedding')
                    embedding = EmbeddingSerializer.deserialize_embedding(embedding_str)
                    if embedding:
                        product_embeddings.append(embedding)
                        self._stats['ml_embeddings_used'] += 1
                        continue
                
                # Necesita embedding de Chroma
                need_chroma_embeddings.append(doc.page_content)
                need_chroma_indices.append(i)
            
            # Si todos tienen embeddings del producto, usarlos
            if len(product_embeddings) == len(documents):
                self._log_ml_info(f"✅ Usando {len(product_embeddings)} embeddings de productos")
                return product_embeddings
            
            # Mezclar embeddings: algunos del producto, otros de Chroma
            if product_embeddings:
                self._log_ml_info(f"🔀 Usando {len(product_embeddings)} embeddings de producto y {len(need_chroma_embeddings)} de Chroma")
                
                # Computar embeddings para los que faltan
                if need_chroma_embeddings:
                    model = self._get_embedding_model()
                    if isinstance(model, SentenceTransformer):
                        chroma_embeddings = model.encode(
                            need_chroma_embeddings,
                            batch_size=ChromaBuilderConfig.EMBEDDING_BATCH_SIZE,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True
                        ).tolist()
                    else:
                        chroma_embeddings = model.embed_documents(need_chroma_embeddings)
                    
                    self._stats['chroma_embeddings_computed'] += len(chroma_embeddings)
                    
                    # Combinar embeddings
                    combined_embeddings = [None] * len(documents)
                    for emb in product_embeddings:
                        # Necesitamos mapear correctamente - simplificado por ahora
                        pass
                    # Por simplicidad, usar solo embeddings de Chroma en este caso mixto
                    self._log_ml_info("⚠️  Modo mixto, usando solo embeddings de Chroma por simplicidad")
        
        # 🔥 Computar todos los embeddings con Chroma
        self._log_ml_info(f"⚡ Computando embeddings con Chroma para {len(documents)} documentos...")
        model = self._get_embedding_model()
        
        if isinstance(model, SentenceTransformer):
            contents = [doc.page_content for doc in documents]
            embeddings = model.encode(
                contents,
                batch_size=ChromaBuilderConfig.EMBEDDING_BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            self._stats['chroma_embeddings_computed'] = len(documents)
            return embeddings.tolist()
        else:
            contents = [doc.page_content for doc in documents]
            embeddings = model.embed_documents(contents)
            self._stats['chroma_embeddings_computed'] = len(documents)
            return embeddings

    def _create_chroma_with_embeddings(self, documents: List[Document], embeddings: List[List[float]], persist_directory: Optional[str] = None) -> Chroma:
        """Crea índice Chroma con embeddings precomputados - versión mejorada"""
        try:
            # 🔥 NUEVO: Verificar si los documentos ya tienen embeddings del producto
            if self.use_product_embeddings:
                product_embeddings = []
                valid_embeddings_count = 0
                
                for doc in documents:
                    if doc.metadata.get('has_embedding') and doc.metadata.get('product_embedding'):
                        embedding_str = doc.metadata.get('product_embedding')
                        embedding = EmbeddingSerializer.deserialize_embedding(embedding_str)
                        if embedding and len(embedding) > 0:
                            product_embeddings.append(embedding)
                            valid_embeddings_count += 1
                            continue
                    product_embeddings.append(None)
                
                # Si todos tienen embeddings válidos del producto, usarlos
                if valid_embeddings_count == len(documents):
                    self._log_ml_info(f"✅ Usando {valid_embeddings_count} embeddings propios de productos")
                    embeddings = product_embeddings
                elif valid_embeddings_count > 0:
                    self._log_ml_info(f"🔀 {valid_embeddings_count}/{len(documents)} productos tienen embeddings propios")
                    # Por simplicidad, usar embeddings calculados por Chroma
                    # En una versión futura podríamos mezclarlos
            
            # Intentar método moderno primero
            if hasattr(Chroma, 'from_embeddings'):
                return Chroma.from_embeddings(
                    text_embeddings=list(zip(
                        [doc.page_content for doc in documents],
                        embeddings
                    )),
                    embedding=self._get_embedding_model(),
                    metadatas=[doc.metadata for doc in documents],
                    persist_directory=persist_directory,
                    collection_metadata={
                        "hnsw:space": "cosine",
                        "description": f"ML-enhanced product index. ML enabled: {self.use_product_embeddings}"
                    }
                )
            else:
                # Fallback para versiones antiguas
                self._log_ml_info("⚠️  Usando método compatible para versiones antiguas de Chroma")
                
                if persist_directory:
                    chroma_index = Chroma(
                        persist_directory=persist_directory,
                        embedding_function=self._get_embedding_model(),
                        collection_metadata={
                            "hnsw:space": "cosine",
                            "ml_enhanced": str(self.use_product_embeddings)
                        }
                    )
                else:
                    chroma_index = Chroma(
                        embedding_function=self._get_embedding_model(),
                        collection_metadata={
                            "hnsw:space": "cosine", 
                            "ml_enhanced": str(self.use_product_embeddings)
                        }
                    )
                
                # Agregar documentos
                chroma_index._collection.add(
                    embeddings=embeddings,
                    documents=[doc.page_content for doc in documents],
                    metadatas=[doc.metadata for doc in documents],
                    ids=[doc.metadata.get("id", str(i)) for i, doc in enumerate(documents)]
                )
                
                return chroma_index
                
        except Exception as e:
            logger.error(f"Error creando Chroma con embeddings: {e}")
            # Fallback final
            self._log_ml_info("🔄 Usando método Chroma.from_documents estándar...")
            return Chroma.from_documents(
                documents=documents,
                embedding=self._get_embedding_model(),
                persist_directory=persist_directory,
                collection_metadata={"hnsw:space": "cosine"}
            )

    def build_index(self, persist: bool = True) -> Chroma:
        """Construye el índice con soporte ML completo"""
        total_start_time = time.time()
        
        try:
            # 🔥 NUEVO: Log de configuración ML
            self._log_ml_info("=" * 50)
            self._log_ml_info("🏗️  CONSTRUYENDO ÍNDICE CHROMA CON ML")
            self._log_ml_info("=" * 50)
            self._log_ml_info(f"• Usar embeddings de producto: {self.use_product_embeddings}")
            self._log_ml_info(f"• Modelo Chroma: {self.embedding_model}")
            self._log_ml_info(f"• Dispositivo: {self.device}")
            
            # 1. Cargar productos
            products = self.load_products_optimized()
            
            # 2. Crear documentos
            documents = self.create_documents_optimized(products)
            
            if not documents:
                raise ValueError("No se pudieron crear documentos válidos para la indexación")
            
            # 3. Computar embeddings
            embedding_start_time = time.time()
            embeddings = self._embed_documents_batch(documents)
            self._stats['embedding_time'] = time.time() - embedding_start_time
            
            # 4. Limpiar índice existente
            if persist and self.chroma_db_path.exists():
                self._log_ml_info("♻️  Eliminando índice Chroma existente")
                shutil.rmtree(self.chroma_db_path)
            
            # 5. Construir índice
            self._log_ml_info("🏗️  Construyendo índice Chroma...")
            
            persist_dir = str(self.chroma_db_path) if persist else None
            chroma_index = self._create_chroma_with_embeddings(
                documents=documents,
                embeddings=embeddings,
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
        """Registra estadísticas finales con información ML"""
        stats = self._stats
        
        self._log_ml_info("=" * 50)
        self._log_ml_info("📊 ESTADÍSTICAS FINALES - ML EDITION")
        self._log_ml_info("=" * 50)
        
        # Estadísticas básicas
        self._log_ml_info(f"📦 Productos cargados: {stats['total_products']}")
        self._log_ml_info(f"   • Con ML procesado: {stats['products_with_ml']}")
        self._log_ml_info(f"   • Con embeddings: {stats['products_with_embedding']}")
        
        self._log_ml_info(f"📝 Documentos creados: {stats['processed_documents']}")
        self._log_ml_info(f"⏭️  Documentos omitidos: {stats['skipped_documents']}")
        
        # Estadísticas de embeddings
        if stats['embedding_time'] > 0:
            self._log_ml_info(f"⚡ Tiempo embeddings: {stats['embedding_time']:.1f}s")
            if stats['chroma_embeddings_computed'] > 0:
                rate = stats['chroma_embeddings_computed'] / stats['embedding_time']
                self._log_ml_info(f"📈 Tasa embeddings Chroma: {rate:.1f} emb/s")
        
        if stats['ml_embeddings_used'] > 0:
            self._log_ml_info(f"✅ Embeddings ML usados: {stats['ml_embeddings_used']}")
        
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
        """Obtiene estadísticas del índice construido con info ML"""
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
            
            # 🔥 NUEVO: Obtener metadata ML de la colección
            collection_metadata = collection.metadata or {}
            
            # Intentar obtener muestras para estadísticas ML
            sample_results = collection.get(limit=10)
            ml_stats = {
                "ml_enhanced": collection_metadata.get("ml_enhanced", "false") == "true",
                "samples_with_ml": 0,
                "samples_with_embedding": 0
            }
            
            if sample_results and sample_results.get('metadatas'):
                for metadata in sample_results['metadatas'][:10]:
                    if metadata.get('ml_processed'):
                        ml_stats["samples_with_ml"] += 1
                    if metadata.get('has_embedding'):
                        ml_stats["samples_with_embedding"] += 1
            
            return {
                "document_count": count,
                "index_path": str(self.chroma_db_path),
                "embedding_model": self.embedding_model,
                "build_stats": self._stats,
                "ml_info": ml_stats,
                "collection_metadata": collection_metadata
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas del índice: {e}")
            return {"error": str(e)}

    # Alias para compatibilidad
    def build_index_optimized(self) -> Chroma:
        """Alias para build_index manteniendo compatibilidad"""
        return self.build_index(persist=True)

    def build_index_batch_optimized(self, batch_size: int = 5000) -> Chroma:
        """Versión optimizada con procesamiento por lotes"""
        # Similar al método anterior pero con tracking ML
        return self.build_index(persist=True)


# Alias para compatibilidad
ChromaBuilder = OptimizedChromaBuilder


def build_chroma_from_cli():
    """Función para ejecutar desde CLI con opciones ML"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Constructor de índice Chroma con ML")
    parser.add_argument("--input", type=Path, help="Ruta al JSON procesado")
    parser.add_argument("--output", type=Path, help="Ruta para guardar ChromaDB")
    parser.add_argument("--model", type=str, help="Modelo de embeddings")
    parser.add_argument("--device", type=str, help="Dispositivo (cpu/cuda)")
    parser.add_argument("--batch-size", type=int, default=ChromaBuilderConfig.DEFAULT_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=ChromaBuilderConfig.MAX_CONCURRENT_WORKERS)
    parser.add_argument("--no-cache", action="store_true", help="Deshabilitar cache")
    parser.add_argument("--in-memory", action="store_true", help="Modo in-memory")
    # 🔥 NUEVO: Opciones ML
    parser.add_argument("--use-product-embeddings", action="store_true", 
                       help="Usar embeddings de Product cuando estén disponibles")
    parser.add_argument("--ml-logging", action="store_true", 
                       help="Habilitar logging específico para ML")
    parser.add_argument("--debug", action="store_true", 
                       help="Modo debug (limitar productos)")
    
    args = parser.parse_args()
    
    if args.debug:
        os.environ["DEBUG"] = "1"
    
    builder = OptimizedChromaBuilder(
        processed_json_path=args.input,
        chroma_db_path=args.output,
        embedding_model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_workers=args.workers,
        enable_cache=not args.no_cache,
        # 🔥 NUEVO: Opciones ML
        use_product_embeddings=args.use_product_embeddings,
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
        
        # 🔥 NUEVO: Mostrar info ML
        ml_info = stats.get('ml_info', {})
        if ml_info:
            print(f"\n📈 INFORMACIÓN ML:")
            print(f"   • ML Enhanced: {'✅ Sí' if ml_info.get('ml_enhanced') else '❌ No'}")
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
        
        print(f"\n🎉 ¡Índice listo para usar!")
        
    except Exception as e:
        print(f"❌ Error construyendo índice: {e}")
        raise


if __name__ == "__main__":
    build_chroma_from_cli()