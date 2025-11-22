# src/core/data/chroma_builder.py

import os
import json
import logging
import shutil
import time
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
        compression: bool = False
    ):
        """
        Constructor optimizado para ChromaDB.
        
        Args:
            processed_json_path: Ruta al JSON unificado de productos
            chroma_db_path: Donde se guardará el índice Chroma
            embedding_model: Modelo de embeddings a usar
            device: Dispositivo para los embeddings (cpu/cuda)
            batch_size: Tamaño de lote para procesamiento
            max_workers: Número máximo de workers paralelos
            enable_cache: Habilitar cache de documentos
            compression: OBSOLETO - mantener por compatibilidad
        """
        self.processed_json_path = processed_json_path or settings.PROC_DIR / "products.json"
        self.chroma_db_path = chroma_db_path or Path(settings.CHROMA_DB_PATH)
        self.embedding_model = embedding_model or ChromaBuilderConfig.EMBEDDING_MODEL_NAME
        self.device = device or ChromaBuilderConfig.EMBEDDING_DEVICE
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.enable_cache = enable_cache
        
        # Cache para documentos procesados
        self._document_cache = {}
        self._embedding_model = None
        self._stats = {
            'total_products': 0,
            'processed_documents': 0,
            'skipped_documents': 0,
            'total_time': 0,
            'embedding_time': 0
        }

    def _get_embedding_model(self):
        """Obtiene el modelo de embeddings con lazy loading"""
        if self._embedding_model is None:
            logger.info("🔄 Cargando modelo de embeddings...")
            start_time = time.time()
            
            try:
                # Usar SentenceTransformer directamente para mejor control
                self._embedding_model = SentenceTransformer(
                    self.embedding_model,
                    device=self.device
                )
            except Exception as e:
                logger.warning(f"Error con SentenceTransformer, usando HuggingFaceEmbeddings: {e}")
                self._embedding_model = HuggingFaceEmbeddings(
                    model_name=self.embedding_model,
                    model_kwargs={"device": self.device},
                    encode_kwargs={
                        "batch_size": ChromaBuilderConfig.EMBEDDING_BATCH_SIZE,
                        "normalize_embeddings": True
                    }
                )
            
            load_time = time.time() - start_time
            logger.info(f"✅ Modelo de embeddings cargado en {load_time:.2f}s")
        
        return self._embedding_model

    def _ensure_data_loaded(self):
        """Asegura que los datos estén cargados, ejecutando FastDataLoader si es necesario"""
        if not self.processed_json_path.exists():
            logger.warning("📦 Archivo procesado no encontrado. Ejecutando FastDataLoader automáticamente...")
            loader = FastDataLoader(use_progress_bar=True)
            loader.load_data(self.processed_json_path)
            
            if not self.processed_json_path.exists():
                raise FileNotFoundError(f"No se pudo crear el archivo procesado: {self.processed_json_path}")

    def load_products_optimized(self) -> List[Product]:
        """Carga optimizada de productos"""
        logger.info("🔵 CHROMA BUILDER STARTED")
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
            
            # Procesamiento secuencial optimizado - más rápido que threading para Python puro
            products = [Product.from_dict(item) for item in data]
            
            load_time = time.time() - start_time
            logger.info(f"✅ {len(products)} productos cargados en {load_time:.2f}s")
            
            self._stats['total_products'] = len(products)
            return products
            
        except Exception as e:
            logger.error(f"❌ Error cargando productos: {e}")
            raise

    def create_documents_optimized(self, products: List[Product]) -> List[Document]:
        """Crea documentos optimizados usando los métodos nativos de Product"""
        logger.info("📝 Paso 2: Creando documentos...")
        
        start_time = time.time()
        documents = []
        skipped_count = 0
        
        # Procesamiento secuencial más eficiente
        for product in products:
            try:
                document = self._product_to_optimized_document(product)
                if document:
                    documents.append(document)
                else:
                    skipped_count += 1
            except Exception as e:
                logger.debug(f"Error creando documento para producto {product.id}: {e}")
                skipped_count += 1
            
            # Liberar memoria periódicamente
            if len(documents) % ChromaBuilderConfig.MEMORY_CHECK_INTERVAL == 0:
                gc.collect()
        
        process_time = time.time() - start_time
        logger.info(f"📝 {len(documents)} documentos creados en {process_time:.2f}s")
        logger.info(f"⏭️  {skipped_count} documentos omitidos")
        
        self._stats['processed_documents'] = len(documents)
        self._stats['skipped_documents'] = skipped_count
        return documents

    def _product_to_optimized_document(self, product: Product) -> Optional[Document]:
        """Convierte producto a documento usando los métodos nativos de Product"""
        try:
            # MEJORA 1: Usar los métodos nativos de Product
            content = product.to_text()
            if not content or len(content.strip()) < 10:
                return None
            
            # MEJORA 8: Validación adicional para productos basura
            if self._is_low_quality_product(product):
                return None
            
            # MEJORA 2: Usar content_hash para detección de duplicados
            if self.enable_cache:
                h = product.content_hash
                if h in self._document_cache:
                    return None
                self._document_cache[h] = True
            
            # MEJORA 1: Usar metadata nativa de Product
            metadata = product.to_metadata()
            
            return Document(page_content=content, metadata=metadata)
            
        except Exception as e:
            logger.warning(f"Error convirtiendo producto {product.id} a documento: {e}")
            return None

    def _is_low_quality_product(self, product: Product) -> bool:
        """Valida si un producto es de baja calidad (MEJORA 8)"""
        # Productos sin descripción Y sin features Y sin precio
        has_no_description = (not product.description or 
                            product.description.startswith("No description"))
        has_no_features = (not product.details.features or 
                          len(product.details.features) == 0)
        has_no_price = not product.price
        
        if has_no_description and has_no_features and has_no_price:
            return True
        
        # Productos con metadata extremadamente pobre
        metadata = product.to_metadata()
        if (metadata.get("price", 0) == 0 and 
            metadata.get("average_rating", 0) == 0 and 
            not metadata.get("features")):
            return True
        
        return False

    def _embed_documents_batch(self, documents: List[Document]) -> List[List[float]]:
        """MEJORA 4: Embeddings por lote para máximo rendimiento"""
        if not documents:
            return []
        
        # Usar SentenceTransformer directamente para embeddings por lote
        model = self._get_embedding_model()
        
        # Si es un SentenceTransformer, usar encode directamente
        if isinstance(model, SentenceTransformer):
            contents = [doc.page_content for doc in documents]
            embeddings = model.encode(
                contents,
                batch_size=ChromaBuilderConfig.EMBEDDING_BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        else:
            # Fallback para HuggingFaceEmbeddings
            contents = [doc.page_content for doc in documents]
            return model.embed_documents(contents)

    def _create_chroma_with_embeddings(self, documents: List[Document], embeddings: List[List[float]], persist_directory: Optional[str] = None) -> Chroma:
        """Crea índice Chroma con embeddings precomputados - compatible con versiones antiguas"""
        try:
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
                    collection_metadata={"hnsw:space": "cosine"}
                )
            else:
                # Fallback para versiones antiguas - crear documentos con embeddings
                logger.warning("⚠️  Usando método compatible para versiones antiguas de Chroma")
                
                # Crear índice vacío primero
                if persist_directory:
                    chroma_index = Chroma(
                        persist_directory=persist_directory,
                        embedding_function=self._get_embedding_model(),
                        collection_metadata={"hnsw:space": "cosine"}
                    )
                else:
                    chroma_index = Chroma(
                        embedding_function=self._get_embedding_model(),
                        collection_metadata={"hnsw:space": "cosine"}
                    )
                
                # Agregar documentos con embeddings precalculados
                chroma_index._collection.add(
                    embeddings=embeddings,
                    documents=[doc.page_content for doc in documents],
                    metadatas=[doc.metadata for doc in documents],
                    ids=[doc.metadata.get("id", str(i)) for i, doc in enumerate(documents)]
                )
                
                return chroma_index
                
        except Exception as e:
            logger.error(f"Error creando Chroma con embeddings: {e}")
            # Fallback final - usar método estándar
            logger.info("🔄 Usando método Chroma.from_documents estándar...")
            return Chroma.from_documents(
                documents=documents,
                embedding=self._get_embedding_model(),
                persist_directory=persist_directory,
                collection_metadata={"hnsw:space": "cosine"}
            )

    def build_index(self, persist: bool = True) -> Chroma:
        """MEJORA 9: Construye el índice con soporte para modo in-memory"""
        total_start_time = time.time()
        
        try:
            # 1. Cargar productos
            products = self.load_products_optimized()
            
            # 2. Crear documentos
            documents = self.create_documents_optimized(products)
            
            if not documents:
                raise ValueError("No se pudieron crear documentos válidos para la indexación")
            
            logger.info(f"⚡ Computando embeddings para {len(documents)} documentos...")
            
            # 3. Computar embeddings por lote (MEJORA 4)
            embedding_start_time = time.time()
            embeddings = self._embed_documents_batch(documents)
            self._stats['embedding_time'] = time.time() - embedding_start_time
            
            # 4. Limpiar índice existente si persistimos
            if persist and self.chroma_db_path.exists():
                logger.warning("♻️ Eliminando índice Chroma existente")
                shutil.rmtree(self.chroma_db_path)
            
            # 5. Construir índice con embeddings precomputados
            logger.info("🏗️ Construyendo índice Chroma...")
            
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

    def build_index_batch_optimized(self, batch_size: int = 5000) -> Chroma:
        """Versión optimizada con procesamiento por lotes para grandes datasets"""
        total_start_time = time.time()
        logger.info("🚀 Iniciando indexación por lotes optimizada...")
        
        # Cargar todos los productos primero
        products = self.load_products_optimized()
        total_products = len(products)
        logger.info(f"📦 Total de productos a indexar: {total_products}")
        
        # Limpiar índice existente
        if self.chroma_db_path.exists():
            shutil.rmtree(self.chroma_db_path)
        
        # Procesar por lotes optimizados
        chroma_index = None
        total_documents = 0
        
        for batch_start in range(0, total_products, batch_size):
            batch_end = min(batch_start + batch_size, total_products)
            batch_products = products[batch_start:batch_end]
            
            logger.info(f"🔄 Procesando lote {batch_start+1}-{batch_end}/{total_products}")
            
            # Convertir a documentos usando métodos nativos
            documents = []
            for product in batch_products:
                doc = self._product_to_optimized_document(product)
                if doc:
                    documents.append(doc)
            
            if not documents:
                logger.warning(f"⚠️ Lote {batch_start+1}-{batch_end} no produjo documentos válidos")
                continue
            
            # Computar embeddings por lote para este batch
            embeddings = self._embed_documents_batch(documents)
            
            # Crear o añadir al índice
            if chroma_index is None:
                chroma_index = self._create_chroma_with_embeddings(
                    documents=documents,
                    embeddings=embeddings,
                    persist_directory=str(self.chroma_db_path)
                )
            else:
                # Para batches adicionales, usar add_documents
                chroma_index.add_documents(documents)
            
            total_documents += len(documents)
            
            # Liberar memoria
            del documents, embeddings
            gc.collect()
        
        # Estadísticas finales
        total_time = time.time() - total_start_time
        logger.info(f"💾 Indexación completada → {self.chroma_db_path}")
        logger.info(f"✅ {total_documents} documentos indexados en {total_time:.2f}s")
        
        return chroma_index

    def _log_final_stats(self):
        """MEJORA 7: Registra estadísticas finales compactas y legibles"""
        stats = self._stats
        logger.info("📊 ESTADÍSTICAS FINALES:")
        logger.info(f"   • Productos cargados: {stats['total_products']}")
        logger.info(f"   • Documentos creados: {stats['processed_documents']}")
        logger.info(f"   • Documentos omitidos: {stats['skipped_documents']}")
        
        if stats['embedding_time'] > 0:
            logger.info(f"⚡ Embeddings computados en {stats['embedding_time']:.1f}s")
        
        if stats['total_time'] > 0:
            logger.info(f"⏱️  Tiempo total: {stats['total_time']:.1f}s")
            
            if stats['processed_documents'] > 0:
                rate = stats['processed_documents'] / stats['total_time']
                logger.info(f"📈 Tasa: {rate:.1f} doc/s")
        
        logger.info(f"💾 Índice guardado → {self.chroma_db_path}")

    def get_index_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del índice construido"""
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
            
            return {
                "document_count": count,
                "index_path": str(self.chroma_db_path),
                "embedding_model": self.embedding_model,
                "build_stats": self._stats
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas del índice: {e}")
            return {"error": str(e)}

    # Alias para compatibilidad con código existente
    def build_index_optimized(self) -> Chroma:
        """Alias para build_index manteniendo compatibilidad"""
        return self.build_index(persist=True)


# Alias para compatibilidad
ChromaBuilder = OptimizedChromaBuilder


def build_chroma_from_cli():
    """Función optimizada para ejecutar desde CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Constructor optimizado de índice Chroma")
    parser.add_argument("--input", type=Path, help="Ruta al JSON procesado")
    parser.add_argument("--output", type=Path, help="Ruta para guardar ChromaDB")
    parser.add_argument("--model", type=str, help="Modelo de embeddings")
    parser.add_argument("--device", type=str, help="Dispositivo (cpu/cuda)")
    parser.add_argument("--batch-size", type=int, default=ChromaBuilderConfig.DEFAULT_BATCH_SIZE, 
                       help="Tamaño de lote para procesamiento")
    parser.add_argument("--workers", type=int, default=ChromaBuilderConfig.MAX_CONCURRENT_WORKERS,
                       help="Número de workers paralelos")
    parser.add_argument("--no-cache", action="store_true", help="Deshabilitar cache")
    parser.add_argument("--in-memory", action="store_true", help="Modo in-memory (no persistir)")
    
    args = parser.parse_args()
    
    builder = OptimizedChromaBuilder(
        processed_json_path=args.input,
        chroma_db_path=args.output,
        embedding_model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_workers=args.workers,
        enable_cache=not args.no_cache
    )
    
    try:
        index = builder.build_index(persist=not args.in_memory)
        stats = builder.get_index_stats()
        
        print(f"✅ Índice construido exitosamente:")
        print(f"   • Documentos: {stats.get('document_count', 'N/A')}")
        print(f"   • Ubicación: {stats.get('index_path', 'N/A')}")
        print(f"   • Modelo: {stats.get('embedding_model', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error construyendo índice: {e}")
        raise


if __name__ == "__main__":
    build_chroma_from_cli()