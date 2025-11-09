# src/core/data/chroma_builder.py

import os
import json
import logging
import shutil
import time
import re
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache
import gc

from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np

from src.core.data.product import Product
from src.core.config import settings
from src.core.utils.logger import get_logger

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
    
    # Configuración de calidad
    MIN_DOCUMENT_LENGTH = 10
    MAX_DOCUMENT_LENGTH = 2000

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
        compression: bool = True
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
            compression: Comprimir documentos para ahorrar espacio
        """
        self.processed_json_path = processed_json_path or settings.PROC_DIR / "products.json"
        self.chroma_db_path = chroma_db_path or Path(settings.CHROMA_DB_PATH)
        self.embedding_model = embedding_model or ChromaBuilderConfig.EMBEDDING_MODEL_NAME
        self.device = device or ChromaBuilderConfig.EMBEDDING_DEVICE
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.enable_cache = enable_cache
        self.compression = compression
        
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
                # Intentar usar SentenceTransformer directamente para mejor control
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

    @lru_cache(maxsize=ChromaBuilderConfig.DOCUMENT_CACHE_SIZE)
    def _get_document_hash(self, product_data: str) -> str:
        """Genera hash para detección de documentos duplicados"""
        return hashlib.md5(product_data.encode()).hexdigest()

    def load_products_optimized(self) -> List[Product]:
        """Carga optimizada de productos con procesamiento por lotes"""
        logger.info("📦 Paso 1: Cargando productos optimizado...")
        
        if not self.processed_json_path.exists():
            raise FileNotFoundError(f"Archivo procesado no encontrado: {self.processed_json_path}")
        
        start_time = time.time()
        
        try:
            with open(self.processed_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Modo DEBUG: Limitar productos
            if os.getenv("DEBUG"):
                original_count = len(data)
                data = data[:1000]
                logger.info(f"🔧 MODO DEBUG: Limitado a {len(data)} de {original_count} productos")
            
            # Procesar productos en lotes paralelos
            products = self._process_products_batch(data)
            
            load_time = time.time() - start_time
            logger.info(f"✅ {len(products)} productos cargados en {load_time:.2f}s")
            
            self._stats['total_products'] = len(products)
            return products
            
        except Exception as e:
            logger.error(f"❌ Error cargando productos: {e}")
            raise

    def _process_products_batch(self, data: List[Dict]) -> List[Product]:
        """Procesa productos en lotes paralelos"""
        if len(data) < 1000:  # Para datasets pequeños, procesar secuencialmente
            return [Product.from_dict(item) for item in data]
        
        # Para datasets grandes, usar procesamiento paralelo
        products = []
        chunks = self._split_into_chunks(data, self.batch_size)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk): chunk 
                for chunk in chunks
            }
            
            for future in tqdm(
                as_completed(future_to_chunk), 
                total=len(chunks),
                desc="Procesando productos",
                unit="chunk"
            ):
                try:
                    chunk_products = future.result()
                    products.extend(chunk_products)
                except Exception as e:
                    logger.warning(f"Error procesando chunk: {e}")
        
        return products

    def _process_chunk(self, chunk: List[Dict]) -> List[Product]:
        """Procesa un chunk de productos"""
        return [Product.from_dict(item) for item in chunk]

    def _split_into_chunks(self, data: List, chunk_size: int) -> List[List]:
        """Divide los datos en chunks del tamaño especificado"""
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    def create_documents_optimized(self, products: List[Product]) -> List[Document]:
        """Crea documentos optimizados con filtrado y cache"""
        logger.info("📄 Paso 2: Creando documentos optimizados...")
        
        start_time = time.time()
        documents = []
        
        # Procesar en lotes para mejor manejo de memoria
        for i in range(0, len(products), self.batch_size):
            batch = products[i:i + self.batch_size]
            batch_documents = self._process_batch_to_documents(batch)
            documents.extend(batch_documents)
            
            # Liberar memoria periódicamente
            if i > 0 and i % ChromaBuilderConfig.MEMORY_CHECK_INTERVAL == 0:
                gc.collect()
                logger.info(f"🧹 Liberada memoria después de {i} productos")
        
        process_time = time.time() - start_time
        logger.info(f"✅ {len(documents)} documentos creados en {process_time:.2f}s")
        logger.info(f"📊 {self._stats['skipped_documents']} documentos omitidos")
        
        self._stats['processed_documents'] = len(documents)
        return documents

    def _process_batch_to_documents(self, products: List[Product]) -> List[Document]:
        """Procesa un lote de productos a documentos"""
        documents = []
        
        for product in products:
            try:
                document = self._product_to_optimized_document(product)
                if document and self._is_valid_document(document):
                    documents.append(document)
                else:
                    self._stats['skipped_documents'] += 1
            except Exception as e:
                logger.debug(f"Error creando documento para producto {product.id}: {e}")
                self._stats['skipped_documents'] += 1
        
        return documents

    def _product_to_optimized_document(self, product: Product) -> Optional[Document]:
        """Convierte producto a documento optimizado"""
        # Generar contenido optimizado
        page_content = self._generate_optimized_content(product)
        
        if not page_content or len(page_content.strip()) < ChromaBuilderConfig.MIN_DOCUMENT_LENGTH:
            return None
        
        # Comprimir contenido si está habilitado
        if self.compression and len(page_content) > ChromaBuilderConfig.MAX_DOCUMENT_LENGTH:
            page_content = self._compress_content(page_content)
        
        # Verificar duplicados usando cache
        if self.enable_cache:
            content_hash = self._get_document_hash(page_content)
            if content_hash in self._document_cache:
                return None
            self._document_cache[content_hash] = True
        
        # Crear metadata optimizada
        metadata = self._generate_optimized_metadata(product)
        
        return Document(page_content=page_content, metadata=metadata)

    def _generate_optimized_content(self, product: Product) -> str:
        """Genera contenido optimizado para embeddings"""
        content_parts = []
        
        # Campos prioritarios (alto valor semántico)
        if product.title:
            content_parts.append(product.title)
        
        if product.description and product.description != "No description available":
            # Limitar longitud de descripción
            desc = product.description
            if len(desc) > 500:
                desc = desc[:497] + "..."
            content_parts.append(desc)
        
        # Campos secundarios (valor medio)
        if product.details and product.details.features:
            features_text = ". ".join(product.details.features[:5])  # Limitar features
            content_parts.append(features_text)
        
        if product.tags:
            tags_text = " ".join(product.tags[:10])  # Limitar tags
            content_parts.append(tags_text)
        
        # Campos terciarios (bajo valor)
        if product.compatible_devices:
            devices_text = " ".join(product.compatible_devices)
            content_parts.append(devices_text)
        
        # Campos de detalles específicos
        if product.details and product.details.specifications:
            key_specs = self._extract_key_specifications(product.details.specifications)
            if key_specs:
                content_parts.append(key_specs)
        
        content = ". ".join(filter(None, content_parts))
        
        # Limitar longitud total
        if len(content) > ChromaBuilderConfig.MAX_DOCUMENT_LENGTH:
            content = content[:ChromaBuilderConfig.MAX_DOCUMENT_LENGTH - 3] + "..."
        
        return content

    def _extract_key_specifications(self, specs: Dict[str, Any]) -> str:
        """Extrae especificaciones clave automáticamente"""
        key_specs = []
        priority_keys = {'brand', 'model', 'color', 'weight', 'size', 'dimensions', 'material'}
        
        for key, value in specs.items():
            key_lower = key.lower()
            if any(priority in key_lower for priority in priority_keys):
                key_specs.append(f"{key}: {value}")
        
        return ". ".join(key_specs[:5])  # Limitar a 5 especificaciones clave

    def _compress_content(self, content: str) -> str:
        """Comprime contenido manteniendo información clave"""
        # Estrategias de compresión:
        # 1. Remover espacios extra
        compressed = re.sub(r'\s+', ' ', content)
        
        # 2. Limitar oraciones muy largas
        sentences = compressed.split('. ')
        if len(sentences) > 10:
            compressed = '. '.join(sentences[:8]) + '...'
        
        # 3. Remover palabras muy comunes si es necesario
        if len(compressed) > ChromaBuilderConfig.MAX_DOCUMENT_LENGTH:
            words = compressed.split()
            if len(words) > 300:
                compressed = ' '.join(words[:250]) + '...'
        
        return compressed

    def _generate_optimized_metadata(self, product: Product) -> Dict[str, Any]:
        """Genera metadata optimizada para filtrado"""
        metadata = {
            "id": product.id,
            "title": product.title[:100] if product.title else "Unknown",
            "product_type": product.product_type or "unknown",
            "main_category": product.main_category or "uncategorized",
            "price": float(product.price) if product.price else 0.0,
            "rating": float(product.average_rating) if product.average_rating else 0.0,
            "content_hash": product.content_hash or "",
        }
        
        # Agregar campos adicionales si existen
        if product.details:
            if product.details.brand:
                metadata["brand"] = product.details.brand[:50]
            if product.details.model:
                metadata["model"] = product.details.model[:50]
        
        # Agregar confidence scores si están disponibles
        if hasattr(product, 'auto_category_confidence') and product.auto_category_confidence:
            metadata["category_confidence"] = float(product.auto_category_confidence)
        
        return metadata

    def _is_valid_document(self, document: Document) -> bool:
        """Valida si un documento es adecuado para indexación"""
        if not document.page_content or not document.page_content.strip():
            return False
        
        if len(document.page_content.strip()) < ChromaBuilderConfig.MIN_DOCUMENT_LENGTH:
            return False
        
        if len(document.page_content) > ChromaBuilderConfig.MAX_DOCUMENT_LENGTH:
            return False
        
        # Verificar que no sea principalmente placeholder text
        placeholder_indicators = [
            "no description available",
            "unknown product",
            "price not available"
        ]
        
        content_lower = document.page_content.lower()
        if any(indicator in content_lower for indicator in placeholder_indicators):
            # Solo rechazar si es principalmente placeholder
            meaningful_content = content_lower
            for indicator in placeholder_indicators:
                meaningful_content = meaningful_content.replace(indicator, "")
            
            if len(meaningful_content.strip()) < ChromaBuilderConfig.MIN_DOCUMENT_LENGTH:
                return False
        
        return True

    def build_index_optimized(self) -> Chroma:
        """Construye el índice Chroma optimizado"""
        total_start_time = time.time()
        logger.info("🚀 Iniciando proceso de indexación optimizado...")
        
        try:
            # 1. Cargar productos optimizado
            products = self.load_products_optimized()
            
            # 2. Crear documentos optimizados
            documents = self.create_documents_optimized(products)
            
            if not documents:
                raise ValueError("No se pudieron crear documentos válidos para la indexación")
            
            # 3. Configurar embeddings
            embeddings = self._get_embedding_model()
            
            # 4. Limpiar índice existente
            if self.chroma_db_path.exists():
                logger.warning("♻️ Eliminando índice Chroma existente")
                shutil.rmtree(self.chroma_db_path)
            
            # 5. Construir índice por lotes
            logger.info("🏗️ Construyendo índice Chroma optimizado...")
            embedding_start_time = time.time()
            
            chroma_index = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=str(self.chroma_db_path),
                collection_metadata={"hnsw:space": "cosine"}
            )
            
            self._stats['embedding_time'] = time.time() - embedding_start_time
            
            # 6. Estadísticas finales
            total_time = time.time() - total_start_time
            self._stats['total_time'] = total_time
            
            self._log_final_stats()
            
            return chroma_index
            
        except Exception as e:
            logger.error(f"❌ Error en la indexación optimizada: {e}")
            raise

    def build_index_batch_optimized(self, batch_size: int = 5000) -> Chroma:
        """Versión optimizada con procesamiento por lotes para grandes datasets"""
        total_start_time = time.time()
        
        # Cargar todos los productos primero
        products = self.load_products_optimized()
        total_products = len(products)
        logger.info(f"📦 Total de productos a indexar: {total_products}")
        
        # Configurar embeddings
        embeddings = self._get_embedding_model()
        
        # Limpiar índice existente
        if self.chroma_db_path.exists():
            shutil.rmtree(self.chroma_db_path)
        
        # Procesar por lotes optimizados
        chroma_index = None
        
        for batch_start in range(0, total_products, batch_size):
            batch_end = min(batch_start + batch_size, total_products)
            batch_products = products[batch_start:batch_end]
            
            logger.info(f"🔄 Procesando lote {batch_start+1}-{batch_end}/{total_products}")
            
            # Convertir a documentos optimizados
            documents = self._process_batch_to_documents(batch_products)
            
            if not documents:
                logger.warning(f"⚠️ Lote {batch_start+1}-{batch_end} no produjo documentos válidos")
                continue
            
            # Crear o añadir al índice
            if chroma_index is None:
                chroma_index = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=str(self.chroma_db_path)
                )
            else:
                chroma_index.add_documents(documents)
            
            # Liberar memoria
            del documents
            gc.collect()
        
        # Estadísticas finales
        total_time = time.time() - total_start_time
        logger.info(f"✅ Indexación por lotes completada en {total_time:.2f} segundos")
        
        return chroma_index

    def _log_final_stats(self):
        """Registra estadísticas finales del proceso"""
        stats = self._stats
        logger.info("📊 ESTADÍSTICAS FINALES DE INDEXACIÓN:")
        logger.info(f"   • Total de productos: {stats['total_products']}")
        logger.info(f"   • Documentos procesados: {stats['processed_documents']}")
        logger.info(f"   • Documentos omitidos: {stats['skipped_documents']}")
        logger.info(f"   • Tiempo total: {stats['total_time']:.2f}s")
        logger.info(f"   • Tiempo de embeddings: {stats['embedding_time']:.2f}s")
        
        if stats['total_products'] > 0:
            efficiency = stats['processed_documents'] / stats['total_products'] * 100
            logger.info(f"   • Eficiencia: {efficiency:.1f}%")
            
            rate = stats['processed_documents'] / stats['total_time'] if stats['total_time'] > 0 else 0
            logger.info(f"   • Tasa: {rate:.2f} doc/s")

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
    parser.add_argument("--no-compression", action="store_true", help="Deshabilitar compresión")
    
    args = parser.parse_args()
    
    builder = OptimizedChromaBuilder(
        processed_json_path=args.input,
        chroma_db_path=args.output,
        embedding_model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_workers=args.workers,
        enable_cache=not args.no_cache,
        compression=not args.no_compression
    )
    
    index = builder.build_index_optimized()
    stats = builder.get_index_stats()
    
    print(f"✅ Índice construido exitosamente:")
    print(f"   • Documentos: {stats.get('document_count', 'N/A')}")
    print(f"   • Ubicación: {stats.get('index_path', 'N/A')}")
    print(f"   • Modelo: {stats.get('embedding_model', 'N/A')}")


if __name__ == "__main__":
    build_chroma_from_cli()