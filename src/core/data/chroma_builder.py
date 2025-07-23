# src/core/data/chroma_builder.py
import os
import json
import logging
import shutil
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Optional, List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # Actualizado el import

from src.core.data.product import Product
from src.core.config import settings
from src.core.utils.logger import get_logger

logger = get_logger(__name__)

class ChromaBuilder:
    def __init__(
        self,
        *,
        processed_json_path: Optional[Path] = None,
        chroma_db_path: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Args:
            processed_json_path: Ruta al JSON unificado de productos
            chroma_db_path: Donde se guardará el índice Chroma
            embedding_model: Modelo de embeddings a usar
            device: Dispositivo para los embeddings (cpu/cuda)
        """
        self.processed_json_path = processed_json_path or settings.PROC_DIR / "products.json"
        self.chroma_db_path = chroma_db_path or Path(settings.CHROMA_DB_PATH)
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.device = device or settings.DEVICE

    def load_products(self) -> List[Product]:
        """Carga y valida los productos desde el JSON procesado"""
        logger.info("Paso 1: Cargando productos...")
        if not self.processed_json_path.exists():
            raise FileNotFoundError(f"Archivo procesado no encontrado: {self.processed_json_path}")
        
        with open(self.processed_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        products = [Product.from_dict(item) for item in data]
        
        # Modo DEBUG: Procesar solo 100 productos
        if os.getenv("DEBUG"):
            products = products[:100000]
        
        logger.info(f"✅ Cargados {len(products)} productos")
        return products

    def create_documents(self, products: List[Product]) -> List[Document]:
        """Convierte Productos a Documents para Chroma"""
        logger.info("Paso 2: Creando documentos...")
        documents = []
        for product in tqdm(products, desc="Creando documentos", unit="doc"):
            logger.debug(f"Creando documento para producto: {product.id}")
            page_content = self._generate_page_content(product)
            metadata = product.to_metadata()
            documents.append(Document(page_content=page_content, metadata=metadata))
        logger.info(f"✅ Documentos creados: {len(documents)}")
        return documents

    def _generate_page_content(self, product: Product) -> str:
        """Genera el texto para embeddings a partir de un Producto"""
        logger.debug(f"Generando contenido para producto: {product.id}")
        content_parts = [
            product.title,
            product.description or "",
            " ".join(product.tags),
            " ".join(product.compatible_devices),
            product.details.features if product.details else ""
        ]
        content = "\n".join(filter(None, content_parts))
        logger.debug(f"Contenido generado para producto {product.id}: {content}")
        return content

    def build_index(self) -> Chroma:
        """Construye el índice Chroma completo"""
        import time
        start_time = time.time()
        
        logger.info("🚀 Iniciando proceso de indexación...")
        
        # 1. Cargar productos con barra de progreso
        logger.info("📦 Paso 1/5: Cargando productos...")
        products = self.load_products()
        logger.info(f"✅ {len(products)} productos cargados")
        
        # 2. Convertir a documentos
        logger.info("📄 Paso 2/5: Creando documentos...")
        documents = []
        for i, product in enumerate(tqdm(products, desc="Creando documentos"), 1):
            if i % 1000 == 0:
                logger.info(f"📝 Procesados {i}/{len(products)} productos")
            documents.append(self._product_to_document(product))
        
        # 3. Configurar embeddings
        logger.info("🧠 Paso 3/5: Configurando embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs={
                "batch_size": 64,
                "num_workers": 4 if self.device == "cpu" else 0
            }
        )
        
        # 4. Limpiar índice existente
        if self.chroma_db_path.exists():
            logger.warning("♻️ Eliminando índice Chroma existente")
            shutil.rmtree(self.chroma_db_path)
        
        # 5. Construir ChromaDB
        logger.info("🏗️ Paso 4/5: Construyendo índice Chroma...")
        chroma_index = Chroma.from_documents(
            documents=tqdm(documents, desc="Indexando", unit="doc"),
            embedding=embeddings,
            persist_directory=str(self.chroma_db_path)
        )
        
        # Estadísticas finales
        total_time = time.time() - start_time
        logger.info(f"🎉 Indexación completada en {total_time:.2f} segundos")
        logger.info(f"📊 Tasa de indexación: {len(products)/total_time:.2f} productos/segundo")
        
        return chroma_index
    
    def build_index_batch(self, batch_size: int = 5000) -> Chroma:
        """Versión con procesamiento por lotes para grandes datasets"""
        import time
        start_time = time.time()
        
        # Cargar todos los productos primero
        products = self.load_products()
        total_products = len(products)
        logger.info(f"📦 Total de productos a indexar: {total_products}")
        
        # Configurar embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"batch_size": 64}
        )
        
        # Limpiar índice existente
        if self.chroma_db_path.exists():
            shutil.rmtree(self.chroma_db_path)
        
        # Procesar por lotes
        for batch_start in range(0, total_products, batch_size):
            batch_end = min(batch_start + batch_size, total_products)
            batch = products[batch_start:batch_end]
            
            logger.info(f" Procesando lote {batch_start+1}-{batch_end}/{total_products}")
            
            # Convertir a documentos
            documents = [self._product_to_document(p) for p in batch]
            
            # Crear o añadir al índice
            if batch_start == 0:
                chroma_index = Chroma.from_documents(
                    documents=documents,
                    embedding=embeddings,
                    persist_directory=str(self.chroma_db_path)
                )
            else:
                chroma_index.add_documents(documents)
        
        # Estadísticas finales
        total_time = time.time() - start_time
        logger.info(f"✅ Indexación completada en {total_time:.2f} segundos")
        return chroma_index

def build_chroma_from_cli():
    """Función para ejecutar desde CLI"""
    import argparse
    parser = argparse.ArgumentParser(description="Constructor de índice Chroma")
    parser.add_argument("--input", type=Path, help="Ruta al JSON procesado")
    parser.add_argument("--output", type=Path, help="Ruta para guardar ChromaDB")
    parser.add_argument("--model", type=str, help="Modelo de embeddings")
    parser.add_argument("--device", type=str, help="Dispositivo (cpu/cuda)")
    args = parser.parse_args()
    
    builder = ChromaBuilder(
        processed_json_path=args.input,
        chroma_db_path=args.output,
        embedding_model=args.model,
        device=args.device
    )
    builder.build_index()

if __name__ == "__main__":
    build_chroma_from_cli()