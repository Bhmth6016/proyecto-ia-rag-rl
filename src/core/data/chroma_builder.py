# src/core/data/chroma_builder.py
"""
ChromaDB Index Builder

Proceso:
1. Carga el archivo JSON unificado de productos limpios
2. Genera embeddings con el modelo configurado
3. Construye el índice ChromaDB
4. Guarda el índice en la ubicación especificada
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

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
        if not self.processed_json_path.exists():
            raise FileNotFoundError(f"Archivo procesado no encontrado: {self.processed_json_path}")
        
        with open(self.processed_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [Product.from_dict(item) for item in data]

    def create_documents(self, products: List[Product]) -> List[Document]:
        """Convierte Productos a Documents para Chroma"""
        return [
            Document(
                page_content=self._generate_page_content(product),
                metadata=product.to_metadata()
            )
            for product in products
        ]

    def _generate_page_content(self, product: Product) -> str:
        """Genera el texto para embeddings a partir de un Producto"""
        content_parts = [
            product.title,
            product.description or "",
            " ".join(product.tags),
            " ".join(product.compatible_devices),
            product.details.features if product.details else ""
        ]
        return "\n".join(filter(None, content_parts))

    def build_index(self) -> Chroma:
        """Construye el índice Chroma completo"""
        # 1. Cargar productos
        products = self.load_products()
        logger.info(f"✅ Cargados {len(products)} productos validados")
        
        # 2. Convertir a documentos
        documents = self.create_documents(products)
        logger.info(f"🛠️ Generados {len(documents)} documentos para indexar")
        
        # 3. Configurar embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.device}
        )
        
        # 4. Construir ChromaDB
        self.chroma_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Limpiar índice existente si hay
        if self.chroma_db_path.exists():
            logger.warning("♻️ Eliminando índice Chroma existente")
            import shutil
            shutil.rmtree(self.chroma_db_path)
        
        logger.info("🏗️ Construyendo índice Chroma...")
        chroma_index = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(self.chroma_db_path)
        )
        
        logger.info(f"💾 Índice guardado en {self.chroma_db_path}")
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