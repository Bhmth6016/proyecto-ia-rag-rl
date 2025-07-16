import json
import os
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.core.utils.logger_setup import configurar_logger

logger = configurar_logger()

class Producto:
    def __init__(self, data: Dict[str, Any]):
        self.title = data.get('title', '')
        self.main_category = data.get('main_category', '')
        self.price = data.get('price', 0)
        self.details = data.get('details', {})
        self.average_rating = data.get('average_rating', 0)
        self.rating_number = data.get('rating_number', 0)

def convertir_jsonl_a_json(ruta_entrada: str, ruta_salida: str) -> None:
    """Convierte un archivo JSONL a JSON estructurado."""
    productos = []
    try:
        with open(ruta_entrada, 'r', encoding='utf-8') as f:
            for linea in f:
                try:
                    productos.append(json.loads(linea.strip()))
                except json.JSONDecodeError as e:
                    logger.warning(f"Error al decodificar línea: {e}. Línea omitida.")
        
        Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
        
        with open(ruta_salida, 'w', encoding='utf-8') as out_file:
            json.dump(productos, out_file, indent=2, ensure_ascii=False)
            logger.info(f"✅ Conversión completada: {len(productos)} productos → {ruta_salida}")
            
    except Exception as e:
        logger.error(f"Error en convertir_jsonl_a_json: {e}")
        raise

def producto_a_documento(producto: Dict[str, Any]) -> Document:
    """Convierte un diccionario de producto a Document de LangChain."""
    detalles = producto.get("details", {})
    texto = (
        f"Título: {producto.get('title', '')}.\n"
        f"Categoría: {producto.get('main_category', '')}.\n"
        f"Marca: {detalles.get('Brand', '')}.\n"
        f"Precio: {producto.get('price', 'No disponible')}.\n"
        f"Valoración: {producto.get('average_rating', '')} en {producto.get('rating_number', '')} reseñas.\n"
        f"Descripción técnica: {', '.join([f'{k}: {v}' for k, v in detalles.items()])}"
    )
    return Document(
        page_content=texto,
        metadata={
            "title": producto.get("title", "Sin título"),
            "category": producto.get("main_category", ""),
            "price": producto.get("price", 0)
        }
    )

def indexar_productos(json_path: str, persist_directory: str = "chroma_index") -> None:
    """Indexa productos desde un archivo JSON usando ChromaDB."""
    try:
        if not Path(json_path).exists():
            raise FileNotFoundError(f"Archivo JSON no encontrado en: {json_path}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            productos = json.load(f)
        
        documentos = [producto_a_documento(prod) for prod in productos]
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectordb = Chroma.from_documents(
            documents=documentos,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        
        logger.info(f"📦 Indexación completada con {len(documentos)} documentos en '{persist_directory}'.")
        
    except Exception as e:
        logger.error(f"Error en indexar_productos: {e}")
        raise