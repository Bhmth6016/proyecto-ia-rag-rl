import json
import os
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from src.utils.logger_setup import configurar_logger
logger = configurar_logger()

def convertir_jsonl_a_json(ruta_entrada, ruta_salida):
    productos = []
    with open(ruta_entrada, 'r', encoding='utf-8') as f:
        for linea in f:
            try:
                productos.append(json.loads(linea.strip()))
            except json.JSONDecodeError:
                logger.warning("Línea inválida omitida: %s", linea.strip())
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    with open(ruta_salida, 'w', encoding='utf-8') as out_file:
        json.dump(productos, out_file, indent=2, ensure_ascii=False)
        logger.info("✅ Conversión completada: %d productos → %s", len(productos), ruta_salida)

def producto_a_documento(producto):
    detalles = producto.get("details", {})
    texto = f"Título: {producto.get('title', '')}.\n"
    texto += f"Categoría: {producto.get('main_category', '')}.\n"
    texto += f"Marca: {detalles.get('Brand', '')}.\n"
    texto += f"Precio: {producto.get('price', 'No disponible')}.\n"
    texto += f"Valoración: {producto.get('average_rating', '')} en {producto.get('rating_number', '')} reseñas.\n"
    texto += f"Descripción técnica: {', '.join([f'{k}: {v}' for k, v in detalles.items()])}"
    return Document(page_content=texto, metadata={"title": producto.get("title", "Sin título")})

def indexar_productos(json_path, persist_directory="chroma_index"):
    if not os.path.exists(json_path):
        logger.error("Archivo JSON no encontrado en: %s", json_path)
        return

    with open(json_path, "r", encoding="utf-8") as f:
        productos = json.load(f)

    documentos = [producto_a_documento(prod) for prod in productos]

    vectordb = Chroma.from_documents(
        documents=documentos,
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        persist_directory=persist_directory
    )

    
    logger.info("📦 Indexación completada con %d documentos en '%s'.", len(documentos), persist_directory)