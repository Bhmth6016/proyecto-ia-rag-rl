import os
import json
import sys
import logging
from typing import List, Dict, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("generator.log", encoding='utf-8'),
        # This will handle Unicode characters in Windows console
        logging.StreamHandler(stream=sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def convertir_jsonl_a_json(ruta_entrada: str, ruta_salida: str) -> List[Dict]:
    """
    Convierte archivo JSONL a JSON con manejo de errores
    
    Args:
        ruta_entrada: Ruta del archivo JSONL de entrada
        ruta_salida: Ruta donde guardar el JSON resultante
    
    Returns:
        Lista de productos procesados
    """
    productos = []
    try:
        with open(ruta_entrada, 'r', encoding='utf-8') as f:
            for i, linea in enumerate(f, 1):
                try:
                    producto = json.loads(linea.strip())
                    if isinstance(producto, dict):  # Validación básica
                        productos.append(producto)
                except json.JSONDecodeError:
                    logger.warning(f"Línea {i} inválida en {os.path.basename(ruta_entrada)}")
                except Exception as e:
                    logger.error(f"Error procesando línea {i}: {str(e)}")
        
        os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
        with open(ruta_salida, 'w', encoding='utf-8') as out_file:
            json.dump(productos, out_file, indent=2, ensure_ascii=False)
            
        logger.info(f"Conversión completada: {len(productos)} productos -> {os.path.basename(ruta_salida)}")        
        return productos
        
    except Exception as e:
        logger.error(f"Error en convertir_jsonl_a_json: {str(e)}")
        raise

def producto_a_documento(producto: Dict) -> Document:
    """Transforma un producto en un Documento de LangChain"""
    try:
        detalles = producto.get("details", {})
        texto = "\n".join([
            f"Título: {producto.get('title', '')}",
            f"Categoría: {producto.get('main_category', '')}",
            f"Precio: {producto.get('price', 'No disponible')}",
            f"Valoración: {producto.get('average_rating', '')}",
            "Detalles: " + ", ".join([f"{k}: {v}" for k, v in detalles.items()])
        ])
        return Document(
            page_content=texto,
            metadata={
                "title": producto.get("title", "Sin título"),
                "price": producto.get("price"),
                "category": producto.get("main_category")
            }
        )
    except Exception as e:
        logger.error(f"Error creando documento para producto: {str(e)}")
        raise

def procesar_archivo(archivo: str, input_dir: str, output_dir: str, index_dir: str) -> int:
    """
    Procesa un archivo individual y crea su índice Chroma
    
    Args:
        archivo: Nombre del archivo a procesar
        input_dir: Directorio de entrada
        output_dir: Directorio para JSON procesados
        index_dir: Directorio para índices Chroma
    
    Returns:
        Número de documentos indexados
    """
    try:
        nombre_base = os.path.splitext(archivo)[0]
        ruta_entrada = os.path.join(input_dir, archivo)
        ruta_salida = os.path.join(output_dir, f"{nombre_base}.json")
        persist_directory = os.path.join(index_dir, f"chroma_{nombre_base}")

        # Paso 1: Cargar datos
        if archivo.endswith('.jsonl'):
            productos = convertir_jsonl_a_json(ruta_entrada, ruta_salida)
        elif archivo.endswith('.json'):
            with open(ruta_entrada, 'r', encoding='utf-8') as f:
                productos = json.load(f)
        else:
            raise ValueError(f"Formato no soportado: {archivo}")

        # Paso 2: Crear documentos
        documentos = []
        for prod in productos:
            try:
                documentos.append(producto_a_documento(prod))
            except Exception as e:
                logger.warning(f"Producto omitido por error: {str(e)}")
                continue

        # Paso 3: Indexar
        vectordb = Chroma.from_documents(
            documents=documentos,
            embedding=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            ),
            persist_directory=persist_directory
        )
        vectordb.persist()
        
        logger.info(f"Índice creado: {len(documentos)} documentos en {persist_directory}")
        return len(documentos)
        
    except Exception as e:
        logger.error(f"Error procesando archivo {archivo}: {str(e)}")
        raise

def run_generator(data_dir: str = None) -> int:
    """
    Función principal para procesar todos los archivos en el directorio raw
    
    Args:
        data_dir: Ruta base del directorio de datos (opcional)
    
    Returns:
        Total de documentos procesados
    """
    try:
        # Configuración de rutas
        if not data_dir:
            data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        
        raw_dir = os.path.join(data_dir, "raw")
        processed_dir = os.path.join(data_dir, "processed")
        index_dir = os.path.join(data_dir, "chroma_indexes")

        # Crear directorios si no existen
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(index_dir, exist_ok=True)

        logger.info("Iniciando procesamiento de archivos...")
        total_docs = 0

        # Procesar cada archivo
        for archivo in os.listdir(raw_dir):
            if archivo.endswith(('.json', '.jsonl')):
                try:
                    docs_procesados = procesar_archivo(archivo, raw_dir, processed_dir, index_dir)
                    total_docs += docs_procesados
                    logger.info(f"Archivo {archivo} procesado: {docs_procesados} documentos")
                except Exception as e:
                    logger.error(f"Error procesando {archivo}: {str(e)}")
                    continue

        logger.info(f"Procesamiento completado. Total documentos indexados: {total_docs}")
        return total_docs
        
    except Exception as e:
        logger.error(f"Error en run_generator: {str(e)}")
        raise

if __name__ == "__main__":
    # Solo para pruebas directas
    try:
        total = run_generator()
        print(f"Procesamiento completado. Documentos indexados: {total}")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)