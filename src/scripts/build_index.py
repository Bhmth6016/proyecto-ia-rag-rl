# build_index.py
import logging
from pathlib import Path
from src.core.data.loader import DataLoader
from src.core.rag.basic.retriever import Retriever

# Configurar el logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definir rutas de directorios
RAW_DIR = Path("path/to/raw_dir")
PROC_DIR = Path("path/to/processed_dir")

def main():
    # Inicializar DataLoader
    loader = DataLoader(RAW_DIR, PROC_DIR)
    unified_file = PROC_DIR / "products.json"
    
    # Crear archivo unificado si no existe
    if not unified_file.exists():
        logger.info("Creando archivo unificado de productos...")
        loader.create_unified_json(unified_file)
        logger.info("Archivo unificado creado con éxito.")
    else:
        logger.info("Archivo unificado ya existe. No es necesario recrearlo.")
    
    # Inicializar Retriever y construir índice
    retriever = Retriever()
    logger.info("Construyendo índice de productos...")
    retriever.build_index(unified_file)
    logger.info("Índice construido con éxito.")

if __name__ == "__main__":
    main()