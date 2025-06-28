#version establecida: 1.1.0 
# #!/usr/bin/env python3
import logging
from typing import Optional
from src.data_loader import DataLoader
from src.category_selector.category_tree import generar_categorias_y_filtros
from src.ui_client.ui_interface import run_interface
from pathlib import Path
import json

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler("recommendation_system.log"),
            logging.StreamHandler()
        ]
    )

def initialize_data_loader() -> Optional[list]:
    try:
        loader = DataLoader()
        productos = loader.load_data(use_cache=True)
        if not productos or not isinstance(productos, list):
            logging.error("Datos no válidos o vacíos")
            return None
        logging.info(f"Datos cargados. Productos: {len(productos)}")
        return productos
    except Exception as e:
        logging.critical(f"Error DataLoader: {str(e)}", exc_info=True)
        return None

def main():
    print("\n" + "="*50)
    print(" SISTEMA DE RECOMENDACIÓN ".center(50))
    print("="*50)

    configure_logging()

    productos = initialize_data_loader()
    if not productos:
        return

    try:
        logging.info("Generando categorías y filtros...")
        category_filters = generar_categorias_y_filtros(productos)
        if category_filters is None:
            logging.error("No se pudieron generar las categorías y filtros")
            return
        
        with open(Path("data/processed/category_filters.json"), 'r', encoding='utf-8') as f:
            filters_data = json.load(f)
        
        run_interface(productos, filters_data)  # Falta paréntesis de cierre aquí
    except KeyboardInterrupt:
        logging.info("Aplicación interrumpida")
    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
    finally:
        logging.info("Finalizando...")
        print("\n" + "="*50)
        print(" SESIÓN TERMINADA ".center(50))
        print("="*50)

if __name__ == "__main__":
    main()