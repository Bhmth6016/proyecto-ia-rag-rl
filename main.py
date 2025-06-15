#!/usr/bin/env python3
import logging
from pathlib import Path
from typing import Optional
from src.data_loader import DataLoader
from src.category_selector.category_tree import generar_categorias_y_filtros
from src.ui_client.ui_interface import run_interface

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
        if not productos:
            logging.error("No se encontraron datos válidos.")
            return None
        logging.info(f"Datos cargados correctamente. Total productos: {len(productos)}")
        return productos
    except Exception as e:
        logging.critical(f"Error inicializando DataLoader: {str(e)}", exc_info=True)
        return None

def main():
    print("\n" + "="*50)
    print(" SISTEMA DE RECOMENDACIÓN ".center(50))
    print("="*50)

    configure_logging()

    if not (productos := initialize_data_loader()):
        return

    try:
        logging.info("Verificando categorías y filtros...")
        generar_categorias_y_filtros(productos)
        run_interface(productos)
    except KeyboardInterrupt:
        logging.info("Aplicación interrumpida por el usuario")
    except Exception as e:
        logging.error(f"Error crítico: {str(e)}", exc_info=True)
    finally:
        logging.info("Liberando recursos...")
        print("\n" + "="*50)
        print(" SESIÓN TERMINADA ".center(50))
        print("="*50)

if __name__ == "__main__":
    main()