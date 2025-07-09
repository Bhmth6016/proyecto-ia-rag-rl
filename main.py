#!/usr/bin/env python3
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Configuración de paths
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_INDEX_DIR = DATA_DIR / "chroma_indexes"

def configure_logging():
    """Configuración centralizada del logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler("recommendation_system.log"),
            logging.StreamHandler()
        ]
    )
    # Configurar niveles para dependencias
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

def initialize_data_loader() -> Optional[List[Dict]]:
    """Carga y valida los datos de productos"""
    try:
        from src.data_loader import DataLoader
        loader = DataLoader()
        productos = loader.load_data(use_cache=True)
        
        if not productos or not isinstance(productos, list):
            logging.error("Datos no válidos o vacíos")
            return None
            
        logging.info(f"Datos cargados. Productos totales: {len(productos)}")
        
        # Validación básica de estructura
        valid_products = [p for p in productos if isinstance(p, dict) and p.get('title')]
        invalid_count = len(productos) - len(valid_products)
        
        if invalid_count > 0:
            logging.warning(f"{invalid_count} productos con estructura inválida")
        
        return valid_products
        
    except Exception as e:
        logging.critical(f"Error en DataLoader: {str(e)}", exc_info=True)
        return None

def run_data_generator():
    """Ejecuta el generador de datos e índices"""
    try:
        from demo.generator import run_generator
        print("\n  Iniciando preprocesamiento de datos...")
        
        # Crear directorios si no existen
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(CHROMA_INDEX_DIR, exist_ok=True)
        
        total_docs = run_generator(str(DATA_DIR))
        print(f"\n Preprocesamiento completado. {total_docs} documentos indexados.")
        
    except Exception as e:
        logging.error(f"Error en el generador de datos: {str(e)}")
        print(f" Error durante el preprocesamiento: {str(e)}")

def run_rag_advanced():
    """Ejecuta el agente RAG avanzado"""
    try:
        from demo.rag_agent_v2 import run_rag_agent_v2_interactive
        print("\n Iniciando Agente RAG Avanzado...")
        run_rag_agent_v2_interactive(str(CHROMA_INDEX_DIR))
    except Exception as e:
        logging.error(f"Error en el agente RAG avanzado: {str(e)}")
        print(f" Error iniciando el agente RAG: {str(e)}")

def run_category_mode(productos: List[Dict]):
    """Ejecuta el modo de categorías con validación"""
    try:
        if not productos:
            logging.error("No hay productos válidos para categorizar")
            return
            
        logging.info("Generando categorías y filtros...")
        from src.category_selector.category_tree import generar_categorias_y_filtros
        category_filters = generar_categorias_y_filtros(productos)
        
        if category_filters is None:
            logging.error("No se pudieron generar las categorías y filtros")
            return
            
        # Validar estructura del archivo de filtros
        filters_file = PROCESSED_DATA_DIR / "category_filters.json"
        if not filters_file.exists():
            logging.error("Archivo de filtros no encontrado")
            return
            
        with open(filters_file, 'r', encoding='utf-8') as f:
            filters_data = json.load(f)
            
        if not all(k in filters_data for k in ['global', 'by_category']):
            logging.error("Estructura de filtros inválida")
            return
            
        from src.ui_client.ui_interface import run_interface
        run_interface(productos, filters_data)
        
    except json.JSONDecodeError:
        logging.error("Error leyendo el archivo de filtros")
    except KeyboardInterrupt:
        logging.info("Modo categorías interrumpido")
    except Exception as e:
        logging.error(f"Error en modo categorías: {str(e)}", exc_info=True)
    finally:
        logging.info("Finalizando modo clásico...")

def show_welcome_banner():
    """Muestra el banner de bienvenida"""
    print("\n" + "="*50)
    print(" SISTEMA DE RECOMENDACIÓN AVANZADO ".center(50))
    print("="*50)
    print(f"\n Directorio de datos: {DATA_DIR}")
    print(f" Índices vectoriales: {CHROMA_INDEX_DIR}\n")

def main():
    """Función principal con flujo mejorado"""
    configure_logging()
    show_welcome_banner()
    
    # Verificar estructura de directorios
    if not RAW_DATA_DIR.exists():
        print(f" Advertencia: Directorio 'raw' no encontrado en {RAW_DATA_DIR}")
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        print(f" Se creó el directorio 'raw'. Coloca allí tus archivos JSON/JSONL.")

    productos = initialize_data_loader()
    
    while True:
        print("\nMenú Principal:")
        print("1. Preprocesar datos (generar índices)")
        print("2. Buscar por texto (RAG Avanzado)")
        print("3. Explorar por categorías y filtros")
        print("4. Salir")
        
        try:
            opcion = input("\nSeleccione una opción (1-4): ").strip()
            
            if opcion == "1":
                run_data_generator()
            elif opcion == "2":
                if not CHROMA_INDEX_DIR.exists() or not any(CHROMA_INDEX_DIR.iterdir()):
                    print("\n No se encontraron índices vectoriales. Ejecuta primero la opción 1.")
                    continue
                run_rag_advanced()
            elif opcion == "3":
                if not productos:
                    print("\n No se pudieron cargar los productos. Verifica los logs.")
                    continue
                run_category_mode(productos)
            elif opcion == "4":
                print("\n¡Hasta luego!")
                break
            else:
                print("\nOpción no válida. Intenta de nuevo.")
                
        except KeyboardInterrupt:
            print("\nOperación cancelada por el usuario")
            break
        except Exception as e:
            logging.error(f"Error inesperado: {str(e)}")
            print("\nOcurrió un error. Por favor intenta de nuevo.")

    print("\n" + "="*50)
    print(" SESIÓN TERMINADA ".center(50))
    print("="*50)

if __name__ == "__main__":
    main()