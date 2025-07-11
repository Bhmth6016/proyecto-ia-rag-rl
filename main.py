#!/usr/bin/env python3
# main.py - Punto de entrada principal del sistema de recomendación de Amazon

import argparse
import logging
from pathlib import Path
from typing import Optional

from src.core.data.loader import DataLoader
from src.core.rag.advanced import RAGAdvancedAgent
from src.core.category_search.category_tree import CategoryTree
from src.interfaces.cli import AmazonRecommendationCLI
from src.interfaces.ui import AmazonProductUI
from src.core.utils.logger import configure_root_logger

def initialize_system(data_dir: Optional[str] = None, 
                     log_level: str = "INFO",
                     enable_ui: bool = False):
    """
    Inicializa todos los componentes del sistema
    
    Args:
        data_dir: Directorio con los datos (opcional)
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
        enable_ui: Si se habilita la interfaz gráfica
    """
    # Configuración inicial
    configure_root_logger(level=log_level)
    logger = logging.getLogger(__name__)
    logger.info("Inicializando sistema de recomendación de Amazon")
    
    # Carga de datos
    loader = DataLoader(data_dir)
    products = loader.load_data()
    logger.info(f"Cargados {len(products)} productos")
    
    # Inicialización de componentes principales
    category_tree = CategoryTree(products)
    category_tree.build_tree()
    
    # Configuración del agente RAG
    rag_agent = RAGAdvancedAgent(
        llm=initialize_llm(),
        retriever=initialize_retriever(products),
        memory=initialize_memory()
    )
    
    # Selección de interfaz
    if enable_ui:
        from src.interfaces import AmazonProductUI
        AmazonProductUI(products, category_tree, rag_agent)
    else:
        cli = AmazonRecommendationCLI(products, category_tree, rag_agent)
        cli.run()

def initialize_llm():
    """Configura el modelo de lenguaje"""
    from langchain_community.llms import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    
    return OpenAI(
        model_name="gpt-3.5-turbo-instruct",
        temperature=0.3,
        max_tokens=500
    )

def initialize_retriever(products):
    """Configura el sistema de recuperación"""
    from src.core.rag.basic.retriever import VectorRetriever
    return VectorRetriever(products)

def initialize_memory():
    """Configura la memoria para conversaciones"""
    from langchain.memory import ConversationBufferMemory
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

def parse_arguments():
    """Configura el parser de argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Sistema de Recomendación de Productos Amazon"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directorio con los datos de productos"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Nivel de logging"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Usar interfaz gráfica en lugar de CLI"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Reindexar los datos antes de iniciar"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.reindex:
        from demo.generator import run_generator
        run_generator(args.data_dir)
    
    initialize_system(
        data_dir=args.data_dir,
        log_level=args.log_level,
        enable_ui=args.ui
    )