#!/usr/bin/env python3
# main.py - Amazon Recommendation System Entry Point

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.data.loader import DataLoader
from src.core.rag.advanced.agent import RAGAgent
from src.core.category_search.category_tree import CategoryTree
from src.interfaces.cli import main as cli_main  # Import the main function from cli.py
from src.core.utils.logger import configure_root_logger
from src.core.config import settings  # Ensure settings.py is correctly defined
from src.core.data.product import Product  # Import Product class
from src.core.init import get_system  # Import get_system function
from src.core.rag.basic.retriever import Retriever  # Import Retriever class

# Load .env configuration
load_dotenv()

# Setup logger at the module level
configure_root_logger(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", "logs/system.log")
)
logger = logging.getLogger(__name__)

def initialize_system(data_dir: Optional[str] = None, log_level: Optional[str] = None, include_rag_agent: bool = True):
    """Initialize system components with better error handling"""
    try:
        # Configurar ruta de datos
        data_path = Path(data_dir or os.getenv("DATA_DIR") or "./data/raw")
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Created data directory at {data_path}")

        if not any(data_path.glob("*.json")) and not any(data_path.glob("*.jsonl")):
            raise FileNotFoundError(f"No product data found in {data_path}")

        # Configurar clave de API de Gemini
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        logger.info("🔐 Gemini API key configured")

        # Cargar productos
        loader = DataLoader(
            raw_dir=data_path,
            processed_dir=settings.PROC_DIR,
            cache_enabled=settings.CACHE_ENABLED
        )
        max_products = int(os.getenv("MAX_PRODUCTS_TO_LOAD", "10000"))
        products = loader.load_data()[:max_products]
        if not products:
            raise RuntimeError("No products loaded from data directory")
        logger.info(f"📦 Loaded {len(products)} products")

        # Árbol de categorías
        category_tree = CategoryTree(products)
        category_tree.build_tree()
        logger.info("🗂️ Category tree built")

        # Inicializar sistema base (retriever)
        system = get_system()
        logger.info("🔍 Retriever ready")

        # Inicializar agente RAG con fallback
        rag_agent = None
        if include_rag_agent:
            try:
                rag_agent = RAGAgent(
                    products=products,
                    enable_translation=True
                )
                logger.info("🧠 RAG agent initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize RAG agent: {e}")
                logger.warning("⚠️ Running in limited mode without RAG functionality")
                rag_agent = None

        return products, category_tree, rag_agent

    except Exception as e:
        logger.critical(f"🔥 System initialization failed: {e}", exc_info=True)
        raise



def parse_arguments():
    parser = argparse.ArgumentParser(description="🔎 Amazon Product Recommendation System")

    # Main subcommands
    subparsers = parser.add_subparsers(dest='command', required=True)

    # RAG command
    rag_parser = subparsers.add_parser('rag', help='RAG recommendation mode')
    rag_parser.add_argument('--ui', action='store_true', help='Enable graphical interface')

    # Category command
    category_parser = subparsers.add_parser('category', help='Category search mode')

    # Index command
    index_parser = subparsers.add_parser('index', help='Reindex data')
    index_parser.add_argument('--reindex', action='store_true', help='Force reindexing')

    # Common arguments
    for p in [rag_parser, category_parser, index_parser]:
        p.add_argument('--data-dir', type=str, help='Custom data directory path')
        p.add_argument('--log-level', choices=['DEBUG','INFO','WARNING','ERROR'], help='Logging level')

    return parser.parse_args()


def _run_category_mode(products: List[Product], start: Optional[str]) -> None:
    """Interactive category explorer."""
    tree = CategoryTree(products)
    tree.build_tree()

    node = tree.root
    print("\n=== Category Explorer ===\nPress Ctrl+C twice to exit.\n")

    try:
        while True:
            if node.children:
                print(f"\n {node.name} ({len(node.products)} items)")
                for i, child in enumerate(node.children, 1):
                    print(f"  {i}. {child.name} ({len(child.products)} items)")
                print("  0. Back" if node.parent else "  0. Exit")

                choice = input("Select: ").strip()
                if choice == "0":
                    node = node.parent or tree.root
                    continue
                if choice.isdigit() and 1 <= int(choice) <= len(node.children):
                    node = node.children[int(choice) - 1]
                    continue
                print("Invalid choice.")
            else:
                print(f"\n  {node.name} – {len(node.products)} products")
                for i, p in enumerate(node.products[:20], 1):
                    print(f"{i:2}. {p.title} – ${p.price}")
                input("\nPress Enter to go back…")
                node = node.parent or tree.root
    except KeyboardInterrupt:
        print("\nLeaving category mode.")


def _run_index_mode():
    # Initialize system without trying to load the retriever first
    system = get_system()
    
    # Load products
    logger.info(" Loading products...")
    products = system.products  # This will trigger product loading
    logger.info(f" Loaded {len(products)} products")
    
    # Initialize retriever with build_if_missing=False to prevent automatic building
    system._retriever = Retriever(
        index_path=settings.VECTOR_INDEX_PATH,
        embedding_model=settings.EMBEDDING_MODEL,
        device=settings.DEVICE,
        build_if_missing=False
    )
    
    # Check if the index already exists
    if system.retriever.index_exists():
        overwrite = input("Index exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != "y":
            logger.info("Index building aborted by user.")
            return
    
    # Calculate safe batch size
    try:
        import psutil
        available_mem = psutil.virtual_memory().available / (1024 ** 3)  # GB
        # Estimación más conservadora: ~2MB por documento
        safe_batch_size = min(4000, int(available_mem * 500))  # Máximo 4000 por lote
        logger.info(f" Available memory: {available_mem:.2f}GB, using safe batch size: {safe_batch_size}")
    except:
        safe_batch_size = 2000  # Valor por defecto más conservador
        logger.info(f" Could not detect memory, using safe default batch size: {safe_batch_size}")
    
    # Now build the index with progress monitoring
    logger.info(" Starting index build process...")
    try:
        system.retriever.build_index(products, force_rebuild=True, batch_size=safe_batch_size)
        logger.info(" Index built successfully!")
    except Exception as e:
        logger.error(f" Failed to build index: {str(e)}")
        return

if __name__ == "__main__":
    args = parse_arguments()

    if args.command == "index":
        _run_index_mode()
    elif args.command in {"rag", "category"}:
        products, category_tree, rag_agent = initialize_system(
            data_dir=args.data_dir,
            log_level=args.log_level
        )

        if args.command == "rag":
            print("DEBUG - Inicializando sistema para modo RAG")  # <-- Agrega esto
            products, category_tree, rag_agent = initialize_system(
                data_dir=args.data_dir,
                log_level=args.log_level
            )
            print("DEBUG - Sistema inicializado, llamando a CLI")  # <-- Agrega esto
            cli_main() 
            from src.interfaces.cli import main as cli_main
            cli_main(['rag'])

