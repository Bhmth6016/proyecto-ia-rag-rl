#!/usr/bin/env python3
# main.py - Amazon Recommendation System Entry Point

import argparse
import logging
import os
import sys
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
from src.core.utils.parsers import parse_binary_score
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

        # Inicializar retriever con los productos cargados
        retriever = Retriever(
            index_path=settings.VECTOR_INDEX_PATH,
            embedding_model=settings.EMBEDDING_MODEL,
            device=settings.DEVICE,
        )
        
        # Construir índice si no existe
        logger.info("Building vector index...")
        retriever.build_index(products)

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
    parser = argparse.ArgumentParser(
        description="🔎 Amazon Product Recommendation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Argumentos comunes para todos los comandos
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--data-dir", type=str, 
                             help="Custom data directory path",
                             default=None)
    common_parser.add_argument("--log-file", type=Path, 
                             help="Path to log file",
                             default=None)
    common_parser.add_argument("--log-level", 
                             choices=['DEBUG','INFO','WARNING','ERROR'], 
                             help="Logging level",
                             default='INFO')
    common_parser.add_argument("-v", "--verbose", 
                             action="store_true",
                             help="Enable verbose logging")

    # Subcomandos
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Comando index
    index_parser = subparsers.add_parser(
        'index', 
        help='(Re)build vector index',
        parents=[common_parser]
    )
    index_parser.add_argument('--clear-cache', 
                            action='store_true',
                            help='Clear product cache before rebuilding')
    index_parser.add_argument('--force', 
                            action='store_true',
                            help='Force reindexing even if index exists')
    index_parser.add_argument('--batch-size', 
                            type=int, 
                            default=4000,
                            help='Number of products per batch')

    # Comando rag
    rag_parser = subparsers.add_parser(
        'rag', 
        help='RAG recommendation mode',
        parents=[common_parser]
    )
    rag_parser.add_argument('--ui', 
                          action='store_true',
                          help='Enable graphical interface')
    rag_parser.add_argument('-k', '--top-k', 
                          type=int, 
                          default=5,
                          help='Number of results to return')

    # Comando category
    category_parser = subparsers.add_parser(
        'category', 
        help='Category search mode',
        parents=[common_parser]
    )
    category_parser.add_argument('-c', '--category', 
                               type=str,
                               help='Starting category')

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
    logger.info("Loading products...")
    products = system.products  # This will trigger product loading
    logger.info(f"Loaded {len(products)} products")
    
    # Initialize retriever with build_if_missing=False to prevent automatic building
    system._retriever = Retriever(
        index_path=settings.VECTOR_INDEX_PATH,
        embedding_model=settings.EMBEDDING_MODEL,
        device=settings.DEVICE,
    )
    
    # Calculate safe batch size
    try:
        import psutil
        available_mem = psutil.virtual_memory().available / (1024 ** 3)  # GB
        # Estimación más conservadora: ~2MB por documento
        safe_batch_size = min(4000, int(available_mem * 500))  # Máximo 4000 por lote
        logger.info(f"Available memory: {available_mem:.2f}GB, using safe batch size: {safe_batch_size}")
    except:
        safe_batch_size = 2000  # Valor por defecto más conservador
        logger.info(f"Could not detect memory, using safe default batch size: {safe_batch_size}")
    
    # Now build the index with progress monitoring
    logger.info("Starting index build process...")
    try:
        system.retriever.build_index(products, batch_size=safe_batch_size)
        logger.info("Index built successfully!")
    except Exception as e:
        logger.error(f"Failed to build index: {str(e)}")
        return


def _handle_rag_mode(system, args):
    """Handle RAG interaction with automatic index creation"""
    print("🛠️ Preparing RAG system...")
    
    # Load products
    products = system.products
    if not products:
        raise RuntimeError("No products loaded")
    
    # Create index automatically
    print("🔨 Building vector index...")
    system.retriever.build_index(products)
    
    # Initialize RAG agent
    agent = RAGAgent(
        products=products,
        enable_translation=True
    )
    
    print("\n=== Amazon RAG ===\nType 'exit' to quit\n")
    while True:
        try:
            query = input("🧑 You: ").strip()
            if query.lower() in {"exit", "quit", "q"}:
                break

            answer = agent.ask(query)
            print(f"\n🤖 {answer}\n")

            if not getattr(args, 'no_feedback', False):
                rating = input("Helpful? (y/n): ").strip().lower()
                score = parse_binary_score(rating)
                logging.info(f"Feedback|{query}|{score.name}")

        except KeyboardInterrupt:
            print("\n🛑 Session ended")
            break


if __name__ == "__main__":
    args = parse_arguments()

    # Configure logging with safe attribute handling
    log_level = getattr(args, 'log_level', 'INFO')
    if getattr(args, 'verbose', False):
        log_level = 'DEBUG'
    
    configure_root_logger(
        level=log_level,
        log_file=getattr(args, 'log_file', None),
        module_levels={"urllib3": logging.WARNING}
    )

    try:
        system = get_system()
        
        if args.command == "index":
            _run_index_mode()
        elif args.command == "rag":
            _handle_rag_mode(system, args)
        elif args.command == "category":
            _run_category_mode(system.products, getattr(args, 'category', None))
            
    except Exception as e:
        logging.error(f"System failed: {str(e)}")
        sys.exit(1)