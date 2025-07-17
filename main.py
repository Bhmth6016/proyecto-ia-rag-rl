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

# Load .env configuration
load_dotenv()

def initialize_system(data_dir: Optional[str] = None,
                     log_level: Optional[str] = None):
    """
    Initialize system components with better default handling
    """
    # Setup logger
    configure_root_logger(
        level=log_level or os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "logs/system.log")
    )
    logger = logging.getLogger(__name__)

    # Configure data directory with multiple fallbacks
    data_path = Path(data_dir or os.getenv("DATA_DIR") or "./data/raw")
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Created data directory at {data_path}")

    if not any(data_path.glob("*.json")) and not any(data_path.glob("*.jsonl")):
        logger.error(f"No JSON data files found in {data_path}")
        raise FileNotFoundError(f"No product data found in {data_path}")

    # Configure Gemini API key
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    logger.info("✅ Gemini API configured")

    loader = DataLoader(
        raw_dir=data_path,
        processed_dir=settings.PROC_DIR,
        cache_enabled=settings.CACHE_ENABLED
    )

    max_products = int(os.getenv("MAX_PRODUCTS_TO_LOAD", "10000"))
    products = loader.load_data()[:max_products]
    logger.info(f"📦 Loaded {len(products)} products")

    # Build category tree
    category_tree = CategoryTree(products)
    category_tree.build_tree()
    logger.info("🌲 Category tree built")

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Modelo más estable
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
        top_k=40,
        top_p=0.95
    )

    # Initialize memory for RLHF interaction
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Setup retriever (RAG backbone)
    retriever = initialize_retriever(products)
    logger.info("📚 Retriever ready")

    # Build advanced RAG agent
    rag_agent = RAGAgent(
        products=products,
        enable_translation=True
    )

    logger.info("🧠 RAG agent initialized")

    return products, category_tree, rag_agent


def initialize_retriever(products: List[Product]):
    """Instantiate and return retriever engine for RAG."""
    from src.core.rag.basic.retriever import Retriever
    return Retriever(
        index_path=settings.VECTOR_INDEX_PATH,  # Use from settings
        embedding_model=settings.EMBEDDING_MODEL,
        device=settings.DEVICE
    )


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
                print(f"\n📂 {node.name} ({len(node.products)} items)")
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
                print(f"\n🛍️  {node.name} – {len(node.products)} products")
                for i, p in enumerate(node.products[:20], 1):
                    print(f"{i:2}. {p.title} – ${p.price}")
                input("\nPress Enter to go back…")
                node = node.parent or tree.root
    except KeyboardInterrupt:
        print("\nLeaving category mode.")


if __name__ == "__main__":
    args = parse_arguments()

    if args.command == "index":
        # Initialize system to load products
        products, _, _ = initialize_system(
            data_dir=args.data_dir,
            log_level=args.log_level
        )

        # Initialize and build retriever
        retriever = initialize_retriever(products)

        # Clear existing index if reindex flag is set
        if args.reindex and retriever.index_exists():
            import shutil
            shutil.rmtree(settings.VECTOR_INDEX_PATH)
            print("♻️  Cleared existing index")

        # Build the index
        print(f"🛠️ Building vector index at {settings.VECTOR_INDEX_PATH}...")
        retriever.build_index(products)
        print(f"✅ Success! Index contains {len(products)} product embeddings")

    elif args.command in {"rag", "category"}:
        products, category_tree, rag_agent = initialize_system(
            data_dir=args.data_dir,
            log_level=args.log_level
        )

        if args.command == "rag":
            cli_main()  # Call the main function from cli.py
        elif args.command == "category":
            # Run category mode
            _run_category_mode(products, start=None)