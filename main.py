#!/usr/bin/env python3
# main.py - Amazon Recommendation System Entry Point

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.data.loader import DataLoader
from src.core.rag.advanced.agent import AdvancedRAGAgent
from src.core.category_search.category_tree import CategoryTree
from src.interfaces.cli import AmazonRecommendationCLI
from src.core.utils.logger import configure_root_logger

# Load .env configuration
load_dotenv()

def initialize_system(data_dir: Optional[str] = None,
                     log_level: Optional[str] = None,
                     enable_ui: bool = False):
    """
    Initialize system components: logger, data loader, LLM, retriever, agent, and interface.
    """
    # Setup logger
    configure_root_logger(
        level=log_level or os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "logs/system.log")
    )
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Amazon Product Recommendation System")

    # Configure Gemini API key
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    logger.info("✅ Gemini API configured")

    # Load product data
    data_path = data_dir or os.getenv("DATA_DIR")
    if not data_path or not Path(data_path).exists():
        logger.error(f"❌ Data path not found: {data_path}")
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    loader = DataLoader(data_path)
    max_products = int(os.getenv("MAX_PRODUCTS_TO_LOAD", "10000"))
    products = loader.load_data()[:max_products]
    logger.info(f"📦 Loaded {len(products)} products")

    # Build category tree
    category_tree = CategoryTree(products)
    category_tree.build_tree()
    logger.info("🌲 Category tree built")

    # Initialize Gemini LLM
    # En main.py, modifica:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Modelo más estable
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
        top_k=40,
        top_p=0.95
    )

    # Initialize memory for RLHF interaction
    memory = ConversationBufferMemory(return_messages=True)
    
    # Setup retriever (RAG backbone)
    retriever = initialize_retriever(products)
    logger.info("📚 Retriever ready")

    # Build advanced RAG agent
    rag_agent = AdvancedRAGAgent(
        llm=llm,
        retriever=retriever,
        memory=memory,
        rate_limit=30  # 30 llamadas por minuto (dentro del límite gratuito)
    )
    logger.info("🧠 RAG agent initialized")

    # Launch interface
    if enable_ui:
        from src.interfaces.ui import launch_ui
        launch_ui(products, category_tree, rag_agent)
    else:
        cli = AmazonRecommendationCLI(
            products=products,
            category_tree=category_tree,
            llm=llm,
            memory=memory
        )
        cli.run()

def initialize_retriever(products):
    """Instantiate and return retriever engine for RAG."""
    from src.core.rag.basic.retriever import Retriever
    return Retriever(
        index_path=os.getenv("VECTOR_INDEX_PATH", "./data/vector_index"),
        embedding_model=os.getenv("EMBEDDING_MODEL")
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

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.command == "index":
        if args.reindex:
            from demo.generator import run_generator
            run_generator(args.data_dir or os.getenv("DATA_DIR"))
    elif args.command in {"rag", "category"}:
        initialize_system(
            data_dir=args.data_dir,
            log_level=args.log_level,
            enable_ui=getattr(args, 'ui', False)
        )