#!/usr/bin/env python3
# main.py - Amazon Recommendation System Entry Point

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from src.core.rag.basic.retriever import Retriever
from src.core.data.loader import DataLoader
from src.core.rag.advanced.agent import RAGAgent
from src.core.category_search.category_tree import CategoryTree
from src.interfaces.cli import main as cli_main  # Import the main function from cli.py
from src.core.utils.logger import configure_root_logger
from src.core.config import settings  # Ensure settings.py is correctly defined

# Load .env configuration
load_dotenv()
configure_root_logger(level=logging.DEBUG)

def initialize_system(data_dir: Optional[str] = None,
                     log_level: Optional[str] = None,
                     enable_ui: bool = False):
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
    retriever = initialize_retriever()
    logger.info("📚 Retriever ready")

    # Build advanced RAG agent
    rag_agent = RAGAgent(
        products=products,
        lora_checkpoint=settings.RLHF_CHECKPOINT,
        enable_translation=True
    )

    logger.info("🧠 RAG agent initialized")

    # Launch interface
    if enable_ui:
        from src.interfaces.ui import launch_ui
        launch_ui(products, category_tree, rag_agent)
    else:
        cli_main()  # Call the main function from cli.py

def initialize_retriever():
    """Instantiate and return retriever engine for RAG."""
    return Retriever(
        index_path=settings.VECTOR_INDEX_PATH,  # Use from settings
        embedding_model=settings.EMBEDDING_MODEL,
        vectorstore_type=settings.VECTOR_BACKEND,
        device=settings.DEVICE
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description="🔎 Amazon Product Recommendation System")
    
    # Main subcommands
    subparsers = parser.add_subparsers(dest='command')
    
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
    
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    return args

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.command == "index":
        # Load products without instantiating RAGAgent
        data_path = Path(args.data_dir or os.getenv("DATA_DIR") or "./data/raw")
        loader = DataLoader(
            raw_dir=data_path,
            processed_dir=settings.PROC_DIR,
            cache_enabled=settings.CACHE_ENABLED,
            disable_tqdm=getattr(args, 'disable_tqdm', False)
        )
        
        # Clear cache if requested
        if args.reindex:
            loader.clear_cache()
        
        products = loader.load_data(use_cache=not args.reindex)
        print(f"✅ Loaded {len(products)} products")

        # Initialize retriever
        index_path = Path(settings.VECTOR_INDEX_PATH).resolve()
        print(f"🛠️ Building vector index at {index_path}...")
        
        retriever = initialize_retriever()
        
        # Clear existing index if requested
        if args.reindex and retriever.index_exists():
            try:
                # Close any open connections to Chroma
                if isinstance(retriever.store, Chroma):
                    retriever.store._client = None
                
                # Wait a moment to ensure resources are released
                import time
                time.sleep(1)
                
                # Remove the index directory
                import shutil
                shutil.rmtree(index_path)
                index_path.mkdir(parents=True, exist_ok=True)
                print("♻️  Cleared existing index")
            except Exception as e:
                print(f"⚠️  Error clearing index: {e}")

        # Build new index
        retriever.build_index(products)
        print(f"✅ Success! Index contains {len(products)} product embeddings")
        
    elif args.command in {"rag", "category"}:
        initialize_system(
            data_dir=args.data_dir,
            log_level=args.log_level,
            enable_ui=getattr(args, 'ui', False)
        )