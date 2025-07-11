#!/usr/bin/env python3
# main.py - Amazon Recommendation System Entry Point

import argparse
import logging
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from google.api_core.exceptions import InvalidArgument
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.data.loader import DataLoader
from src.core.rag.advanced.agent import AdvancedRAGAgent
from src.core.category_search.category_tree import CategoryTree
from src.interfaces.cli import AmazonRecommendationCLI
from src.core.utils.logger import configure_root_logger

# Load environment variables
load_dotenv()

def initialize_system(data_dir: Optional[str] = None, 
                    log_level: Optional[str] = None,
                    enable_ui: bool = False):
    """
    Initialize all system components
    
    Args:
        data_dir: Directory with product data (optional)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_ui: Whether to enable graphical interface
    """
    # Configure logging
    configure_root_logger(
        level=log_level or os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE")
    )
    logger = logging.getLogger(__name__)
    logger.info("Initializing Amazon Recommendation System")
    
    # Configure Gemini - MODIFIED VERSION
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            convert_system_message_to_human=True,
            google_api_key=os.getenv("GEMINI_API_KEY")  # Explicitly pass the key
        )
    except InvalidArgument as e:
        logger.error(f"Failed to configure Gemini: {str(e)}")
        raise
    
    # Load data
    loader = DataLoader(data_dir or os.getenv("DATA_DIR"))
    max_products = int(os.getenv("MAX_PRODUCTS_TO_LOAD", 10000))
    products = loader.load_data()[:max_products]
    logger.info(f"Loaded {len(products)} products")
    
    # Initialize category tree
    category_tree = CategoryTree(products)
    category_tree.build_tree()
    
    # Configure LLM and memory
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    memory = ConversationBufferMemory()
    
    # Configure RAG agent
    rag_agent = AdvancedRAGAgent(
        llm=llm,
        retriever=initialize_retriever(products),
        memory=memory,
        enable_rewrite=True,
        enable_validation=True
    )
    
    # Select interface
    if enable_ui:
        from src.interfaces.ui import launch_ui
        launch_ui(products, category_tree, rag_agent)
    else:
        cli = AmazonRecommendationCLI(products, category_tree, rag_agent)
        cli.run()

def initialize_retriever(products):
    """Configure the retrieval system"""
    from src.core.rag.basic.retriever import Retriever
    return Retriever(
        index_path=os.getenv("VECTOR_INDEX_PATH", "./data/vector_index"),
        embedding_model=os.getenv("EMBEDDING_MODEL")
    )

def parse_arguments():
    """Configure command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Amazon Product Recommendation System"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Directory with product data"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Use graphical interface instead of CLI"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Reindex data before starting"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.reindex:
        from demo.generator import run_generator
        run_generator(args.data_dir or os.getenv("DATA_DIR"))
    
    initialize_system(
        data_dir=args.data_dir,
        log_level=args.log_level,
        enable_ui=args.ui
    )