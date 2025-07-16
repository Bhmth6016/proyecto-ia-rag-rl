#!/usr/bin/env python3
# main.py - Amazon Recommendation System Entry Point

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.data.loader import DataLoader
from src.core.rag.advanced.agent import RAGAgent
from src.core.category_search.category_tree import CategoryTree
from src.interfaces.cli import main as cli_main  # Import the main function from cli.py
from src.core.utils.logger import configure_root_logger
from src.core.config import settings  # Ensure settings.py is correctly defined

# Load .env configuration
load_dotenv()
configure_root_logger
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
    retriever = initialize_retriever(products)
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

def initialize_retriever(products):
    """Instantiate and return retriever engine for RAG."""
    from src.core.rag.basic.retriever import Retriever
    return Retriever(
        index_path=settings.VECTOR_INDEX_PATH,  # Use from settings
        embedding_model=settings.EMBEDDING_MODEL,
        vectorstore_type=settings.VECTOR_BACKEND,
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

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.command == "index":
        # Cargar productos sin instanciar RAGAgent
        data_path = Path(args.data_dir or os.getenv("DATA_DIR") or "./data/raw")
        loader = DataLoader(
            raw_dir=data_path,
            processed_dir=settings.PROC_DIR,
            cache_enabled=settings.CACHE_ENABLED
        )
        products = loader.load_data(use_cache=True)

        # Inicializar retriever
        retriever = initialize_retriever(products)
        
        # Limpiar índice si es necesario
        if args.reindex and retriever.index_exists():
            import os
            import shutil
            import time
            
            index_path = Path(settings.VECTOR_INDEX_PATH)
            
            # Intentar cerrar cualquier conexión existente a Chroma
            if hasattr(retriever, 'store'):
                if isinstance(retriever.store, Chroma):
                    try:
                        retriever.store._client = None
                    except:
                        pass
            
            # Esperar un momento para que se liberen los recursos
            time.sleep(1)
            
            # Eliminar contenido del directorio de forma segura
            try:
                for filename in os.listdir(index_path):
                    file_path = os.path.join(index_path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")
                        # Intento adicional para archivos bloqueados
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        except:
                            pass
                
                print("♻️  Cleared existing index")
            except Exception as e:
                print(f"Error clearing index: {e}")
                # Si falla, intentar crear el directorio de nuevo
                try:
                    shutil.rmtree(index_path)
                    os.makedirs(index_path, exist_ok=True)
                except:
                    pass

        # Crear índice
        print(f"🛠️ Building vector index at {settings.VECTOR_INDEX_PATH}...")
        retriever.build_index(products)
        print(f"✅ Success! Index contains {len(products)} product embeddings")
        
    elif args.command in {"rag", "category"}:
        initialize_system(
            data_dir=args.data_dir,
            log_level=args.log_level,
            enable_ui=getattr(args, 'ui', False)
        )