"""
Centralized system initialization with Singleton pattern.
"""
from pathlib import Path
from typing import List, Optional
from src.core.data.loader import DataLoader
from src.core.data.product import Product
from src.core.rag.basic.retriever import Retriever
from src.core.category_search.category_tree import CategoryTree
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

class SystemInitializer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._products = None
            self._retriever = None
            self._category_tree = None
            self._initialized = True

    @property
    def products(self) -> List[Product]:
        if self._products is None:
            self._load_products()
        return self._products

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._initialize_retriever()
        return self._retriever

    @property
    def category_tree(self) -> CategoryTree:
        if self._category_tree is None:
            self._build_category_tree()
        return self._category_tree
    
    @property
    def loader(self) -> DataLoader:
        if not hasattr(self, '_loader'):
            self._loader = DataLoader(
                raw_dir=settings.RAW_DIR,
                processed_dir=settings.PROC_DIR
            )
        return self._loader

    def _load_products(self) -> None:
        """Load products with caching."""
        loader = DataLoader(
            raw_dir=settings.RAW_DIR,
            processed_dir=settings.PROC_DIR,
            cache_enabled=settings.CACHE_ENABLED
        )
        self._products = loader.load_data()

    def _initialize_retriever(self) -> None:
        """Initialize retriever and build index if needed."""
        logger.info(f"Initializing retriever at {settings.VECTOR_INDEX_PATH}")
        
        # Crear directorio si no existe
        index_path = Path(settings.VECTOR_INDEX_PATH)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._retriever = Retriever(
            index_path=settings.VECTOR_INDEX_PATH,
            embedding_model=settings.EMBEDDING_MODEL,
            device=settings.DEVICE,
            build_if_missing=False  # No construir automáticamente aquí
        )
        
        try:
            if not self._retriever.index_exists():
                logger.warning("Index not found. Building...")
                if not hasattr(self, '_products') or not self._products:
                    self._load_products()  # Asegurar que los productos están cargados
                
                self._retriever.build_index(self.products)
                logger.info("Index built successfully")
            else:
                logger.info("Index loaded successfully")
                
                # Verificar que el índice contiene documentos
                try:
                    doc_count = len(self._retriever.store.get()['ids'])
                    logger.info(f"Index contains {doc_count} documents")
                except Exception as e:
                    logger.warning(f"Could not verify document count: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise

    def _build_category_tree(self) -> None:
        """Build category hierarchy."""
        self._category_tree = CategoryTree(self.products)
        self._category_tree.build_tree()

def get_system() -> SystemInitializer:
    """Global access point for initialized system."""
    return SystemInitializer()