#src/core/__init__.py
# Exporta las principales funcionalidades del core
from .rag import Retriever, Indexer, AdvancedRAGAgent
from .category_search import CategoryTree
from .data import DataLoader, Product, ProductDetails
from .utils import get_logger

__all__ = [
    'Retriever',
    'Indexer',
    'AdvancedRAGAgent',
    'CategoryTree',
    'DataLoader',
    'Product',          # Exportado desde data
    'ProductDetails',   # Exportado desde data
    'get_logger'
]