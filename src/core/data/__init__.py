#src/core/data/__init__.py
# Exporta las principales funcionalidades del core
from .product import Product, ProductDetails
from .loader import DataLoader
from ..utils import get_logger
from ..category_search import CategoryTree

__all__ = [
    'Product',
    'ProductDetails',
    'DataLoader',
    'CategoryTree',
    'get_logger'
]
