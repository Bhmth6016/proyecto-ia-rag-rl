# src/core/__init__.py - MÍNIMO
"""
Package initialization for core modules.
"""

# Exportar configuración
from src.core.config import settings, get_settings

# Exportar componentes principales
from src.core.data.product import Product
from src.core.data.ml_processor import ProductDataPreprocessor
from src.core.data.product_reference import ProductReference

# Versión
__version__ = "1.0.0"
__author__ = "Amazon Recommendation System"

# Lista de exports
__all__ = [
    "settings",
    "get_settings",
    "Product",
    "ProductDataPreprocessor", 
    "ProductReference"
]