#src/__init__.py
# Exporta las interfaces principales para el paquete raíz
from .interfaces import AmazonRecommendationCLI, AmazonProductUI
from .core import DataLoader, RAGAdvancedAgent

__all__ = [
    'AmazonRecommendationCLI',
    'AmazonProductUI',
    'DataLoader',
    'RAGAdvancedAgent'
]