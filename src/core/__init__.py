"""
Core module exports.
"""
from typing import List

# Define __all__ first
__all__: List[str] = []

# ProductReference should be available
try:
    from .data.product_reference import ProductReference
    __all__.append('ProductReference')
except ImportError:
    ProductReference = None  # Define as None if not available
    pass

# You might also want to export other core components
try:
    from .retriever import Retriever
    __all__.append('Retriever')
except ImportError:
    Retriever = None
    pass

try:
    from .working_rag_agent import WorkingRAGAgent
    __all__.append('WorkingRAGAgent')
except ImportError:
    WorkingRAGAgent = None
    pass

try:
    from .collaborative_filter import CollaborativeFilter
    __all__.append('CollaborativeFilter')
except ImportError:
    CollaborativeFilter = None
    pass

# Optionally, define what gets imported with "from src.core import *"
__all__ = list(set(__all__))  # Remove duplicates