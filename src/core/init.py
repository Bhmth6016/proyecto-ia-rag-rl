# src/core/init.py - SIMPLIFICADO PARA EVITAR CIRCULARIDAD
"""
Inicialización simple del sistema.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SimpleInitializer:
    """Inicializador simple que evita dependencias circulares."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self._initialized = False
        if not self._initialized:
            self._initialized = True
            logger.info("🚀 SimpleInitializer creado")
    
    def get_config(self):
        """Obtiene configuración de forma diferida."""
        try:
            from .config import get_settings
            return get_settings()
        except ImportError as e:
            logger.warning(f"No se pudo cargar configuración: {e}")
            return None

def get_system():
    """Punto de acceso simple al sistema."""
    return SimpleInitializer()