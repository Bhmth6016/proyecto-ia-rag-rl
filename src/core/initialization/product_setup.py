# src/core/initialization/product_setup.py
"""
Configuración inicial de ProductReference para evitar importaciones circulares.
"""
import logging
from typing import Type

from src.core.data.product_reference import ProductClassHolder
from src.core.data.product import Product as ProductClass

logger = logging.getLogger(__name__)

def setup_product_reference():
    """
    Configura la clase Product en ProductReference.
    Debe llamarse al inicio de la aplicación.
    """
    try:
        ProductClassHolder.set_product_class(ProductClass)
        logger.info("✅ ProductReference configurado correctamente con Product class")
        return True
    except Exception as e:
        logger.error(f"❌ Error configurando ProductReference: {e}")
        return False


def check_product_reference_setup() -> bool:
    """Verifica si ProductReference está correctamente configurado."""
    try:
        # Intentar usar ProductClassHolder para verificar
        ProductClassHolder.get_product_class()
        logger.info("✅ ProductReference está configurado y listo para usar")
        return True
    except RuntimeError as e:
        logger.warning(f"⚠️ ProductReference no configurado: {e}")
        return False


# También puedes proporcionar un decorador para métodos que necesitan Product
def requires_product_class(func):
    """Decorador que verifica que Product class esté disponible."""
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not ProductClassHolder.is_available():
            # Intentar importar automáticamente
            try:
                ProductClassHolder.get_product_class()
            except RuntimeError:
                raise RuntimeError(
                    f"La función {func.__name__} requiere que ProductReference esté configurado. "
                    "Llama a setup_product_reference() al inicio de tu aplicación."
                )
        return func(*args, **kwargs)
    
    return wrapper