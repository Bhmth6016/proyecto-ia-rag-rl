# src/core/utils/delayed_init.py
"""
Utilidades para inicialización retardada y manejo de dependencias.
"""
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class DelayedInitializer:
    """Maneja inicializaciones con delays para evitar conflictos."""
    
    @staticmethod
    def ensure_directories(directories: list, delay_seconds: float = 0.5):
        """
        Asegura que los directorios existan con un delay entre creaciones.
        
        Args:
            directories: Lista de rutas de directorios a crear
            delay_seconds: Delay entre creación de directorios
        """
        for i, dir_path in enumerate(directories):
            path = Path(dir_path)
            if not path.exists():
                logger.info(f"📁 Creando directorio {i+1}/{len(directories)}: {path}")
                path.mkdir(parents=True, exist_ok=True)
                if i < len(directories) - 1:  # No delay en el último
                    time.sleep(delay_seconds)
    
    @staticmethod
    def wait_for_component(component_name: str, check_func, 
                          timeout_seconds: int = 30, interval_seconds: float = 0.5):
        """
        Espera a que un componente esté listo.
        
        Args:
            component_name: Nombre del componente
            check_func: Función que retorna True cuando el componente está listo
            timeout_seconds: Tiempo máximo de espera
            interval_seconds: Intervalo entre checks
        """
        logger.info(f"⏳ Esperando por {component_name}...")
        
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            if check_func():
                logger.info(f"✅ {component_name} está listo")
                return True
            time.sleep(interval_seconds)
        
        logger.warning(f"⚠️ Timeout esperando por {component_name}")
        return False

# Función de conveniencia
def setup_system_directories():
    """Configura todos los directorios del sistema en orden correcto."""
    directories = [
        "data/raw",
        "data/processed",
        "data/processed/historial",  # 🔥 CRÍTICO: Este va ANTES de feedback
        "data/feedback",
        "data/users",
        "data/models",
        "logs",
    ]
    
    DelayedInitializer.ensure_directories(directories, delay_seconds=0.3)
    logger.info("✅ Todos los directorios del sistema creados")