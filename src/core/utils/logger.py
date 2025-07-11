# src/core/utils/logger.py
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Formateador personalizado para logs en JSON"""
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    json_format: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configura y retorna un logger con handlers para consola y archivo.
    
    Args:
        name: Nombre del logger (usualmente __name__)
        log_file: Ruta del archivo de log (opcional)
        level: Nivel de logging (default: INFO)
        json_format: Si True, usa formato JSON
        max_bytes: Tamaño máximo por archivo de log
        backup_count: Número de archivos de backup a mantener
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Evitar agregar handlers múltiples
    if logger.handlers:
        return logger
    
    # Formateador común
    formatter = (
        JSONFormatter() if json_format 
        else logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    )
    
    # Handler para consola (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler para archivo (si se especifica)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def configure_root_logger(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    modules_levels: Optional[Dict[str, int]] = None
) -> None:
    """
    Configura el logger raíz y niveles específicos para módulos.
    
    Args:
        level: Nivel de logging por defecto
        log_file: Archivo de log principal
        modules_levels: Diccionario con niveles específicos por módulo
                       Ej: {"requests": logging.WARNING}
    """
    # Configurar handler básico para evitar "No handler found" warnings
    logging.basicConfig(level=level, handlers=[logging.NullHandler()])
    
    # Configurar niveles específicos
    if modules_levels:
        for module, lvl in modules_levels.items():
            logging.getLogger(module).setLevel(lvl)
    
    # Configurar logger raíz si se especifica archivo
    if log_file:
        root_logger = logging.getLogger()
        root_logger.handlers.clear()  # Remover handlers por defecto
        
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        )
        root_logger.addHandler(file_handler)