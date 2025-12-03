from __future__ import annotations
# src/core/utils/logger.py
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# --------------------------------------------------
# JSON formatter
# --------------------------------------------------
class JSONFormatter(logging.Formatter):
    """
    Emits structured logs – perfect for ingestion by Loki, Datadog, etc.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "lvl": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return self._json_dumps(payload)

    @staticmethod
    def _json_dumps(obj: Any) -> str:
        import json

        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------
# Logger factory
# --------------------------------------------------
def get_logger(name: str, verbose: bool = False):
    logger = logging.getLogger(name)
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    # Configurar handlers si no existen
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(console)
    
    return logger


# --------------------------------------------------
# ML Logger específico
# --------------------------------------------------
def get_ml_logger(name: str = "ml_system", level: int = logging.INFO) -> logging.Logger:
    """
    Obtiene un logger específico para componentes ML con formato especial.
    
    Args:
        name: Nombre del logger (default: 'ml_system')
        level: Nivel de logging (default: INFO)
        
    Returns:
        Logger configurado para tracking ML
    """
    ml_logger = logging.getLogger(name)
    
    # Configurar solo si no tiene handlers
    if not ml_logger.handlers:
        ml_logger.setLevel(level)
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Formato específico para ML
        ml_formatter = logging.Formatter(
            fmt='%(asctime)s 🔥 ML [%(levelname)s] %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(ml_formatter)
        ml_logger.addHandler(console_handler)
        
        # Handler para archivo específico ML
        try:
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / "ml_system.log",
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            
            # Formato JSON para archivo (mejor para análisis)
            json_formatter = JSONFormatter()
            file_handler.setFormatter(json_formatter)
            ml_logger.addHandler(file_handler)
        except Exception as e:
            # Si falla el archivo, continuar solo con consola
            ml_logger.warning(f"No se pudo configurar archivo de log ML: {e}")
    
    # Evitar propagación al root logger para mantener logs separados
    ml_logger.propagate = False
    
    return ml_logger


# --------------------------------------------------
# Root logger quick-setup
# --------------------------------------------------
def configure_root_logger(
    *,
    level: int = logging.ERROR,
    log_file: Optional[str] = None,
    module_levels: Optional[Dict[str, int]] = None,
    enable_ml_logger: bool = True,  # 🔥 NUEVO: Habilitar logger ML
    ml_log_file: Optional[str] = "logs/ml_system.log",  # 🔥 NUEVO: Archivo específico ML
) -> None:
    """
    One-call configuration for the root logger (useful for scripts).

    Example
    -------
    configure_root_logger(
        level=logging.ERROR,
        log_file="logs/app.log",
        module_levels={"urllib3": logging.WARNING, "transformers": logging.ERROR},
        enable_ml_logger=True,
        ml_log_file="logs/ml_system.log"
    )
    """
    # Configurar niveles por módulo
    if module_levels:
        for module, lvl in module_levels.items():
            logging.getLogger(module).setLevel(lvl)

    # Configurar archivo de log principal
    if log_file:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        root = logging.getLogger()
        root.handlers.clear()  # remove defaults

        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
            )
        )
        root.addHandler(handler)
        root.setLevel(level)

    # 🔥 NUEVO: Configurar logger específico para ML
    if enable_ml_logger:
        _setup_ml_logger(ml_log_file)


def _setup_ml_logger(ml_log_file: Optional[str] = None) -> None:
    """
    Configura el logger específico para sistema ML.
    """
    ml_logger = logging.getLogger('ml_system')
    
    # Limpiar handlers existentes para evitar duplicados
    if ml_logger.handlers:
        for handler in ml_logger.handlers[:]:
            ml_logger.removeHandler(handler)
    
    ml_logger.setLevel(logging.INFO)
    
    # Handler para consola con formato especial ML
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s 🔥 ML [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    ml_logger.addHandler(console_handler)
    
    # Handler para archivo específico ML si se especifica
    if ml_log_file:
        try:
            ml_log_path = Path(ml_log_file).expanduser().resolve()
            ml_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(ml_log_path, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            
            # Usar JSON formatter para archivos ML
            json_formatter = JSONFormatter()
            file_handler.setFormatter(json_formatter)
            ml_logger.addHandler(file_handler)
        except Exception as e:
            ml_logger.warning(f"No se pudo configurar archivo de log ML: {e}")
    
    # No propagar al root logger
    ml_logger.propagate = False
    
    # 🔥 NUEVO: Configurar loggers específicos para componentes ML
    _setup_ml_component_loggers()


def _setup_ml_component_loggers() -> None:
    """
    Configura loggers específicos para diferentes componentes ML.
    """
    ml_components = {
        'ml_collaborative': '🤝',
        'ml_embeddings': '🔤',
        'ml_similarity': '📐',
        'ml_training': '🎯',
        'ml_evaluation': '📊'
    }
    
    for component, emoji in ml_components.items():
        comp_logger = logging.getLogger(f'ml_system.{component}')
        comp_logger.setLevel(logging.INFO)
        
        # Solo añadir handlers si no tiene
        if not comp_logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter(
                f'%(asctime)s {emoji} ML [{component.upper()}] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            comp_logger.addHandler(console_handler)
        
        comp_logger.propagate = False


# --------------------------------------------------
# Funciones utilitarias para logging ML
# --------------------------------------------------
def log_ml_metric(metric_name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> None:
    """
    Registra una métrica ML de forma estructurada.
    
    Args:
        metric_name: Nombre de la métrica (ej: 'accuracy', 'f1_score')
        value: Valor de la métrica
        tags: Tags adicionales para contexto
    """
    ml_logger = get_ml_logger("ml_metrics")
    
    log_data = {
        "metric": metric_name,
        "value": value,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if tags:
        log_data.update(tags)
    
    ml_logger.info(f"METRIC: {metric_name}={value}", extra={"ml_data": log_data})


def log_ml_event(event_type: str, details: Dict[str, Any]) -> None:
    """
    Registra un evento del sistema ML.
    
    Args:
        event_type: Tipo de evento (ej: 'model_trained', 'prediction_made')
        details: Detalles del evento
    """
    ml_logger = get_ml_logger("ml_events")
    
    event_data = {
        "event": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        **details
    }
    
    ml_logger.info(f"EVENT: {event_type}", extra={"ml_data": event_data})


def log_ml_performance(start_time: datetime, operation: str, metrics: Optional[Dict[str, Any]] = None) -> None:
    """
    Registra performance de operaciones ML.
    
    Args:
        start_time: Tiempo de inicio
        operation: Nombre de la operación
        metrics: Métricas adicionales
    """
    ml_logger = get_ml_logger("ml_performance")
    
    duration = (datetime.utcnow() - start_time).total_seconds()
    
    perf_data = {
        "operation": operation,
        "duration_seconds": duration,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if metrics:
        perf_data.update(metrics)
    
    ml_logger.info(f"PERF: {operation} took {duration:.3f}s", extra={"ml_data": perf_data})