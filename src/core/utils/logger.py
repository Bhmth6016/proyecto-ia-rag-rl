# src/core/utils/logger.py
"""
Centralised, reusable logging utilities.

Features
--------
• JSON or plain-text formatting  
• Console + rotating file handlers  
• Per-module level overrides  
• Thread-safe, no duplicate handlers
"""

from __future__ import annotations

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
# Root logger quick-setup
# --------------------------------------------------
# src/core/utils/logger.py

def configure_root_logger(
    *,
    level: int = logging.ERROR,  # Cambia de logging.INFO a logging.ERROR
    log_file: Optional[str] = None,
    module_levels: Optional[Dict[str, int]] = None,
) -> None:
    """
    One-call configuration for the root logger (useful for scripts).

    Example
    -------
    configure_root_logger(
        level=logging.ERROR,
        log_file="logs/app.log",
        module_levels={"urllib3": logging.WARNING, "transformers": logging.ERROR},
    )
    """
    if module_levels:
        for module, lvl in module_levels.items():
            logging.getLogger(module).setLevel(lvl)

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