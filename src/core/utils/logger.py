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
def get_logger(
    name: str,
    *,
    log_file: Optional[str] = None,
    level: int = logging.WARNING,  # Cambia de logging.INFO a logging.WARNING
    json_format: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Obtain or reuse a configured logger.

    Parameters
    ----------
    name : str
        Logger name (usually __name__)
    log_file : str or Path, optional
        When given, a RotatingFileHandler is added.
    level : int
        Logging level for this logger.
    json_format : bool
        Use JSONFormatter instead of plain text.
    max_bytes : int
        Max size of each log file before rotation.
    backup_count : int
        Number of rotated files to keep.

    Returns
    -------
    logging.Logger
        Ready-to-use logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers on reload (Jupyter, pytest, etc.)
    if logger.handlers:
        return logger

    formatter = JSONFormatter() if json_format else logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stderr)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Optional file handler
    if log_file:
        log_path = Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(level)
    logger.propagate = False  # Avoid double logging from root
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