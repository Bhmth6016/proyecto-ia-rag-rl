import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib
import re

from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class FeedbackProcessor:
    """
    Procesador de feedback SIN TRADUCCIÓN.

    Funcionalidad:
    - Cargar exclusivamente de conversation_*.json en data/processed/historial
    - Generar failed_queries.log solo con feedback < 4
    - Generar success_queries.log solo con feedback >= 4
    - Trabaja directamente con 'query' y 'response'
    """

    def __init__(
        self,
        feedback_dir: str = "data/feedback",
        max_workers: int = 4,
        batch_size: int = 10,
        flush_interval: float = 5.0,
    ):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        # Concurrencia
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Buffers
        self.feedback_buffer: List[Dict[str, Any]] = []
        self.last_flush = datetime.now()

        # Archivos
        self.failed_queries_log = Path("data/feedback/failed_queries.log")
        self.failed_queries_log.parent.mkdir(parents=True, exist_ok=True)
        self.existing_failed_ids = self._load_existing_queries(self.failed_queries_log)

        self.success_queries_log = Path("data/feedback/success_queries.log")
        self.success_queries_log.parent.mkdir(parents=True, exist_ok=True)
        self.existing_success_ids = self._load_existing_queries(self.success_queries_log)

        # Pre-cargar historial
        self._load_historial_queries()

        # Iniciar flush periódico
        self._flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
        self._flush_thread.start()

        logger.info("FeedbackProcessor inicializado correctamente (sin traducción).")

    # ----------------------------------------------------------
    # CARGA DEL HISTORIAL
    # ----------------------------------------------------------
    def _load_historial_queries(self):
        historial_dir = Path("data/processed/historial")
        logger.info(f"Cargando historial desde: {historial_dir}")

        if not historial_dir.exists():
            logger.error(f"Directorio no encontrado: {historial_dir}")
            return

        total_checked = 0
        added_failures = 0
        added_successes = 0

        for file in historial_dir.glob("conversation_*.json"):
            try:
                with file.open("r", encoding="utf-8") as f:
                    conversations = json.load(f)
                    if not isinstance(conversations, list):
                        continue

                    for conv in conversations:
                        total_checked += 1
                        if not isinstance(conv, dict):
                            continue

                        raw_feedback = conv.get("feedback")
                        if raw_feedback is None:
                            continue

                        try:
                            feedback = float(raw_feedback)
                        except:
                            continue

                        if feedback < 4:
                            added_failures += self._process_failed_conversation(conv, file.name)
                        else:
                            added_successes += self._process_successful_conversation(conv, file.name)

            except Exception as e:
                logger.error(f"Error procesando {file.name}: {e}", exc_info=True)

        logger.info(
            f"Historial cargado: {total_checked} evaluados — "
            f"{added_failures} fallidos añadidos, "
            f"{added_successes} exitosos añadidos."
        )

    # ----------------------------------------------------------
    # PROCESAMIENTO DE CONVERSACIONES
    # ----------------------------------------------------------
    def _process_failed_conversation(self, conv: Dict, source_file: str) -> int:
        if not self._is_valid_failed_conversation(conv):
            return 0

        record = self._create_failure_record(conv, source_file)
        entry_id = self._generate_entry_id(conv)

        if entry_id not in self.existing_failed_ids:
            self._append_to_queries_log(self.failed_queries_log, record)
            self.existing_failed_ids.add(entry_id)
            return 1

        return 0

    def _process_successful_conversation(self, conv: Dict, source_file: str) -> int:
        if not self._is_valid_success_conversation(conv):
            return 0

        record = self._create_success_record(conv, source_file)
        entry_id = self._generate_entry_id(conv)

        if entryId := entry_id not in self.existing_success_ids:
            self._append_to_queries_log(self.success_queries_log, record)
            self.existing_success_ids.add(entry_id)
            return 1

        return 0

    # ----------------------------------------------------------
    # VALIDACIÓN
    # ----------------------------------------------------------
    def _is_valid_failed_conversation(self, conv: Dict) -> bool:
        try:
            return all(k in conv for k in ["query", "response", "feedback"]) and float(conv["feedback"]) < 4
        except:
            return False

    def _is_valid_success_conversation(self, conv: Dict) -> bool:
        try:
            return all(k in conv for k in ["query", "response", "feedback"]) and float(conv["feedback"]) >= 4
        except:
            return False

    # ----------------------------------------------------------
    # CREACIÓN DE REGISTROS
    # ----------------------------------------------------------
    def _create_failure_record(self, conv: Dict, source_file: str) -> Dict:
        return {
            "timestamp": conv.get("timestamp"),
            "session_id": conv.get("session_id"),
            "query": conv["query"],
            "response": conv["response"],
            "feedback": float(conv["feedback"]),
            "failure_reason": self._diagnose_failure(conv),
            "source_file": source_file,
            "processed": False
        }

    def _create_success_record(self, conv: Dict, source_file: str) -> Dict:
        return {
            "timestamp": conv.get("timestamp"),
            "session_id": conv.get("session_id"),
            "query": conv["query"],
            "response": conv["response"],
            "feedback": float(conv["feedback"]),
            "selected_product_id": self._extract_product_id(conv["response"]),
            "source_file": source_file,
            "processed": False
        }

    # ----------------------------------------------------------
    # EXTRAER PRODUCT ID
    # ----------------------------------------------------------
    def _extract_product_id(self, response: str) -> Optional[str]:
        match = re.search(r"(ASIN|ID)[:\s]*([A-Z0-9]{10})", response, re.IGNORECASE)
        return match.group(2) if match else None

    # ----------------------------------------------------------
    # MANEJO DE LOGS Y DUPLICADOS
    # ----------------------------------------------------------
    def _load_existing_queries(self, log_file: Path) -> set:
        existing = set()

        if log_file.exists():
            try:
                with log_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            query_text = entry.get("query", "")
                            if query_text:
                                qhash = hashlib.md5(query_text.encode("utf-8")).hexdigest()
                                entry_id = f"{entry.get('session_id','')}-{qhash}"
                                existing.add(entry_id)
                        except:
                            continue
            except Exception as e:
                logger.error(f"Error leyendo {log_file}: {e}")

        return existing

    def _generate_entry_id(self, conv: Dict) -> str:
        query_text = conv.get("query") or ""
        h = hashlib.md5(query_text.encode("utf-8")).hexdigest()
        return f"{conv.get('session_id', '')}-{h}"

    def _append_to_queries_log(self, log_file: Path, record: Dict):
        try:
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Error escribiendo en {log_file}: {e}", exc_info=True)

    # ----------------------------------------------------------
    # GUARDADO DE FEEDBACK EN TIEMPO REAL
    # ----------------------------------------------------------
    def save_feedback(
        self,
        query: str,
        answer: str,
        rating: int,
        retrieved_docs: Optional[List[str]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:

        try:
            rating = int(rating)

            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "response": answer,
                "feedback": rating,
                "processed": False
            }

            if extra_meta:
                record.update(extra_meta)

            if rating < 4:
                record["failure_reason"] = self._diagnose_failure(record)
                self._write_feedback(record, is_success=False)
            else:
                record["selected_product_id"] = self._extract_product_id(answer)
                self._write_feedback(record, is_success=True)

            with self.lock:
                self.feedback_buffer.append(record)
                if len(self.feedback_buffer) >= self.batch_size:
                    self._flush_buffer()

        except Exception as e:
            logger.error(f"Error registrando feedback: {e}", exc_info=True)

    def _write_feedback(self, record: Dict, is_success: bool):
        log_file = self.success_queries_log if is_success else self.failed_queries_log
        existing_ids = self.existing_success_ids if is_success else self.existing_failed_ids

        try:
            entry_id = self._generate_entry_id(record)
            if entry_id not in existing_ids:
                existing_ids.add(entry_id)
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Error escribiendo feedback: {e}", exc_info=True)

    # ----------------------------------------------------------
    # DIAGNÓSTICO DE FALLAS
    # ----------------------------------------------------------
    def _diagnose_failure(self, conv: Dict) -> str:
        response = conv["response"].lower()

        if "error" in response:
            return "system_error"
        if any(w in response for w in ["no disponible", "no especificad"]):
            return "incomplete_data"
        if any(w in response for w in ["no encontrado", "no tenemos"]):
            return "product_not_found"
        if conv["query"].lower() in response:
            return "echo_response"
        return "low_quality_response"

    # ----------------------------------------------------------
    # FLUSH DEL BUFFER
    # ----------------------------------------------------------
    def _flush_buffer(self):
        with self.lock:
            if not self.feedback_buffer:
                return
            batch = self.feedback_buffer.copy()
            self.feedback_buffer.clear()
            self.last_flush = datetime.now()

        file_path = self._today_file()
        try:
            with file_path.open("a", encoding="utf-8") as f:
                for rec in batch:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Flush fallido: {e}")
            with self.lock:
                self.feedback_buffer.extend(batch)

    def _periodic_flush(self):
        while True:
            time.sleep(1.0)
            with self.lock:
                if self.feedback_buffer and (datetime.now() - self.last_flush).total_seconds() >= self.flush_interval:
                    self.executor.submit(self._flush_buffer)

    def _today_file(self) -> Path:
        return self.feedback_dir / f"feedback_{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"

    # ----------------------------------------------------------
    # CIERRE
    # ----------------------------------------------------------
    def close(self):
        self._flush_buffer()
        self.executor.shutdown(wait=True)
        logger.info("FeedbackProcessor cerrado limpiamente.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
