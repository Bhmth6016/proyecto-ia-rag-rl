import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib

from src.core.utils.logger import get_logger
from src.core.utils.translator import TextTranslator, Language

logger = get_logger(__name__)


class FeedbackProcessor:
    """
    Procesador de feedback especializado en:
    - Cargar exclusivamente de conversation_*.json en data/processed/historial
    - Generar failed_queries.log solo con feedback < 4
    - Generar success_queries.log solo con feedback >= 4
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

        # Configuración de concurrencia
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Buffers
        self.feedback_buffer: List[Dict[str, Any]] = []
        self.last_flush = datetime.now()

        # Failed queries log
        self.failed_queries_log = Path("data/feedback/failed_queries.log")
        self.failed_queries_log.parent.mkdir(parents=True, exist_ok=True)
        self.existing_failed_ids = self._load_existing_queries(self.failed_queries_log)
        
        # Success queries log (NEW)
        self.success_queries_log = Path("data/feedback/success_queries.log")
        self.success_queries_log.parent.mkdir(parents=True, exist_ok=True)
        self.existing_success_ids = self._load_existing_queries(self.success_queries_log)
        
        # Cargar historial al inicio
        self._load_historial_queries()

        # Translator
        self.translator = TextTranslator()

        # Iniciar flush periódico
        self._flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
        self._flush_thread.start()

        logger.info("FeedbackProcessor inicializado")

    # ----------------------------------------------------------
    # Métodos principales
    # ----------------------------------------------------------
    def _load_historial_queries(self):
        """Carga exclusivamente de conversation_*.json en data/processed/historial"""
        historial_dir = Path("data/processed/historial")
        logger.info(f"Cargando feedbacks desde: {historial_dir}")
        
        if not historial_dir.exists():
            logger.error(f"¡Directorio no encontrado! {historial_dir}")
            return

        added_failures = 0
        added_successes = 0
        total_checked = 0

        for file in historial_dir.glob("conversation_*.json"):
            try:
                with file.open("r", encoding="utf-8") as f:
                    conversations = json.load(f)
                    if not isinstance(conversations, list):
                        logger.debug(f"Archivo {file.name} no contiene una lista")
                        continue
                        
                    for conv in conversations:
                        total_checked += 1
                        if not isinstance(conv, dict):
                            continue
                            
                        # Procesar tanto exitosas como fallidas
                        feedback = conv.get("feedback")
                        if feedback is None:
                            continue
                            
                        try:
                            feedback = float(feedback)
                        except (ValueError, TypeError):
                            continue
                            
                        if feedback < 4:
                            added_failures += self._process_failed_conversation(conv, file.name)
                        else:
                            added_successes += self._process_successful_conversation(conv, file.name)
                            
            except Exception as e:
                logger.error(f"Error procesando {file.name}: {str(e)}", exc_info=True)

        logger.info(f"Resumen: {total_checked} conversaciones revisadas, {added_failures} fallidas, {added_successes} exitosas")

    def _process_failed_conversation(self, conv: Dict, source_file: str) -> int:
        """Procesa una conversación fallida y devuelve 1 si se añadió"""
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
        """Procesa una conversación exitosa y devuelve 1 si se añadió (NEW)"""
        if not self._is_valid_success_conversation(conv):
            return 0
            
        record = self._create_success_record(conv, source_file)
        entry_id = self._generate_entry_id(conv)
        
        if entry_id not in self.existing_success_ids:
            self._append_to_queries_log(self.success_queries_log, record)
            self.existing_success_ids.add(entry_id)
            return 1
        return 0

    def _is_valid_failed_conversation(self, conv: Dict) -> bool:
        """Valida si la conversación tiene feedback negativo válido"""
        try:
            return (isinstance(conv, dict) and 
                   all(k in conv for k in ["query", "response", "feedback"]) and 
                   float(conv.get("feedback", 0)) < 4)
        except Exception:
            return False

    def _is_valid_success_conversation(self, conv: Dict) -> bool:
        """Valida si la conversación tiene feedback positivo válido (NEW)"""
        try:
            return (isinstance(conv, dict) and 
                   all(k in conv for k in ["query", "response", "feedback"]) and 
                   float(conv.get("feedback", 0)) >= 4)
        except Exception:
            return False

    def _create_failure_record(self, conv: Dict, source_file: str) -> Dict:
        """Crea un registro estandarizado para failed_queries.log"""
        return {
            "timestamp": conv.get("timestamp"),
            "session_id": conv.get("session_id"),
            "query_es": conv["query"],
            "response_es": conv["response"],
            "feedback": float(conv["feedback"]) if conv["feedback"] else 0,
            "failure_reason": self._diagnose_failure(conv),
            "source_file": source_file,
            "processed": False
        }

    def _create_success_record(self, conv: Dict, source_file: str) -> Dict:
        """Crea un registro estandarizado para success_queries.log (NEW)"""
        return {
            "timestamp": conv.get("timestamp"),
            "session_id": conv.get("session_id"),
            "query_es": conv["query"],
            "response_es": conv["response"],
            "feedback": float(conv["feedback"]) if conv["feedback"] else 5,
            "selected_product_id": self._extract_product_id(conv["response"]),
            "source_file": source_file,
            "processed": False
        }

    def _extract_product_id(self, response: str) -> Optional[str]:
        """Intenta extraer un ID de producto de la respuesta (NEW)"""
        # Busca patrones como ASIN: XXXXXXXX o ID: XXXXXXXX
        import re
        match = re.search(r"(ASIN|ID)[:\s]*([A-Z0-9]{10})", response, re.IGNORECASE)
        return match.group(2) if match else None

    def _load_existing_queries(self, log_file: Path) -> set:
        """Carga los IDs existentes para evitar duplicados (MODIFIED)"""
        existing = set()
        if log_file.exists():
            try:
                with log_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            entry_id = self._generate_entry_id(entry)
                            existing.add(entry_id)
                        except (json.JSONDecodeError, ValueError):
                            continue
            except Exception as e:
                logger.error(f"Error cargando {log_file.name} existentes: {e}")
        return existing

    def _generate_entry_id(self, conv: Dict) -> str:
        """Genera un ID único para control de duplicados"""
        query_text = conv.get("query") or conv.get("query_es")
        if not query_text:
            raise ValueError("La entrada no contiene campo 'query' o 'query_es'")
        query_hash = hashlib.md5(query_text.encode('utf-8')).hexdigest()
        return f"{conv.get('session_id', '')}-{query_hash}"

    def _append_to_queries_log(self, log_file: Path, record: Dict):
        """Escribe de forma atómica en el log especificado (NEW)"""
        try:
            with log_file.open("a", encoding="utf-8") as f:
                json_str = json.dumps(record, ensure_ascii=False)
                f.write(json_str + "\n")
                logger.debug(f"Escrito en {log_file.name}: {json_str}")
        except Exception as e:
            logger.error(f"Error escribiendo en {log_file.name}: {e}", exc_info=True)

    def save_feedback(
        self,
        query: str,
        answer: str,
        rating: int,
        retrieved_docs: Optional[List[str]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            rating = int(rating)  # Asegurar que es int
            
            # Crear registro básico
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "response": answer,
                "feedback": rating,
                "processed": False
            }
            
            # Añadir metadatos adicionales
            if extra_meta:
                record.update(extra_meta)
                
            # Procesar según el rating
            if rating < 4:
                record["failure_reason"] = self._diagnose_failure(record)
                self._write_feedback(record, is_success=False)
            else:
                record["selected_product_id"] = self._extract_product_id(answer)
                self._write_feedback(record, is_success=True)
                
            # Guardar en buffer para el archivo diario
            with self.lock:
                self.feedback_buffer.append(record)
                if len(self.feedback_buffer) >= self.batch_size:
                    self._flush_buffer()

        except Exception as e:
            logger.error(f"Error saving feedback: {str(e)}", exc_info=True)

    def _write_feedback(self, record: Dict, is_success: bool):
        """Escribe el feedback en el log correspondiente (MODIFIED)"""
        log_file = self.success_queries_log if is_success else self.failed_queries_log
        existing_ids = self.existing_success_ids if is_success else self.existing_failed_ids
        
        try:
            entry_id = self._generate_entry_id(record)
            if entry_id not in existing_ids:
                existing_ids.add(entry_id)
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                logger.info(f"Feedback registrado en {log_file.name}")
        except Exception as e:
            logger.error(f"Error escribiendo feedback: {str(e)}", exc_info=True)

    def _diagnose_failure(self, conv: Dict) -> str:
        """Diagnóstico mejorado del tipo de fallo"""
        response = conv["response"].lower()
        
        if "error" in response:
            return "system_error"
        elif any(phrase in response for phrase in ["no disponible", "no especificad"]):
            return "incomplete_data"
        elif any(phrase in response for phrase in ["no encontrado", "no tenemos"]):
            return "product_not_found"
        elif conv["query"].lower() in response:
            return "echo_response"
        else:
            return "low_quality_response"

    # ----------------------------------------------------------
    # Sistema de Flushing
    # ----------------------------------------------------------
    def _flush_buffer(self):
        """Escribe el buffer de feedback en disco"""
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
            logger.debug(f"Flushed {len(batch)} registros a {file_path}")
        except Exception as e:
            logger.error(f"Flush fallido: {e}")
            with self.lock:
                self.feedback_buffer.extend(batch)

    def _periodic_flush(self):
        """Flush automático periódico"""
        while True:
            time.sleep(1.0)
            with self.lock:
                if (self.feedback_buffer and 
                    (datetime.now() - self.last_flush).total_seconds() >= self.flush_interval):
                    self.executor.submit(self._flush_buffer)

    def _today_file(self) -> Path:
        """Genera el nombre del archivo diario"""
        return self.feedback_dir / f"feedback_{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"

    # ----------------------------------------------------------
    # Métodos de cierre
    # ----------------------------------------------------------
    def close(self):
        """Cierre ordenado del procesador"""
        self._flush_buffer()
        self.executor.shutdown(wait=True)
        logger.info("FeedbackProcessor cerrado")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()