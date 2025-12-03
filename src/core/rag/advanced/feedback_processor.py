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
    - 🔥 NUEVO: Tracking de métricas ML para análisis de rendimiento
    """

    def __init__(
        self,
        feedback_dir: str = "data/feedback",
        max_workers: int = 4,
        batch_size: int = 10,
        flush_interval: float = 5.0,
        track_ml_metrics: bool = True,  # 🔥 NUEVO: Habilitar tracking ML
    ):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        # 🔥 NUEVO: Configuración de tracking ML
        self.track_ml_metrics = track_ml_metrics
        self.ml_metrics_log = Path("data/feedback/ml_metrics.log")
        self.ml_metrics_log.parent.mkdir(parents=True, exist_ok=True)

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

        logger.info(f"FeedbackProcessor inicializado correctamente (sin traducción). Track ML: {track_ml_metrics}")

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
        record = {
            "timestamp": conv.get("timestamp"),
            "session_id": conv.get("session_id"),
            "query": conv["query"],
            "response": conv["response"],
            "feedback": float(conv["feedback"]),
            "failure_reason": self._diagnose_failure(conv),
            "source_file": source_file,
            "processed": False
        }
        
        # 🔥 NUEVO: Añadir métricas ML si están presentes en la conversación
        if "ml_metrics" in conv:
            record["ml_metrics"] = conv["ml_metrics"]
            
        return record

    def _create_success_record(self, conv: Dict, source_file: str) -> Dict:
        record = {
            "timestamp": conv.get("timestamp"),
            "session_id": conv.get("session_id"),
            "query": conv["query"],
            "response": conv["response"],
            "feedback": float(conv["feedback"]),
            "selected_product_id": self._extract_product_id(conv["response"]),
            "source_file": source_file,
            "processed": False
        }
        
        # 🔥 NUEVO: Añadir métricas ML si están presentes en la conversación
        if "ml_metrics" in conv:
            record["ml_metrics"] = conv["ml_metrics"]
            
        return record

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

            # 🔥 NUEVO: Extraer y procesar métricas ML del answer
            ml_metrics = self._extract_ml_metrics(answer)
            
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "response": answer,
                "feedback": rating,
                "processed": False
            }
            
            # 🔥 NUEVO: Añadir métricas ML al registro si están disponibles
            if ml_metrics:
                record["ml_metrics"] = ml_metrics

            # Combinar con metadata adicional
            extra_meta = extra_meta or {}
            if extra_meta:
                record.update(extra_meta)
                
            # 🔥 NUEVO: Verificar si la respuesta usó ML
            self._add_ml_tracking_info(answer, record)

            if rating < 4:
                record["failure_reason"] = self._diagnose_failure(record)
                self._write_feedback(record, is_success=False)
            else:
                record["selected_product_id"] = self._extract_product_id(answer)
                self._write_feedback(record, is_success=True)

            # 🔥 NUEVO: Registrar métricas ML separadamente para análisis
            if self.track_ml_metrics and ml_metrics:
                self._log_ml_metrics(record, ml_metrics)

            with self.lock:
                self.feedback_buffer.append(record)
                if len(self.feedback_buffer) >= self.batch_size:
                    self._flush_buffer()

        except Exception as e:
            logger.error(f"Error registrando feedback: {e}", exc_info=True)

    def _add_ml_tracking_info(self, answer: Any, record: Dict):
        """
        Añade información de tracking ML basada en atributos del answer
        """
        if not self.track_ml_metrics:
            return
            
        try:
            # Verificar si la respuesta usó ML
            if hasattr(answer, 'ml_scoring_method'):
                record['ml_method'] = answer.ml_scoring_method
                
            if hasattr(answer, 'ml_embeddings_used'):
                record['ml_embeddings_count'] = answer.ml_embeddings_used
                
            if hasattr(answer, 'ml_confidence_score'):
                record['ml_confidence'] = answer.ml_confidence_score
                
            if hasattr(answer, 'collaborative_filter_weight'):
                record['collab_filter_weight'] = answer.collaborative_filter_weight
                
            if hasattr(answer, 'rag_weight'):
                record['rag_weight'] = answer.rag_weight
                
        except Exception as e:
            logger.debug(f"No se pudieron extraer atributos ML del answer: {e}")

    def _extract_ml_metrics(self, answer: Any) -> Optional[Dict[str, Any]]:
        """
        Extrae métricas ML del answer (puede ser string o objeto)
        """
        if not self.track_ml_metrics:
            return None
            
        try:
            ml_metrics = {}
            
            # Si el answer es un string, buscar patrones ML
            if isinstance(answer, str):
                patterns = {
                    "ml_method": r"ML Method[:\s]*([\w_]+)",
                    "embeddings_used": r"Embeddings[:\s]*(\d+)",
                    "ml_score": r"ML Score[:\s]*([\d\.]+)",
                    "hybrid_weights": r"Weights[:\s]*RAG=([\d\.]+), Collab=([\d\.]+)",
                }
                
                for key, pattern in patterns.items():
                    match = re.search(pattern, answer, re.IGNORECASE)
                    if match:
                        if key == "hybrid_weights" and len(match.groups()) == 2:
                            ml_metrics["rag_weight"] = float(match.group(1))
                            ml_metrics["collab_weight"] = float(match.group(2))
                        else:
                            ml_metrics[key] = match.group(1)
            
            # Si el answer tiene atributos, extraerlos
            elif hasattr(answer, '__dict__'):
                attrs = ['ml_scoring_method', 'ml_embeddings_used', 'ml_confidence_score',
                        'collaborative_filter_weight', 'rag_weight']
                for attr in attrs:
                    if hasattr(answer, attr):
                        value = getattr(answer, attr)
                        if value is not None:
                            ml_metrics[attr] = value
            
            return ml_metrics if ml_metrics else None
            
        except Exception as e:
            logger.debug(f"Error extrayendo métricas ML: {e}")
            return None

    def _log_ml_metrics(self, record: Dict, ml_metrics: Dict):
        """
        Registra métricas ML en archivo separado para análisis
        """
        try:
            ml_entry = {
                "timestamp": record.get("timestamp"),
                "session_id": record.get("session_id"),
                "query": record.get("query", "")[:100],  # Primeros 100 chars
                "feedback": record.get("feedback"),
                "ml_metrics": ml_metrics,
                "failure_reason": record.get("failure_reason"),
                "selected_product_id": record.get("selected_product_id")
            }
            
            with self.ml_metrics_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ml_entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            logger.error(f"Error registrando métricas ML: {e}", exc_info=True)

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
    # ANÁLISIS ML (NUEVOS MÉTODOS)
    # ----------------------------------------------------------
    def get_ml_performance_report(self) -> Dict[str, Any]:
        """
        Genera un reporte de performance ML basado en métricas recolectadas
        """
        if not self.ml_metrics_log.exists():
            return {"error": "No hay datos ML disponibles"}
            
        try:
            ml_entries = []
            with self.ml_metrics_log.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        ml_entries.append(entry)
                    except:
                        continue
            
            if not ml_entries:
                return {"total_ml_entries": 0}
            
            # Calcular métricas
            total = len(ml_entries)
            success_count = sum(1 for e in ml_entries if e.get("feedback", 0) >= 4)
            failure_count = total - success_count
            
            # Métricas por método ML
            methods = {}
            for entry in ml_entries:
                ml_metrics = entry.get("ml_metrics", {})
                method = ml_metrics.get("ml_method", "unknown")
                
                if method not in methods:
                    methods[method] = {"total": 0, "success": 0}
                
                methods[method]["total"] += 1
                if entry.get("feedback", 0) >= 4:
                    methods[method]["success"] += 1
            
            # Calcular éxito por método
            for method, data in methods.items():
                data["success_rate"] = data["success"] / data["total"] if data["total"] > 0 else 0
            
            return {
                "total_ml_entries": total,
                "success_rate": success_count / total if total > 0 else 0,
                "methods_performance": methods,
                "time_range": {
                    "first": ml_entries[0].get("timestamp") if ml_entries else None,
                    "last": ml_entries[-1].get("timestamp") if ml_entries else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error generando reporte ML: {e}")
            return {"error": str(e)}

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