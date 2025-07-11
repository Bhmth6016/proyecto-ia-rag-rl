# src/core/rag/advanced/feedback_processor.py
import json
import logging
import threading
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from src.core.utils.logger import get_logger

logger = get_logger(__name__)

class FeedbackProcessor:
    def __init__(
        self, 
        feedback_dir: str = "data/feedback",
        max_workers: int = 4,
        batch_size: int = 10,
        flush_interval: float = 5.0
    ):
        """
        Procesador de feedback con soporte para concurrencia y batch processing.
        
        Args:
            feedback_dir: Directorio para almacenar feedback
            max_workers: Número máximo de hilos para procesamiento paralelo
            batch_size: Tamaño máximo del lote para escritura en batch
            flush_interval: Intervalo en segundos para flush automático
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuración de concurrencia
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # Buffer para procesamiento por lotes
        self.feedback_buffer = []
        self.last_flush_time = datetime.now()
        
        # Hilo para flush periódico
        self.flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
        self.flush_thread.start()
        
        logger.info(f"FeedbackProcessor inicializado en {self.feedback_dir}")

    def _get_current_feedback_file(self) -> Path:
        """Obtiene el archivo de feedback para la fecha actual"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.feedback_dir / f"feedback_{today}.jsonl"

    def save_feedback(self, feedback_data: Dict) -> None:
        """
        Guarda feedback de forma asíncrona con manejo de concurrencia.
        
        Args:
            feedback_data: Diccionario con datos de feedback
        """
        if not isinstance(feedback_data, dict):
            logger.error("Feedback data must be a dictionary")
            return
            
        # Añadir metadatos
        feedback_data.update({
            "timestamp": datetime.now().isoformat(),
            "processed": False
        })
        
        # Añadir al buffer de forma thread-safe
        with self.lock:
            self.feedback_buffer.append(feedback_data)
            
        # Procesar en segundo plano si se alcanza el batch size
        if len(self.feedback_buffer) >= self.batch_size:
            self.executor.submit(self._flush_buffer)

    def _flush_buffer(self) -> None:
        """Escribe el buffer actual en el archivo de forma segura"""
        if not self.feedback_buffer:
            return
            
        try:
            with self.lock:
                current_buffer = self.feedback_buffer.copy()
                self.feedback_buffer.clear()
                self.last_flush_time = datetime.now()
            
            feedback_file = self._get_current_feedback_file()
            
            # Escritura thread-safe
            with threading.Lock():
                with open(feedback_file, "a", encoding="utf-8") as f:
                    for feedback in current_buffer:
                        f.write(json.dumps(feedback, ensure_ascii=False) + "\n")
                        
            logger.debug(f"Flushed {len(current_buffer)} feedback items to {feedback_file}")
            
        except Exception as e:
            logger.error(f"Error flushing feedback buffer: {str(e)}")
            # Reintentar más tarde
            with self.lock:
                self.feedback_buffer.extend(current_buffer)

    def _periodic_flush(self) -> None:
        """Flush automático periódico del buffer"""
        while True:
            try:
                time_since_flush = (datetime.now() - self.last_flush_time).total_seconds()
                if time_since_flush >= self.flush_interval and self.feedback_buffer:
                    self._flush_buffer()
                    
                # Esperar antes de verificar nuevamente
                threading.Event().wait(1.0)
                
            except Exception as e:
                logger.error(f"Error in periodic flush: {str(e)}")

    def prepare_rlhf_dataset(
        self, 
        min_rating: int = 4,
        max_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Prepara dataset para RLHF a partir de feedback acumulado.
        
        Args:
            min_rating: Rating mínimo para incluir en el dataset
            max_samples: Límite máximo de muestras a incluir
            
        Returns:
            Lista de ejemplos de entrenamiento
        """
        dataset = []
        feedback_files = sorted(self.feedback_dir.glob("feedback_*.jsonl"), reverse=True)
        
        for feedback_file in feedback_files:
            try:
                with open(feedback_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            rating = entry.get("rating", 0)
                            
                            if int(rating) >= min_rating:
                                dataset.append({
                                    "prompt": f"Usuario: {entry['query']}\nContexto: {entry.get('retrieved_titles', '')}\nRespuesta:",
                                    "response": entry.get("answer", ""),
                                    "metadata": {
                                        "rating": rating,
                                        "sources": entry.get("sources", []),
                                        "timestamp": entry.get("timestamp")
                                    }
                                })
                                
                                if max_samples and len(dataset) >= max_samples:
                                    return dataset
                                    
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Invalid feedback entry in {feedback_file}: {str(e)}")
                            continue
                            
            except IOError as e:
                logger.error(f"Error reading feedback file {feedback_file}: {str(e)}")
                continue
                
        return dataset

    def get_feedback_stats(self) -> Dict:
        """
        Estadísticas agregadas del feedback recibido.
        
        Returns:
            Dict con:
            - total: Total de feedbacks
            - ratings: Distribución de ratings
            - common_queries: Consultas más frecuentes
            - avg_rating: Rating promedio
        """
        stats = {
            "total": 0,
            "ratings": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "common_queries": {},
            "avg_rating": 0.0
        }
        
        rating_sum = 0
        feedback_files = self.feedback_dir.glob("feedback_*.jsonl")
        
        for feedback_file in feedback_files:
            try:
                with open(feedback_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            stats["total"] += 1
                            rating = int(entry["rating"])
                            stats["ratings"][rating] += 1
                            rating_sum += rating
                            
                            query = entry["query"].lower()
                            stats["common_queries"][query] = stats["common_queries"].get(query, 0) + 1
                            
                        except (json.JSONDecodeError, KeyError):
                            continue
                            
            except IOError:
                continue
                
        if stats["total"] > 0:
            stats["avg_rating"] = round(rating_sum / stats["total"], 2)
            
        # Ordenar consultas más frecuentes
        stats["common_queries"] = dict(
            sorted(stats["common_queries"].items(), key=lambda item: item[1], reverse=True)[:10]
        )
        
        return stats

    def close(self):
        """Libera recursos y asegura que todo el feedback sea guardado"""
        self._flush_buffer()
        self.executor.shutdown(wait=True)
        logger.info("FeedbackProcessor closed gracefully")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == "__main__":
    # Ejemplo de uso
    with FeedbackProcessor() as processor:
        # Simular feedback
        for i in range(15):
            processor.save_feedback({
                "query": f"Test query {i}",
                "answer": "Test answer",
                "rating": (i % 5) + 1,
                "retrieved_titles": ["Product 1", "Product 2"]
            })
        
        # Obtener estadísticas
        print(json.dumps(processor.get_feedback_stats(), indent=2))
        
        # Preparar dataset
        dataset = processor.prepare_rlhf_dataset(min_rating=3)
        print(f"\nPrepared RLHF dataset with {len(dataset)} samples")