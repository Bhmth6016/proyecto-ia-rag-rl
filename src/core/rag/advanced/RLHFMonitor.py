# src/core/rag/advanced/RLHFMonitor.py
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

logger = logging.getLogger(__name__)

class RLHFMonitor:
    """Monitor para tracking de rendimiento RLHF"""
    
    def __init__(self, log_dir: Path = Path("data/feedback/rlhf_metrics")):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = log_dir / "training_metrics.jsonl"
        
    def log_training_session(self, 
                           examples_used: int,
                           previous_accuracy: float,
                           new_accuracy: float,
                           training_time: float):
        """Registra métricas de sesión de entrenamiento"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "examples_used": examples_used,
            "previous_accuracy": previous_accuracy,
            "new_accuracy": new_accuracy,
            "improvement": new_accuracy - previous_accuracy,
            "training_time_seconds": training_time,
            "success": new_accuracy > previous_accuracy
        }
        
        try:
            with open(self.metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"📊 Métricas RLHF registradas: {record['improvement']:.3f} mejora")
        except Exception as e:
            logger.error(f"Error registrando métricas RLHF: {e}")
    
    def get_training_stats(self, days: int = 7) -> Dict[str, Any]:
        """Obtiene estadísticas de entrenamiento de últimos N días"""
        try:
            improvements = []
            success_count = 0
            total_sessions = 0
            
            if self.metrics_file.exists():
                with open(self.metrics_file, "r", encoding="utf-8") as f:
                    for line in f:
                        record = json.loads(line)
                        timestamp = datetime.fromisoformat(record["timestamp"].replace('Z', '+00:00'))
                        
                        if datetime.now() - timestamp <= timedelta(days=days):
                            improvements.append(record["improvement"])
                            if record["success"]:
                                success_count += 1
                            total_sessions += 1
            
            return {
                "total_sessions": total_sessions,
                "success_rate": success_count / max(1, total_sessions),
                "avg_improvement": sum(improvements) / max(1, len(improvements)),
                "best_improvement": max(improvements) if improvements else 0,
                "worst_improvement": min(improvements) if improvements else 0
            }
        except Exception as e:
            logger.error(f"Error obteniendo stats RLHF: {e}")
            return {}