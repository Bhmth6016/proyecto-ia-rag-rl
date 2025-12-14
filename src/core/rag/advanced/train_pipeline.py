# src/core/rag/advanced/train_pipeline.py
import json
from pathlib import Path
from datetime import datetime
import schedule
import time
from typing import Optional
import logging
import torch

logger = logging.getLogger(__name__)

class RLHFTrainingPipeline:
    """Pipeline completo de entrenamiento RLHF"""
    
    def __init__(self, trainer_class=None):
        from src.core.rag.advanced.trainer import RLHFTrainer
        self.trainer_class = trainer_class or RLHFTrainer
        self.model_dir = Path("data/models/rlhf_model")
        self.feedback_dir = Path("data/feedback")
        
    def train_from_feedback(self, min_samples: int = 10) -> Optional[dict]:
        """Entrena modelo usando logs de feedback"""
        try:
            trainer = self.trainer_class()
            
            # Preparar dataset
            dataset = trainer.prepare_rlhf_dataset_from_logs(
                self.feedback_dir / "failed_queries.log",
                self.feedback_dir / "success_queries.log"
            )
            
            if len(dataset) < min_samples:
                logger.info(f"⚠️ Insuficientes muestras: {len(dataset)}/{min_samples}")
                return None
            
            # Entrenar
            results = trainer.train(dataset, self.model_dir)
            
            # Guardar metadatos
            metadata = {
                "trained_at": datetime.now().isoformat(),
                "samples": len(dataset),
                "loss": results.get('train_loss', 0),
                "training_time": results.get('training_time', 0)
            }
            
            with open(self.model_dir / "training_metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"✅ RLHF entrenado con {len(dataset)} muestras (loss: {metadata['loss']:.4f})")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error entrenando RLHF: {e}")
            return None
    
    def load_model(self):
        """Carga modelo entrenado"""
        try:
            if not (self.model_dir / "pytorch_model.bin").exists():
                logger.info("ℹ️  No hay modelo RLHF entrenado aún")
                return None
            
            logger.info(f"🔍 Intentando cargar modelo RLHF de {self.model_dir}")
            
            # Primero verificar archivos
            model_files = list(self.model_dir.glob("*"))
            logger.info(f"📁 Archivos en directorio: {[f.name for f in model_files]}")
            
            # Cargar trainer
            trainer = self.trainer_class()
            
            # Intentar cargar modelo
            try:
                trainer.model.load_state_dict(
                    torch.load(self.model_dir / "pytorch_model.bin", 
                            map_location=torch.device('cuda'))
                )
                trainer.model.eval()
                logger.info("✅ Modelo RLHF cargado exitosamente")
                return trainer
            except Exception as e:
                logger.error(f"❌ Error cargando pesos del modelo: {e}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error general cargando modelo RLHF: {e}")
            import traceback
            traceback.print_exc()
            return None