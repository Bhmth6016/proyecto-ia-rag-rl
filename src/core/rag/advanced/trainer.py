# src/core/rag/advanced/trainer.py - VERSIÓN COMPATIBLE
import json
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch
from src.core.config import settings

class RLHFTrainer:
    def __init__(self, base_model_name: str = "distilbert-base-uncased", device: str = settings.DEVICE):
        self.base_model_name = base_model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1).to(device)
        
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def prepare_rlhf_dataset_from_logs(self, failed_log: Path, success_log: Path, min_samples: int = 10) -> Dataset:
        """VERSIÓN CORREGIDA - Compatible con el formato real de datos"""
        samples = []

        # ✅ CORREGIDO: Usar "query" y "response" en lugar de "query_es" y "response_es"
        # Buenas respuestas -> score 1.0
        if success_log.exists():
            with open(success_log, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line)
                        
                        # ✅ CORREGIDO: Campos correctos
                        query = record.get("query", "")
                        response = record.get("response", "")
                        
                        if query and response:  # Solo agregar si hay datos válidos
                            samples.append({
                                "query": query,
                                "answer": response,
                                "score": 1.0
                            })
                            print(f"✅ Success sample {line_num}: {query[:50]}...")
                            
                    except Exception as e:
                        print(f"❌ Error en success log línea {line_num}: {e}")
                        continue

        # Malas respuestas -> score 0.0
        if failed_log.exists():
            with open(failed_log, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line)
                        
                        # ✅ CORREGIDO: Campos correctos
                        query = record.get("query", "")
                        response = record.get("response", "")
                        
                        if query and response:  # Solo agregar si hay datos válidos
                            samples.append({
                                "query": query,
                                "answer": response,
                                "score": 0.0
                            })
                            print(f"❌ Failed sample {line_num}: {query[:50]}...")
                            
                    except Exception as e:
                        print(f"❌ Error en failed log línea {line_num}: {e}")
                        continue

        print(f"📊 Total de muestras válidas encontradas: {len(samples)}")
        
        if len(samples) < min_samples:
            raise ValueError(f"Se requieren al menos {min_samples} muestras, pero solo hay {len(samples)}")

        return Dataset.from_list(samples)

    def train(self, dataset: Dataset, save_dir: Path = None) -> None:
        """Fine-tuning del modelo con RLHF - VERSIÓN COMPATIBLE"""
        save_dir = save_dir or settings.MODELS_DIR / "rlhf_model"
        
        print(f"🎯 Iniciando entrenamiento con {len(dataset)} ejemplos...")

        def tokenize(batch):
            # Combinar query y answer para el modelo
            merged = [f"Query: {q} Answer: {a}" for q, a in zip(batch["query"], batch["answer"])]
            return self.tokenizer(
                merged,
                padding="max_length",
                truncation=True,
                max_length=256,  # ✅ REDUCIDO para mayor eficiencia
                return_tensors="pt"
            )

        print("🔧 Tokenizando dataset...")
        dataset = dataset.map(tokenize, batched=True, batch_size=8)
        
        # ✅ Rename 'score' to 'labels' 
        dataset = dataset.rename_column("score", "labels")
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        print("⚙️ Configurando entrenamiento...")
        
        # ✅ VERSIÓN COMPATIBLE: Sin evaluation_strategy problemático
        training_args = TrainingArguments(
            output_dir=str(save_dir),
            per_device_train_batch_size=4,
            num_train_epochs=2,  # ✅ REDUCIDO para entrenamiento más rápido
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_dir=str(save_dir / "logs"),
            logging_steps=5,
            save_steps=100,
            # ✅ ELIMINADO: evaluation_strategy="no",  # Causaba el error
            # ✅ ELIMINADO: report_to=None,  # Podría causar problemas en algunas versiones
            disable_tqdm=False,  # ✅ MOSTRAR barra de progreso
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )

        print("🚀 Iniciando entrenamiento REAL (esto tomará varios minutos)...")
        print("💡 Por favor espera, el progreso se mostrará automáticamente...")
        
        trainer.train()
        
        print("💾 Guardando modelo...")
        trainer.save_model(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        
        print(f"✅ Modelo guardado en: {save_dir}")

    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        """Evalúa el modelo fine-tuned."""
        trainer = Trainer(
            model=self.model,
        )
        return trainer.evaluate(dataset)