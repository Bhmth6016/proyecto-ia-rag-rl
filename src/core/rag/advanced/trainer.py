# src/core/rag/advanced/trainer.py
import json
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset  # <-- Importación faltante
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
        samples = []

        # Buenas respuestas -> score 1.0
        if success_log.exists():
            with open(success_log, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        samples.append({
                            "query": record["query_es"],
                            "answer": record["response_es"],
                            "score": 1.0
                        })
                    except Exception:
                        continue

        # Malas respuestas -> score 0.0
        if failed_log.exists():
            with open(failed_log, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        samples.append({
                            "query": record["query_es"],
                            "answer": record["response_es"],
                            "score": 0.0
                        })
                    except Exception:
                        continue

        if len(samples) < min_samples:
            raise ValueError(f"Se requieren al menos {min_samples} muestras, pero solo hay {len(samples)}")

        return Dataset.from_list(samples)



    def train(self, dataset: Dataset, save_dir: Path = None) -> None:
        """Fine-tuning del modelo con RLHF."""
        save_dir = save_dir or settings.MODELS_DIR / "rlhf_model"

        def tokenize(batch):
            merged = [q + " " + a for q, a in zip(batch["query"], batch["answer"])]
            return self.tokenizer(
                merged,
                padding="max_length",
                truncation=True,
                max_length=512
            )

        dataset = dataset.map(tokenize, batched=True)
        
        # ✅ Rename 'score' to 'labels' so the model can use it
        dataset = dataset.rename_column("score", "labels")
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        print("TrainingArguments viene de:", TrainingArguments.__module__)

        training_args = TrainingArguments(
            output_dir=str(save_dir),
            per_device_train_batch_size=4,
            num_train_epochs=3,
            logging_dir=str(save_dir / "logs"),
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            compute_metrics=lambda p: {"mse": ((p.predictions - p.label_ids) ** 2).mean()},
        )

        trainer.train()
        trainer.save_model(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        print(f"✅ Modelo guardado en: {save_dir}")


    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        """Evalúa el modelo fine-tuned."""
        trainer = Trainer(
            model=self.model,
            compute_metrics=lambda p: {"mse": ((p.predictions - p.label_ids) ** 2).mean()},
        )
        return trainer.evaluate(dataset)