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

    def prepare_rlhf_dataset_from_logs(
        self, failed_log_path: Path, success_log_path: Path, min_samples: int = 5
    ) -> Dict[str, Any]:
        """Prepara dataset RLHF desde logs - VERSIÓN CORREGIDA"""
        import logging
        logger = logging.getLogger(__name__)
        samples = []
        
        all_logs = []

        # Cargar success logs
        if success_log_path.exists():
            with open(success_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            data['label'] = 1
                            all_logs.append(data)
                        except json.JSONDecodeError:
                            logger.debug(f"Error decoding JSON en success log: {line[:50]}...")
                            continue

        # Cargar failed logs
        if failed_log_path.exists():
            with open(failed_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            data['label'] = 0
                            all_logs.append(data)
                        except json.JSONDecodeError:
                            logger.debug(f"Error decoding JSON en failed log: {line[:50]}...")
                            continue

        logger.info(f"📊 Total logs cargados: {len(all_logs)}")

        # Procesar muestras válidas
        for log in all_logs:
            query = log.get('query', '')
            response = log.get('response', '')
            label = log.get('label', 0)
            if query and response:
                samples.append({'query': query, 'answer': response, 'labels': label})

        logger.info(f"📊 Muestras válidas encontradas: {len(samples)}")

        # Dividir dataset en train/eval
        train_size = int(0.8 * len(samples))
        train_data = samples[:train_size]
        eval_data = samples[train_size:]

        return {
            'train': Dataset.from_list(train_data),
            'eval': Dataset.from_list(eval_data),
            'total_samples': len(samples)
        }


    def train(self, dataset: Dataset, save_dir: Path = None) -> Dict[str, Any]:
        """Fine-tuning del modelo con RLHF - VERSIÓN MEJORADA"""
        import time
        start_time = time.time()
        
        save_dir = save_dir or settings.MODELS_DIR / "rlhf_model"
        
        print(f"🎯 Iniciando entrenamiento con {len(dataset)} ejemplos...")

        def tokenize(batch):
            # Combinar query y answer para el modelo
            texts = []
            for i in range(len(batch["query"])):
                query = batch["query"][i]
                answer = batch["answer"][i] if "answer" in batch else batch.get("response", [""])[i]
                texts.append(f"Query: {query} Answer: {answer}")
            
            return self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )

        print("🔧 Tokenizando dataset...")
        
        # ✅ CORRECCIÓN: Verificar y preparar columnas correctamente
        if "labels" not in dataset.column_names and "score" in dataset.column_names:
            dataset = dataset.rename_column("score", "labels")
        
        dataset = dataset.map(tokenize, batched=True, batch_size=8)
        dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        print("⚙️ Configurando entrenamiento...")
        
        training_args = TrainingArguments(
            output_dir=str(save_dir),
            per_device_train_batch_size=4,
            num_train_epochs=2,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_dir=str(save_dir / "logs"),
            logging_steps=5,
            save_steps=100,
            disable_tqdm=False,
            # ✅ Configuración compatible
            evaluation_strategy="no",  # No evaluation during training
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )

        print("🚀 Iniciando entrenamiento...")
        train_result = trainer.train()
        
        training_time = time.time() - start_time
        
        print("💾 Guardando modelo...")
        trainer.save_model(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        
        print(f"✅ Modelo guardado en: {save_dir}")
        print(f"⏱️ Tiempo de entrenamiento: {training_time:.2f} segundos")
        
        return {
            'training_time': training_time,
            'train_loss': train_result.training_loss,
            'model_path': str(save_dir)
        }

    def evaluate(self, dataset: Dataset) -> Dict[str, float]:
        """Evalúa el modelo fine-tuned."""
        trainer = Trainer(
            model=self.model,
        )
        return trainer.evaluate(dataset)