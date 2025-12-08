# src/core/rag/advanced/trainer.py - VERSIÓN FUNCIONAL COMPLETA
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch
import numpy as np
from src.core.config import settings

class RLHFTrainer:
    def __init__(self, base_model_name: str = "distilbert-base-uncased", device: str = settings.DEVICE):
        self.base_model_name = base_model_name
        self.device = device
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.sep_token or "[PAD]"
        
        # Cargar modelo configurado para regresión
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, 
            num_labels=1,
            problem_type="regression"
        ).to(device)
        
        # Configuración adicional
        self.model.config.problem_type = "regression"
        
        # Crear directorio de modelos
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def prepare_rlhf_dataset_from_logs(self, failed_log_path: Path, success_log_path: Path) -> Dataset:
        """Prepara dataset simple para entrenamiento - VERSIÓN COMPATIBLE"""
        samples = []
        
        # Función para cargar logs
        def load_logs(filepath: Path, label: float):
            data = []
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                record['label'] = label
                                data.append(record)
                            except:
                                continue
            return data
        
        # Cargar ambos tipos de logs
        success_data = load_logs(success_log_path, 1.0)
        failed_data = load_logs(failed_log_path, 0.0)
        
        # Combinar y preparar
        for record in success_data + failed_data:
            query = record.get('query', '')
            response = record.get('response', '')
            label = float(record.get('label', 0.0))
            
            if query and response:
                samples.append({
                    'query': query,
                    'response': response,
                    'labels': label
                })
        
        print(f"📊 Dataset preparado: {len(samples)} ejemplos")
        if samples:
            dataset = Dataset.from_list(samples)
            print(f"📊 Columnas del dataset: {dataset.column_names}")
            return dataset
        else:
            return Dataset.from_dict({'query': [], 'response': [], 'labels': []})

    def train(self, dataset: Dataset, save_dir: Path = None) -> Dict[str, Any]:
        """Entrenamiento RLHF simplificado y funcional"""
        try:
            # CORRECCIÓN: Asegurar que el dataset tenga estructura correcta
            print(f"📊 Dataset recibido: {len(dataset)} ejemplos")
            print(f"📊 Columnas: {dataset.column_names}")
            
            # Verificar y preparar datos
            if len(dataset) < 10:
                print("⚠️ Dataset demasiado pequeño para entrenar")
                return {'error': 'insufficient_data'}
            
            # 1. Asegurar que tenemos las columnas necesarias
            required_columns = {'query', 'response', 'labels'}
            available_columns = set(dataset.column_names)
            
            if not required_columns.issubset(available_columns):
                print(f"❌ Faltan columnas: {required_columns - available_columns}")
                # Intentar crear las columnas faltantes
                dataset = dataset.map(self._fix_dataset_columns)
            
            # 2. Tokenización simplificada
            def tokenize_function(examples):
                # Crear texto combinado
                texts = [
                    f"Query: {q} Response: {r}"
                    for q, r in zip(examples['query'], examples['response'])
                ]
                return self.tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=256
                )
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=[col for col in dataset.column_names if col != "labels"]
            )

            
            # 3. Añadir labels
            def add_labels(examples):
                examples['labels'] = examples['labels']
                return examples
            
            tokenized_dataset = tokenized_dataset.map(
                add_labels,
                batched=True
            )
            
            # 4. Configuración de entrenamiento mínima
            training_args = TrainingArguments(
                output_dir=str(save_dir),
                num_train_epochs=2,
                per_device_train_batch_size=4,
                learning_rate=2e-5,
                warmup_steps=50,
                weight_decay=0.01,
                logging_dir=str(save_dir / "logs"),
                logging_steps=10,
                save_strategy="no",
                report_to="none",
                remove_unused_columns=False
            )
            
            # 5. Trainer simple
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset
            )
            
            # 6. Entrenar
            print("🚀 Iniciando entrenamiento...")
            trainer.train()
            
            # 7. Guardar
            trainer.save_model(str(save_dir))
            self.tokenizer.save_pretrained(str(save_dir))
            
            return {'success': True, 'samples': len(dataset)}
            
        except Exception as e:
            print(f"❌ Error en entrenamiento: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}

    def _fix_dataset_columns(self, example):
        """Corrige columnas faltantes en el dataset."""
        fixed = {}
        
        # Asegurar query
        if 'query' not in example:
            fixed['query'] = example.get('question', '') or example.get('text', '')
        else:
            fixed['query'] = example['query']
        
        # Asegurar response
        if 'response' not in example:
            fixed['response'] = example.get('answer', '') or example.get('generation', '')
        else:
            fixed['response'] = example['response']
        
        # Asegurar labels
        if 'labels' not in example:
            # Intentar extraer de feedback
            feedback = example.get('feedback', 3)
            if isinstance(feedback, (int, float)):
                # Normalizar a 0-1
                fixed['labels'] = min(1.0, max(0.0, feedback / 5.0))
            else:
                fixed['labels'] = 0.5  # Neutral
        else:
            fixed['labels'] = example['labels']
        
        return fixed