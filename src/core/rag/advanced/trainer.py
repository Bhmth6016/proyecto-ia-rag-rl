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
        """Entrenamiento RLHF robusto y funcional"""
        import time
        start_time = time.time()
        
        save_dir = save_dir or settings.MODELS_DIR / "rlhf_model"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 Entrenando con {len(dataset)} ejemplos...")
        
        # 1. VERIFICAR COLUMNAS DISPONIBLES
        print(f"📊 Columnas disponibles: {dataset.column_names}")
        
        # 2. CREAR TEXTO COMBINADO (manejar diferentes estructuras)
        def create_combined_text(example):
            """Crea texto combinado a partir de query y response"""
            if 'text' in example:
                # Ya tiene columna 'text'
                return example
            
            # Construir texto combinado
            query = example.get('query', '')
            response = example.get('response', '')
            
            if not query and 'answer' in example:
                query = example.get('answer', '')
            
            example['text'] = f"Query: {query} Response: {response}"
            return example
        
        # Aplicar transformación
        dataset = dataset.map(create_combined_text)
        
        # 3. TOKENIZACIÓN
        def tokenize_function(examples):
            # Usar columna 'text' que acabamos de crear
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
        
        # Tokenizar sin remover columnas primero
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
        )
        
        # Ahora remover columnas no necesarias
        columns_to_remove = [col for col in tokenized_dataset.column_names 
                           if col not in ['input_ids', 'attention_mask', 'labels']]
        if columns_to_remove:
            tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
        
        # 4. ASEGURAR QUE LABELS SON FLOATS
        def convert_labels_to_float(examples):
            if 'labels' in examples:
                # Convertir cada label a float
                labels = []
                for label in examples['labels']:
                    if isinstance(label, (int, np.integer)):
                        labels.append(float(label))
                    elif isinstance(label, (float, np.floating)):
                        labels.append(float(label))
                    else:
                        try:
                            labels.append(float(label))
                        except:
                            labels.append(0.0)
                examples['labels'] = labels
            return examples
        
        tokenized_dataset = tokenized_dataset.map(
            convert_labels_to_float,
            batched=True
        )
        
        # 5. CONFIGURAR FORMATO PARA PYTORCH
        tokenized_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        
        # 6. VERIFICAR TIPOS DE DATOS
        print("🔍 Verificando tipos de datos finales...")
        if len(tokenized_dataset) > 0:
            sample = tokenized_dataset[0]
            for key, value in sample.items():
                dtype = value.dtype if hasattr(value, 'dtype') else type(value)
                shape = value.shape if hasattr(value, 'shape') else 'N/A'
                print(f"  {key}: dtype={dtype}, shape={shape}")
        
        # 7. CONFIGURACIÓN DE ENTRENAMIENTO
        training_args = TrainingArguments(
            output_dir=str(save_dir),
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_dir=str(save_dir / "logs"),
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            disable_tqdm=False,
            use_cpu=settings.DEVICE == "cpu",
            fp16=False,
            remove_unused_columns=False,
        )
        
        # 8. TRAINER SIMPLIFICADO (sin compute_loss personalizado)
        # Primero, intentar con Trainer estándar
        try:
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
            )
            
            print("🚀 Iniciando entrenamiento con Trainer estándar...")
            train_result = trainer.train()
            
        except Exception as e:
            print(f"⚠️ Trainer estándar falló: {e}")
            print("🔄 Intentando con Trainer personalizado para regresión...")
            
            # Trainer personalizado para regresión
            class RegressionTrainer(Trainer):
                def compute_loss(self, model, inputs, return_outputs=False):
                    # Extraer labels
                    labels = inputs.get("labels")
                    
                    # Remover labels de inputs para el forward pass
                    inputs_without_labels = {k: v for k, v in inputs.items() if k != "labels"}
                    
                    # Forward pass
                    outputs = model(**inputs_without_labels)
                    logits = outputs.logits
                    
                    # Calcular MSE loss para regresión
                    if labels is not None:
                        # Asegurar que labels y logits tienen formas compatibles
                        if labels.dim() == 1:
                            labels = labels.unsqueeze(-1)
                        if logits.dim() == 1:
                            logits = logits.unsqueeze(-1)
                        
                        loss_fct = torch.nn.MSELoss()
                        loss = loss_fct(logits, labels.float())
                    else:
                        loss = outputs.loss if hasattr(outputs, 'loss') else None
                    
                    return (loss, outputs) if return_outputs else loss
            
            trainer = RegressionTrainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
            )
            
            train_result = trainer.train()
        
        # 9. GUARDAR RESULTADOS
        training_time = time.time() - start_time
        
        print("💾 Guardando modelo...")
        trainer.save_model(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))
        
        print(f"✅ Entrenamiento completado en {training_time:.2f} segundos")
        print(f"📉 Pérdida final: {train_result.training_loss:.4f}")
        
        return {
            'training_time': training_time,
            'train_loss': train_result.training_loss,
            'model_path': str(save_dir)
        }