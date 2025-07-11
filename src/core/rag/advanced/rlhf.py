# src/core/rag/advanced/rlhf.py
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

from src.core.utils.logger import get_logger
from src.core.rag.advanced.evaluators import RAGEvaluator

logger = get_logger(__name__)

class RLHFTrainer:
    def __init__(
        self,
        base_model_name: str = "google/flan-t5-large",
        reward_model_name: str = "facebook/roberta-hate-speech-dynabench-r4",
        device: Optional[str] = None,
        lora_rank: int = 8,
        use_cross_encoder: bool = True,
        mixed_precision: bool = True
    ):
        """
        Inicializa el entrenador RLHF con manejo robusto de dispositivos.
        
        Args:
            base_model_name: Modelo base para fine-tuning.
            reward_model_name: Modelo de recompensa pre-entrenado.
            device: Dispositivo para entrenamiento (None para auto-detección).
            lora_rank: Rango para LoRA (Low-Rank Adaptation).
            use_cross_encoder: Usar CrossEncoder para recompensas más precisas.
            mixed_precision: Usar mixed precision training si está disponible.
        """
        # Configuración automática de dispositivo
        self.device = self._resolve_device(device)
        self.mixed_precision = mixed_precision and self.device == "cuda"
        self.lora_rank = lora_rank
        self.use_cross_encoder = use_cross_encoder
        
        logger.info(f"Inicializando RLHFTrainer en dispositivo: {self.device}")
        logger.info(f"Mixed precision: {'activado' if self.mixed_precision else 'desactivado'}")

        # Inicializar modelos
        self._load_base_model(base_model_name)
        self._load_reward_models(reward_model_name)
        
        # Configuración PPO
        self.ppo_config = PPOConfig(
            batch_size=8,
            mini_batch_size=4,
            learning_rate=1.41e-5,
            log_with="tensorboard",
            project_kwargs={"logging_dir": "./logs"},
            gradient_accumulation_steps=1,
            optimize_cuda_cache=True,
            use_habana=False,
            seed=42
        )

    def _resolve_device(self, device: Optional[str]) -> str:
        """Determina el mejor dispositivo disponible."""
        if device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA no disponible, usando CPU")
            return "cpu"
        return device

    def _load_base_model(self, model_name: str) -> None:
        """Carga el modelo base con adaptación LoRA y manejo de dispositivo."""
        try:
            logger.info(f"Cargando modelo base: {model_name}")
            
            # Configuración de dispositivo y tipo de dato
            torch_dtype = torch.float16 if self.mixed_precision else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Configuración LoRA para fine-tuning eficiente
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.05,
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
            
            # Cargar modelo con soporte para diferentes dispositivos
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                model = model.to(torch.float32)
            
            self.base_model = get_peft_model(model, lora_config)
            self.base_model.print_trainable_parameters()
            
            # Mover modelo al dispositivo correcto si no se usa device_map
            if self.device != "cuda":
                self.base_model = self.base_model.to(self.device)
            
            logger.info(f"Modelo base cargado: {model_name} (LoRA r={self.lora_rank})")
            
        except Exception as e:
            logger.error(f"Error cargando modelo base: {str(e)}")
            raise

    def _load_reward_models(self, model_name: str) -> None:
        """Carga modelos de recompensa con manejo de dispositivo."""
        try:
            logger.info(f"Cargando modelo de recompensa: {model_name}")
            
            self.reward_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Mover modelo de recompensa al dispositivo
            self.reward_model = self.reward_model.to(self.device)
            
            # CrossEncoder opcional para recompensas más precisas
            if self.use_cross_encoder:
                logger.info("Cargando CrossEncoder para recompensas")
                self.cross_encoder = CrossEncoder(
                    "cross-encoder/stsb-roberta-large",
                    device=self.device
                )
            else:
                self.cross_encoder = None
                
            logger.info(f"Modelo de recompensa cargado: {model_name}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo de recompensa: {str(e)}")
            raise

    def prepare_dataset(
        self,
        feedback_file: str = "feedback.jsonl",
        max_samples: Optional[int] = None,
        validation_split: float = 0.1
    ) -> Tuple[Dataset, Dataset]:
        """
        Prepara dataset de entrenamiento desde feedback de usuarios con división train/val.
        
        Args:
            feedback_file: Ruta al archivo con feedback.
            max_samples: Límite opcional de muestras.
            validation_split: Porcentaje para validación (0-1).
            
        Returns:
            Tupla con (train_dataset, val_dataset)
        """
        try:
            with open(feedback_file, "r", encoding="utf-8") as f:
                samples = [json.loads(line) for line in f if line.strip()]
                
            if max_samples:
                samples = samples[:max_samples]
                
            # Convertir a formato Dataset
            full_dataset = Dataset.from_dict({
                "query": [sample["question"] for sample in samples],
                "response": [sample["answer"] for sample in samples],
                "rating": [sample["rating"] for sample in samples],
                "sources": [sample.get("sources", []) for sample in samples]
            })
            
            # Dividir en train/validation
            if validation_split > 0:
                split = full_dataset.train_test_split(test_size=validation_split, seed=42)
                return split["train"], split["test"]
            return full_dataset, None
            
        except Exception as e:
            logger.error(f"Error preparando dataset: {str(e)}")
            raise

    def compute_reward(
        self,
        query: str,
        response: str,
        sources: List[str],
        rating: Optional[int] = None
    ) -> float:
        """
        Calcula recompensa combinando múltiples modelos con manejo de errores.
        
        Args:
            query: Pregunta del usuario.
            response: Respuesta generada.
            sources: Documentos fuente.
            rating: Calificación humana (opcional).
            
        Returns:
            Puntuación de recompensa normalizada (0-1).
        """
        try:
            rewards = []
            
            # 1. Recompensa basada en rating humano (si existe)
            if rating is not None:
                norm_rating = rating / 5.0  # Normalizar a 0-1
                rewards.append(norm_rating * 2.0)  # Peso mayor
            
            # 2. Recompensa por consistencia con fuentes
            consistency_reward = self._compute_consistency_reward(response, sources)
            rewards.append(consistency_reward)
            
            # 3. Recompensa por calidad (modelo de recompensa)
            quality_reward = self._compute_quality_reward(query, response)
            rewards.append(quality_reward)
            
            # Promedio ponderado con manejo de casos edge
            if not rewards:
                return 0.5
                
            final_reward = np.mean(rewards)
            return float(np.clip(final_reward, 0, 1))  # Asegurar rango 0-1
            
        except Exception as e:
            logger.error(f"Error calculando recompensa: {str(e)}")
            return 0.5  # Valor por defecto seguro

    def _compute_consistency_reward(self, response: str, sources: List[str]) -> float:
        """Calcula recompensa por consistencia con documentos fuente."""
        if not sources:
            return 0.5
            
        try:
            if self.use_cross_encoder and self.cross_encoder:
                # Usar CrossEncoder para comparación semántica
                pairs = [(response, source) for source in sources]
                scores = self.cross_encoder.predict(pairs, convert_to_numpy=True)
                return float(np.mean(scores))
            else:
                # Método simple basado en overlap como fallback
                source_text = " ".join(sources)
                response_words = set(response.lower().split())
                source_words = set(source_text.lower().split())
                overlap = len(response_words & source_words) / max(1, len(response_words))
                return float(overlap)
        except Exception as e:
            logger.warning(f"Error en consistencia: {str(e)}")
            return 0.5

    def _compute_quality_reward(self, query: str, response: str) -> float:
        """Usa modelo de recompensa para evaluar calidad con manejo de errores."""
        try:
            inputs = self.reward_tokenizer(
                query,
                response,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.reward_model(**inputs)
                logits = outputs.logits
                probas = torch.softmax(logits, dim=1)
                
            # Suponiendo que la clase 1 es "alta calidad"
            return float(probas[0][1].item())
        except Exception as e:
            logger.warning(f"Error en calidad: {str(e)}")
            return 0.5

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        epochs: int = 3,
        batch_size: int = 8,
        save_dir: str = "./rlhf_model",
        checkpoint_steps: int = 500
    ) -> None:
        """
        Ejecuta entrenamiento RLHF con PPO y manejo de checkpoints.
        
        Args:
            train_dataset: Dataset de entrenamiento.
            val_dataset: Dataset de validación (opcional).
            epochs: Número de épocas.
            batch_size: Tamaño de lote.
            save_dir: Directorio para guardar el modelo.
            checkpoint_steps: Guardar checkpoint cada N pasos.
        """
        try:
            # Configurar TrainingArguments
            training_args = TrainingArguments(
                output_dir=save_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size if val_dataset else None,
                save_steps=checkpoint_steps,
                save_strategy="steps",
                logging_dir="./logs",
                logging_steps=10,
                evaluation_strategy="steps" if val_dataset else "no",
                eval_steps=checkpoint_steps,
                report_to="tensorboard",
                gradient_accumulation_steps=1,
                fp16=self.mixed_precision,
                remove_unused_columns=False,
                seed=42
            )
            
            # Modelo con cabeza de valor para PPO
            model_with_value_head = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if self.mixed_precision else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Inicializar PPO Trainer
            ppo_trainer = PPOTrainer(
                model=model_with_value_head,
                config=self.ppo_config,
                tokenizer=self.tokenizer,
                dataset=train_dataset,
                args=training_args
            )
            
            logger.info("Iniciando entrenamiento RLHF...")
            
            # Bucle de entrenamiento
            for epoch in range(epochs):
                for batch in ppo_trainer.dataloader:
                    queries = batch["query"]
                    responses = batch["response"]
                    sources = batch["sources"]
                    ratings = batch["rating"]
                    
                    # Tokenizar
                    inputs = self.tokenizer(
                        queries,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=256
                    ).to(self.device)
                    
                    # Generar respuestas
                    outputs = model_with_value_head.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    generated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    
                    # Calcular recompensas
                    rewards = [
                        self.compute_reward(q, r, s, rt)
                        for q, r, s, rt in zip(queries, generated, sources, ratings)
                    ]
                    rewards = torch.tensor(rewards).to(self.device)
                    
                    # Paso PPO
                    stats = ppo_trainer.step(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        responses=outputs,
                        rewards=rewards
                    )
                    
                    logger.info(
                        f"Época {epoch+1}/{epochs} - "
                        f"Recompensa media: {stats['ppo/returns/mean']:.3f} - "
                        f"Loss: {stats['ppo/loss/total']:.3f}"
                    )
            
            # Guardar modelo final
            ppo_trainer.save_model(save_dir)
            self.tokenizer.save_pretrained(save_dir)
            logger.info(f"Modelo guardado en {save_dir}")
            
        except Exception as e:
            logger.error(f"Error en entrenamiento RLHF: {str(e)}")
            raise

    def evaluate(
        self,
        evaluator: RAGEvaluator,
        test_dataset: Dataset,
        num_samples: int = 20
    ) -> Dict[str, float]:
        """
        Evalúa el modelo fine-tuned con métricas completas.
        
        Args:
            evaluator: Instancia de RAGEvaluator.
            test_dataset: Dataset de evaluación.
            num_samples: Número de muestras a evaluar.
            
        Returns:
            Dict con métricas de evaluación.
        """
        metrics = {
            "relevance": [],
            "consistency": [],
            "quality": [],
            "reward": []
        }

        try:
            samples = test_dataset.shuffle().select(range(num_samples))
            
            for sample in samples:
                # Generar respuesta
                inputs = self.tokenizer(
                    sample["query"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(self.device)
                
                output = self.base_model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.7
                )
                response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Evaluar con RAGEvaluator
                eval_result = evaluator.full_evaluation(
                    question=sample["query"],
                    documents=sample["sources"],
                    answer=response
                )
                
                # Calcular recompensa
                reward = self.compute_reward(
                    sample["query"],
                    response,
                    sample["sources"]
                )
                
                # Recolectar métricas
                metrics["relevance"].append(
                    sum(1 for r in eval_result["relevance"].values() if r["binary_score"] == "yes") / 
                    max(1, len(eval_result["relevance"]))
                )
                metrics["consistency"].append(
                    1.0 if eval_result["hallucination"]["binary_score"] == "yes" else 0.0
                )
                metrics["quality"].append(eval_result["quality"]["score"] / 5.0)
                metrics["reward"].append(reward)
            
            # Calcular promedios y desviaciones estándar
            return {
                "relevance_score": np.mean(metrics["relevance"]),
                "relevance_std": np.std(metrics["relevance"]),
                "consistency_score": np.mean(metrics["consistency"]),
                "consistency_std": np.std(metrics["consistency"]),
                "quality_score": np.mean(metrics["quality"]),
                "quality_std": np.std(metrics["quality"]),
                "reward_score": np.mean(metrics["reward"]),
                "reward_std": np.std(metrics["reward"]),
                "num_samples": num_samples
            }
        except Exception as e:
            logger.error(f"Error en evaluación: {str(e)}")
            return {
                "relevance_score": 0.0,
                "consistency_score": 0.0,
                "quality_score": 0.0,
                "reward_score": 0.0,
                "error": str(e)
            }


    @classmethod
    def from_feedback_file(
        cls,
        feedback_file: str,
        base_model: str = "google/flan-t5-large",
        reward_model: str = "facebook/roberta-hate-speech-dynabench-r4",
        save_dir: str = "./rlhf_model",
        epochs: int = 3,
        **kwargs
    ) -> "RLHFTrainer":
        """
        Método conveniente para entrenar directamente desde archivo de feedback.
        
        Args:
            feedback_file: Ruta al archivo JSONL con feedback.
            base_model: Nombre del modelo base.
            reward_model: Nombre del modelo de recompensa.
            save_dir: Directorio para guardar modelo entrenado.
            epochs: Número de épocas.
            **kwargs: Argumentos adicionales para RLHFTrainer.
            
        Returns:
            Instancia de RLHFTrainer con modelo entrenado.
        """
        trainer = cls(
            base_model_name=base_model,
            reward_model_name=reward_model,
            **kwargs
        )
        train_dataset, val_dataset = trainer.prepare_dataset(feedback_file)
        trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=epochs,
            save_dir=save_dir
        )
        return trainer
