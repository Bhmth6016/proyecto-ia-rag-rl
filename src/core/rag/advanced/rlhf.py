from __future__ import annotations
# src/core/rag/advanced/rlhf.py

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    DataCollatorForSeq2Seq  # Note the name change
)
from src.core.config import settings
from src.core.category_search.category_tree import ProductFilter
from src.core.rag.advanced.evaluator import RAGEvaluator, load_evaluator_llm
from src.core.utils.logger import get_logger

logger = get_logger(__name__)


class RLHFTrainer:
    """
    Fine-tune any encoder-decoder model (T5 family) via PPO using human feedback
    collected by `FeedbackProcessor`.
    """

    def __init__(
        self,
        *,
        base_model_name: str | None = None,
        reward_model_name: str | None = None,
        device: str | None = None,
        use_cross_encoder: bool = False,
        cross_encoder_name: str = "cross-encoder/stsb-roberta-base",
        **kwargs,
    ):
        cfg = settings.RLHF_CONFIG
        self.base_model_name = base_model_name or cfg["base_model"]
        self.reward_model_name = reward_model_name or cfg["reward_model"]
        self.device = device or cfg["device"]
        self.lora_rank = kwargs.pop("lora_rank", cfg["lora_rank"])
        self.batch_size = kwargs.pop("batch_size", cfg["batch_size"])
        self.learning_rate = kwargs.pop("learning_rate", cfg["learning_rate"])
        self.epochs = kwargs.pop("epochs", cfg["epochs"])
        self.mixed_precision = torch.cuda.is_available()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            model_max_length=512,
            legacy=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if self.mixed_precision else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.lora_rank,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
        )
        self.base_model = get_peft_model(base_model, lora_cfg)
        self.base_model.print_trainable_parameters()

        self.reward_tok = AutoTokenizer.from_pretrained(self.reward_model_name)
        self.reward_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.reward_model_name,
            torch_dtype=torch.float16 if self.mixed_precision else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )

        self.cross_encoder = (
            CrossEncoder(cross_encoder_name, device=self.device)
            if use_cross_encoder else None
        )

        self.evaluator = RAGEvaluator(load_evaluator_llm())

        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

        logger.info("RLHFTrainer initialised on %s", self.device)

    def prepare_rlhf_dataset(self, feedback_dir: Path, min_samples: int = 1000):
        """Carga y procesa los feedbacks para RLHF"""
        feedback_files = list(feedback_dir.glob("*.jsonl"))
        
        # Validar que hay suficientes muestras
        samples = []
        for file in feedback_files:
            with open(file) as f:
                samples.extend([json.loads(line) for line in f])
        
        if len(samples) < min_samples:
            raise ValueError(f"Se requieren al menos {min_samples} muestras, solo hay {len(samples)}")
        
        # Procesar cada muestra
        dataset = []
        for sample in samples:
            reward_data = self.compute_reward(
                query=sample["query_es"],
                response=sample["answer_en"],
                rating=sample["rating"],
                source_lang=sample["lang"],
                translated_query=sample["query_en"]
            )
            
            dataset.append({
                "prompt": sample["query_en"],
                "response": sample["answer_en"],
                "reward": reward_data["reward"],
                "metadata": {
                    "original_query": sample["query_es"],
                    "lang": sample["lang"],
                    "domain_terms": sample.get("domain_terms", {})
                }
            })
        
        return dataset

    def compute_reward(
        self,
        query: str,
        response: str,
        rating: int,
        source_lang: str,
        translated_query: str
    ) -> Dict[str, float]:
        """Cálculo de recompensa mejorado con múltiples componentes"""
        
        # 1. Componente humano (normalizado y suavizado)
        human_norm = self._normalize_human_feedback(rating)
        
        # 2. Métricas automáticas
        auto_metrics = {
            'consistency': self._check_consistency(response, translated_query),
            'translation_quality': self._eval_translation_quality(query, response, source_lang),
            'domain_term_presence': self._check_domain_terms(response, source_lang),
            'response_quality': self._evaluate_response_quality(response)
        }
        
        # 3. Pesos dinámicos
        weights = self._calculate_dynamic_weights(auto_metrics, human_norm)
        
        # 4. Recompensa final
        reward = sum(
            weights[component] * auto_metrics[component]
            for component in auto_metrics
        ) + weights['human'] * human_norm
        
        return {
            'reward': float(np.clip(reward, 0.0, 1.0)),
            'weights': weights,
            'components': {
                'human': human_norm,
                **auto_metrics
            }
        }
    
    def _normalize_human_feedback(self, rating: int) -> float:
        """Normaliza el rating 1-5 a rango 0-1 con suavizado"""
        rating = max(1, min(5, rating))  # Asegurar rango
        return (rating - 1) / 4.0  # 1→0.0, 5→1.0
    
    def _check_consistency(self, response: str, query: str) -> float:
        """Evalúa consistencia semántica entre pregunta y respuesta"""
        # Embeddings de pregunta y respuesta
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        response_embedding = self.embedder.encode(response, convert_to_tensor=True)
        
        # Similitud coseno
        cos_sim = torch.nn.functional.cosine_similarity(
            query_embedding, response_embedding, dim=0
        )
        return float(cos_sim.item())
    
    def _eval_translation_quality(self, original_query: str, response_en: str, source_lang: str) -> float:
        """Evalúa preservación de términos clave en la respuesta"""
        if source_lang == 'en':
            return 1.0  # No aplica para inglés original
        
        # Extraer términos de dominio del query original
        domain_terms = set(self.translator.DOMAIN_TERMS.keys())
        found_terms = [term for term in domain_terms if term in original_query.lower()]
        
        if not found_terms:
            return 0.8  # Valor base cuando no hay términos específicos
        
        # Verificar términos en respuesta
        preserved = 0
        for term in found_terms:
            en_term = self.translator.DOMAIN_TERMS[term]["en"]
            if en_term.lower() in response_en.lower():
                preserved += 1
        
        return preserved / len(found_terms)
    
    def _calculate_dynamic_weights(self, auto_metrics: Dict, human_score: float) -> Dict:
        """Calcula pesos dinámicos basados en calidad de señales"""
        base_weights = {
            'human': 0.6,
            'consistency': 0.2,
            'translation_quality': 0.1,
            'domain_term_presence': 0.05,
            'response_quality': 0.05
        }
        
        # Ajustar pesos basado en confianza
        if human_score < 0.3:  # Feedback humano poco confiable
            base_weights['human'] *= 0.5
            base_weights['consistency'] *= 1.5
        
        # Re-normalizar
        total = sum(base_weights.values())
        return {k: v/total for k, v in base_weights.items()}

    def build_dataset(
        self,
        feedback_dir: Path,
        min_rating: int = 1,
        max_samples: Optional[int] = None,
        val_ratio: float = 0.1,
    ) -> Tuple[Dataset, Dataset]:
        """
        Load feedback JSONL files created by `FeedbackProcessor` and
        return HF datasets ready for PPO.
        """
        all_records = []
        for fp in sorted(feedback_dir.glob("feedback_*.jsonl")):
            with fp.open() as f:
                all_records.extend([json.loads(l) for l in f if l.strip()])

        all_records = [r for r in all_records if r.get("rating", 0) >= min_rating]
        if max_samples:
            all_records = all_records[:max_samples]

        if not all_records:
            raise ValueError("No feedback records satisfy the filters.")

        raw = Dataset.from_list(all_records)
        split = raw.train_test_split(test_size=val_ratio, seed=42)
        return split["train"], split["test"]

    def train(
        self,
        *,
        feedback_dir: Path,
        save_dir: Path,
    ):
        """
        Train with PPO using feedback dataset.
        """
        train_set, val_set = self.build_dataset(feedback_dir)

        ppo_config = PPOConfig(
            model_name=self.base_model_name,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            log_with=None,
            mini_batch_size=4,
            ppo_epochs=self.epochs,
            init_kl_coef=0.1,
            adap_kl_ctrl=True,
        )

        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.base_model,
            tokenizer=self.tokenizer,
            dataset=train_set,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, pad_to_multiple_of=8),
        )

        logger.info("Starting PPO training...")
        ppo_trainer.train()
        logger.info("Training complete.")

        logger.info("Saving LoRA adapters and tokenizer...")
        self.base_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)