# src/core/rag/advanced/rlhf.py

"""
RLHF fine-tuning orchestrator.

High-level flow
---------------
1. Pulls **context-rich feedback** produced by `FeedbackProcessor`
2. Builds reward signals using:
   - Human rating  
   - Consistency with retrieved docs (CrossEncoder or category filter)  
   - Offline RAG evaluator scores
3. Runs PPO (TRL) with LoRA adapters on the base `flan-t5-*` model.
4. Saves the LoRA checkpoint + tokenizer for the RAG stack.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    DataCollatorForSeq2SeqLM,
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from sentence_transformers import CrossEncoder

from src.core.config import settings
from src.core.category_search.category_tree import ProductFilter
from src.core.rag.advanced.evaluators import RAGEvaluator, load_evaluator_llm
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

        logger.info("RLHFTrainer initialised on %s", self.device)

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

    def compute_reward(
        self,
        query: str,
        response: str,
        sources: List[str],
        record: Dict[str, Any],
    ) -> float:
        """
        Multi-component reward:
        R = w1*R_human + w2*R_eval + w3*R_consistency
        """
        rewards = []

        rating = int(record.get("rating", 3))
        rewards.append(rating / 5.0)

        eval_scores = record.get("eval_scores", {})
        if eval_scores:
            rewards.append(eval_scores.get("quality", {}).get("score", 3.0) / 5.0)
        else:
            rewards.append(0.5)

        if self.cross_encoder and sources:
            scores = self.cross_encoder.predict(
                [(response, doc) for doc in sources]
            )
            rewards.append(float(np.mean(scores)))
        else:
            rewards.append(0.5)

        return float(np.clip(np.mean(rewards), 0.0, 1.0))

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
            data_collator=DataCollatorForSeq2SeqLM(self.tokenizer, pad_to_multiple_of=8),
        )

        logger.info("Starting PPO training...")
        ppo_trainer.train()
        logger.info("Training complete.")

        logger.info("Saving LoRA adapters and tokenizer...")
        self.base_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
