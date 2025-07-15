# src/core/llms/local_llm.py
"""
Factory for local LLMs with optional LoRA adapters.
Supports both base models and RLHF-tuned variants.
"""

from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    GenerationConfig
)
from langchain_huggingface import HuggingFacePipeline
from peft import PeftModel

from src.core.config import settings
from src.core.rag.advanced.prompts import RAG_PROMPT_TEMPLATE

def local_llm(
    base_model_name: str = settings.BASE_LLM,
    lora_checkpoint: Optional[Path] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> HuggingFacePipeline:
    """
    Load a local LLM with optional LoRA adapters.
    
    Args:
        base_model_name: HF model ID or path
        lora_checkpoint: Path to LoRA adapter (optional)
        device: Device to load model on
        max_new_tokens: Generation length
        temperature: Sampling temperature
        
    Returns:
        Initialized HuggingFacePipeline
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        model_max_length=max_new_tokens,
        use_fast=True,
    )
    
    # Load base model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        device_map="auto" if "cuda" in device else None,
    )
    
    # Apply LoRA if specified
    if lora_checkpoint and lora_checkpoint.exists():
        model = PeftModel.from_pretrained(
            model,
            lora_checkpoint,
            device_map="auto" if "cuda" in device else None,
        )
        model = model.merge_and_unload()
    
    # Valid generation configuration for seq2seq models
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0 else False,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Create text generation pipeline
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        device=device,
    )
    
    return HuggingFacePipeline(pipeline=pipe)