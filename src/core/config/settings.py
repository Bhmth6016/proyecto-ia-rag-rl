#src/core/config/settings.py
"""
Central configuration.
Load once, use everywhere.
"""
from __future__ import annotations

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # .env file is optional

# ------------------------------------------------------------------
# Base directories (auto-resolve)
# ------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent.parent.parent.resolve()
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
VEC_DIR    = DATA_DIR / "vector"          # Chroma / FAISS
LOG_DIR    = DATA_DIR / "logs"
DEVICE = os.getenv("DEVICE", "cpu")

for d in (DATA_DIR, RAW_DIR, PROC_DIR, VEC_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Models & hyper-params
# ------------------------------------------------------------------
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
BASE_LLM         = os.getenv("BASE_LLM", "google/flan-t5-base")
EVAL_LLM         = os.getenv("EVAL_LLM", "google/flan-t5-large")

# ------------------------------------------------------------------
# Vector store
# ------------------------------------------------------------------
VECTOR_BACKEND   = os.getenv("VECTOR_BACKEND", "chroma")  # "chroma" | "faiss"
INDEX_NAME       = "products"

# ------------------------------------------------------------------
# Cache / RLHF
# ------------------------------------------------------------------
CACHE_ENABLED    = os.getenv("CACHE_ENABLED", "true").lower() in {"true", "1", "yes"}
RLHF_CHECKPOINT  = os.getenv("RLHF_CHECKPOINT")  # Path or None

_RLHF_YAML = BASE_DIR / "rlhf_config.yaml"
RLHF_CONFIG: dict = {}
if _RLHF_YAML.exists():
    with _RLHF_YAML.open() as f:
        RLHF_CONFIG = yaml.safe_load(f)
else:
    # Defaults inline para que nunca falle
    RLHF_CONFIG = {
        "base_model": "google/flan-t5-large",
        "reward_model": "facebook/roberta-hate-speech-dynabench-r4",
        "device": "cuda",
        "lora_rank": 8,
        "batch_size": 8,
        "learning_rate": 1.41e-5,
        "epochs": 3,
    }