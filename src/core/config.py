import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# Load environment variables from .env file
load_dotenv()

# ============================================================
# API / MODEL CONFIG
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBnXA2lIP6xfyMICg77XctxmninUOdrzLQAIzaSyBnXA2lIP6xfyMICg77XctxmninUOdrzLQ")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")

# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "./logs/amazon_recommendations.log")

# ============================================================
# DATA PATHS
# ============================================================

BASE_DIR = Path.cwd().resolve()

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
LOG_DIR = BASE_DIR / "logs"
MODELS_DIR = DATA_DIR / "models"

FEEDBACK_DIR = BASE_DIR / "data/feedback"
HISTORIAL_DIR = BASE_DIR / "data/processed/historial"

# Ensure directories exist
for d in (DATA_DIR, RAW_DIR, PROC_DIR, LOG_DIR, MODELS_DIR, FEEDBACK_DIR, HISTORIAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# SYSTEM LIMITS
# ============================================================

MAX_PRODUCTS_TO_LOAD = int(os.getenv("MAX_PRODUCTS_TO_LOAD", 1_000_000))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", 20_000))
MAX_QUERY_RESULTS = int(os.getenv("MAX_QUERY_RESULTS", 5))

# ============================================================
# VECTOR STORE / EMBEDDINGS
# ============================================================

CHROMA_DB_COLLECTION = os.getenv("CHROMA_DB_COLLECTION", "amazon_products")

VECTOR_INDEX_PATH = os.getenv(
    "VECTOR_INDEX_PATH",
    str(DATA_DIR / "processed" / "chroma_db")
)

CHROMA_DB_PATH = VECTOR_INDEX_PATH
VECTOR_BACKEND = "chroma"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

CHROMA_SETTINGS = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 200,
    "hnsw:search_ef": 100,
    "hnsw:M": 16
}

BUILD_INDEX_IF_MISSING = os.getenv("BUILD_INDEX_IF_MISSING", "true").lower() == "true"

# ============================================================
# RL / RLHF CONFIGURATION
# ============================================================

# Mantiene un ritmo ideal de aprendizaje sin sobrecargar memoria
RL_MIN_SAMPLES = int(os.getenv("RL_MIN_SAMPLES", 10))

# Tamaño del lote del buffer RLHF
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 500))

# Cache para acelerar recomputación
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() in {"true", "1", "yes"}

# Checkpoint del agente de RL
RLHF_CHECKPOINT = os.getenv("RLHF_CHECKPOINT")

# ============================================================
# TELEMETRY
# ============================================================

ANONYMIZED_TELEMETRY = os.getenv("ANONYMIZED_TELEMETRY", "false").lower() in {
    "true", "1", "yes"
}
class Settings:
    pass

settings = Settings()

# Copiar TODAS las variables globales al objeto settings
globals_copy = dict(globals())   # ← COPIA SEGURA DEL DICCIONARIO

for key, value in globals_copy.items():
    if key.isupper():           # solo variables CONSTANTES
        setattr(settings, key, value)
