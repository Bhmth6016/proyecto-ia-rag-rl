# src/core/config/settings.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# GEMINI API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBnXA2lIP6xfyMICg77XctxmninUOdrzLQ")

# Configuración de ChromaDB
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/processed/chroma_db")
CHROMA_DB_COLLECTION = os.getenv("CHROMA_DB_COLLECTION", "amazon_products")

# Configuración de logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "./logs/amazon_recommendations.log")

# Device configuration
DEVICE = os.getenv("DEVICE", "cpu")

# Límites del sistema
MAX_PRODUCTS_TO_LOAD = int(os.getenv("MAX_PRODUCTS_TO_LOAD", 1000000))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", 20000))
MAX_QUERY_RESULTS = int(os.getenv("MAX_QUERY_RESULTS", 5))

# Vector Store Configuration
VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", "./data/processed/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Rutas de datos
BASE_DIR = Path.cwd().resolve()
DATA_DIR = BASE_DIR / "data"  # Ensure this matches your structure
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
VEC_DIR = DATA_DIR / "vector"  # Chroma / FAISS
LOG_DIR = DATA_DIR / "logs"

# Ensure directories exist
for d in (DATA_DIR, RAW_DIR, PROC_DIR, VEC_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Vector Store
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma")
INDEX_NAME = "products"

# Cache / RLHF
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() in {"true", "1", "yes"}
RLHF_CHECKPOINT = os.getenv("RLHF_CHECKPOINT")

# Telemetry
ANONYMIZED_TELEMETRY = os.getenv("ANONYMIZED_TELEMETRY", "false").lower() in {"true", "1", "yes"}