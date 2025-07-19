import os
from pathlib import Path
from dotenv import load_dotenv
import torch 

# Load environment variables from .env file
load_dotenv()

# GEMINI API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBnXA2lIP6xfyMICg77XctxmninUOdrzLQ")

# ChromaDB Configuration
CHROMA_DB_COLLECTION = os.getenv("CHROMA_DB_COLLECTION", "amazon_products")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "./logs/amazon_recommendations.log")

# System Limits
MAX_PRODUCTS_TO_LOAD = int(os.getenv("MAX_PRODUCTS_TO_LOAD", 1000000))
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", 20000))
MAX_QUERY_RESULTS = int(os.getenv("MAX_QUERY_RESULTS", 5))

# Data Paths
BASE_DIR = Path.cwd().resolve()
DATA_DIR = BASE_DIR / "data"  # Ensure this matches your structure
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
VEC_DIR = DATA_DIR / "vector"  # Chroma
LOG_DIR = BASE_DIR / "logs"
# Vector Store Configuration
VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", str(DATA_DIR / "processed" / "chroma_db"))
CHROMA_DB_PATH = VECTOR_INDEX_PATH
VECTOR_BACKEND = "chroma"  # Explicitly set to chroma
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Ensure directories exist
for d in (DATA_DIR, RAW_DIR, PROC_DIR, VEC_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Cache / RLHF
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() in {"true", "1", "yes"}
RLHF_CHECKPOINT = os.getenv("RLHF_CHECKPOINT")

# Telemetry
ANONYMIZED_TELEMETRY = os.getenv("ANONYMIZED_TELEMETRY", "false").lower() in {"true", "1", "yes"}

CHROMA_SETTINGS = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 200,
    "hnsw:search_ef": 100,
    "hnsw:M": 16,  # Valor más bajo para mayor eficiencia con muchos documentos
}

BUILD_INDEX_IF_MISSING = os.getenv("BUILD_INDEX_IF_MISSING", "false").lower() == "true"