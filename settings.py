# settings.py

# GEMINI API Key
GEMINI_API_KEY = "AIzaSyBnXA2lIP6xfyMICg77XctxmninUOdrzLQ"

# Configuración de ChromaDB
CHROMA_DB_PATH = "./data/processed/chroma_db"
CHROMA_DB_COLLECTION = "amazon_products"

# Configuración de logging
LOG_LEVEL = "INFO"
LOG_FILE = "./logs/amazon_recommendations.log"

# Límites del sistema
MAX_PRODUCTS_TO_LOAD = 1000000
MAX_QUERY_LENGTH = 20000

# Vector Store Configuration
VECTOR_INDEX_PATH = "./data/processed/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Rutas de datos
DATA_DIR = "./data/raw"
MAX_QUERY_RESULTS = 5

ANONYMIZED_TELEMETRY = False