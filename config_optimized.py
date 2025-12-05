# config_optimized.py
"""Configuración optimizada para sistema ML local."""

from pathlib import Path

# Directorios
RAW_DIR = Path('./data/raw')
PROC_DIR = Path('./data/processed')
VECTOR_DIR = Path('./data/vector')

# Configuración ML optimizada
ML_ENABLED = True
ML_FEATURES = ['category', 'entities', 'embedding']  # Tags removido temporalmente para velocidad
ML_CATEGORIES = [
    'Electronics','Home & Kitchen','Clothing & Accessories','Books & Media','Sports & Outdoors',
    'Health & Beauty','Automotive','Office Supplies','Toys & Games','Other'
]

# Modelos optimizados
ML_MODELS = {
    'embedding': 'all-MiniLM-L6-v2',
    'zero_shot': 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',
    'ner': 'Davlan/bert-base-multilingual-cased-ner-hrl',
}

# Performance
ML_BATCH_SIZE = 8
ML_MAX_MEMORY_MB = 1024
ML_USE_CACHE = True
ML_CACHE_DIR = Path.home() / '.cache' / 'proyecto_ml'

# DataLoader optimizado
LOADER_MAX_PRODUCTS = 200
LOADER_USE_PROGRESS = True
LOADER_AUTO_CATEGORIES = True

# Configuración local
LOCAL_LLM_ENABLED = False
LOCAL_LLM_MODEL = 'llama-3.2-3b-instruct'
DEVICE = 'cpu'
