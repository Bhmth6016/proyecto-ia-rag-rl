# src/core/config.py
import os
from pathlib import Path
from dotenv import load_dotenv
import torch
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict, Any, Set
import logging
import warnings

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# ============================================================
# GLOBAL SETTINGS SINGLETON
# ============================================================

class GlobalSettings(BaseModel):
    """Configuración central del sistema con gestión unificada de ML 100% local."""
    
    # ======================
    # CONFIGURACIÓN LOCAL DE LLM
    # ======================
    LOCAL_LLM_ENABLED: bool = Field(
        default=True,
        description="Habilitar LLM local (Ollama)"
    )
    
    LOCAL_LLM_MODEL: str = Field(
        default="llama-3.2-3b-instruct",
        description="Modelo LLM local para Ollama"
    )
    
    LOCAL_LLM_ENDPOINT: str = Field(
        default="http://localhost:11434",
        description="Endpoint de Ollama"
    )
    
    LOCAL_LLM_TIMEOUT: int = Field(
        default=60,
        description="Timeout en segundos para llamadas al LLM local"
    )
    
    LOCAL_LLM_TEMPERATURE: float = Field(
        default=0.1,
        description="Temperatura para generación del LLM local"
    )
    
    # ======================
    # CONFIGURACIÓN DE EMBEDDINGS LOCALES
    # ======================
    LOCAL_EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Modelo de embeddings local (Sentence Transformers)"
    )
    
    LOCAL_EMBEDDING_DEVICE: str = Field(
        default="cpu",
        description="Dispositivo para embeddings (cpu/cuda)"
    )
    
    # ======================
    # LOGGING
    # ======================
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FILE: str = Field(default="./logs/amazon_recommendations.log", description="Log file path")
    
    # ======================
    # DATA PATHS
    # ======================
    BASE_DIR: Path = Field(default_factory=lambda: Path.cwd().resolve())
    DATA_DIR: Path = Field(default_factory=lambda: Path.cwd().resolve() / "data")
    RAW_DIR: Path = Field(default_factory=lambda: Path.cwd().resolve() / "data" / "raw")
    PROC_DIR: Path = Field(default_factory=lambda: Path.cwd().resolve() / "data" / "processed")
    LOG_DIR: Path = Field(default_factory=lambda: Path.cwd().resolve() / "logs")
    MODELS_DIR: Path = Field(default_factory=lambda: Path.cwd().resolve() / "data" / "models")
    FEEDBACK_DIR: Path = Field(default_factory=lambda: Path.cwd().resolve() / "data" / "feedback")
    HISTORIAL_DIR: Path = Field(default_factory=lambda: Path.cwd().resolve() / "data" / "processed" / "historial")
    LOCAL_MODELS_DIR: Path = Field(default_factory=lambda: Path.cwd().resolve() / "data" / "local_models")
    
    # ======================
    # SYSTEM LIMITS
    # ======================
    MAX_PRODUCTS_TO_LOAD: int = Field(default=1_000_000, description="Max products to load")
    MAX_QUERY_LENGTH: int = Field(default=20_000, description="Max query length")
    MAX_QUERY_RESULTS: int = Field(default=5, description="Max query results")
    
    # ======================
    # VECTOR STORE / EMBEDDINGS
    # ======================
    CHROMA_DB_COLLECTION: str = Field(default="amazon_products", description="Chroma collection name")
    VECTOR_INDEX_PATH: str = Field(default="data/processed/chroma_db", description="Vector index path")
    CHROMA_DB_PATH: str = Field(default="data/processed/chroma_db", description="Chroma DB path")
    VECTOR_BACKEND: str = Field(default="chroma", description="Vector backend")
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", description="Embedding model")
    DEVICE: str = Field(default="cpu", description="Device for computations")
    CHROMA_SETTINGS: Dict[str, Any] = Field(
        default_factory=lambda: {
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 200,
            "hnsw:search_ef": 100,
            "hnsw:M": 16
        }
    )
    BUILD_INDEX_IF_MISSING: bool = Field(default=True, description="Build index if missing")
    
    # ======================
    # RL / RLHF CONFIGURATION
    # ======================
    RL_MIN_SAMPLES: int = Field(default=10, description="Minimum RL samples")
    BATCH_SIZE: int = Field(default=500, description="Batch size")
    CACHE_ENABLED: bool = Field(default=True, description="Cache enabled")
    RLHF_CHECKPOINT: Optional[str] = Field(default=None, description="RLHF checkpoint path")
    
    # ======================
    # TELEMETRY
    # ======================
    ANONYMIZED_TELEMETRY: bool = Field(default=False, description="Anonymized telemetry")
    
    # ======================
    # ML CONFIGURATION (ÚNICA FUENTE DE VERDAD)
    # ======================
    ML_ENABLED: bool = Field(
        default=True,
        description="Master switch: Enable ALL ML features"
    )
    
    ML_FEATURES: Set[str] = Field(
        default_factory=lambda: {"category", "entities", "similarity"},
        description="ML features to enable (category, entities, tags, embedding, similarity)"
    )
    
    ML_EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model for semantic search"
    )
    
    ML_USE_GPU: bool = Field(
        default=False,
        description="Use GPU acceleration for ML models"
    )
    
    ML_CATEGORIES: List[str] = Field(
        default_factory=lambda: [
            "Electronics",
            "Home & Kitchen",
            "Clothing & Accessories",
            "Sports & Outdoors",
            "Books & Media",
            "Health & Beauty",
            "Toys & Games",
            "Automotive",
            "Office Products",
            "Other"
        ],
        description="Predefined categories for classification"
    )
    
    ML_CACHE_SIZE: int = Field(
        default=1000,
        description="Maximum number of embeddings to cache"
    )
    
    ML_CONFIDENCE_THRESHOLD: float = Field(
        default=0.6,
        description="Minimum confidence for ML predictions"
    )
    
    ML_WEIGHT: float = Field(
        default=0.3,
        description="Weight for ML scores in hybrid ranking"
    )
    
    ML_MIN_SIMILARITY: float = Field(
        default=0.7,
        description="Minimum similarity threshold for ML-based recommendations"
    )
    
    # 🔥 CORREGIDO: Sin guión bajo - usar propiedad privada de Python
    ml_propagated: bool = Field(default=False, description="Flag to avoid multiple propagation", exclude=True)
    
    # ======================
    # MÉTODOS DE CONVENIENCIA
    # ======================
    @model_validator(mode='after')
    def propagate_ml_settings(self):
        """Propaga configuración ML a todos los componentes del sistema."""
        if self.ml_propagated:
            return self
        
        try:
            # 🔥 PROPAGAR A Product
            from src.core.data.product import Product, AutoProductConfig
            
            # Actualizar configuración estática
            AutoProductConfig.ML_ENABLED = self.ML_ENABLED
            AutoProductConfig.DEFAULT_EMBEDDING_MODEL = self.ML_EMBEDDING_MODEL
            AutoProductConfig.DEFAULT_CATEGORIES = self.ML_CATEGORIES
            
            # Configurar ML en Product
            Product.configure_ml(
                enabled=self.ML_ENABLED,
                features=list(self.ML_FEATURES),
                categories=self.ML_CATEGORIES
            )
            
            logger.info(f"✅ ML settings propagated to Product system: enabled={self.ML_ENABLED}")
            self.ml_propagated = True
            
        except ImportError as e:
            logger.warning(f"Could not propagate ML settings: {e}")
        except Exception as e:
            logger.error(f"Error propagating ML settings: {e}")
        
        return self
    
    def update_ml_settings(self, 
                          ml_enabled: Optional[bool] = None,
                          ml_features: Optional[List[str]] = None,
                          ml_embedding_model: Optional[str] = None):
        """
        Actualiza configuración ML dinámicamente y propaga a todos los componentes.
        
        Args:
            ml_enabled: Habilitar/deshabilitar ML
            ml_features: Lista de features ML a habilitar
            ml_embedding_model: Modelo de embeddings a usar
        """
        # Actualizar valores
        if ml_enabled is not None:
            self.ML_ENABLED = ml_enabled
        
        if ml_features is not None:
            # Convertir 'all' a lista completa
            if ml_features == ['all']:
                self.ML_FEATURES = {"category", "entities", "tags", "embedding", "similarity"}
            else:
                self.ML_FEATURES = set(ml_features)
        
        if ml_embedding_model is not None:
            self.ML_EMBEDDING_MODEL = ml_embedding_model
        
        # Resetear flag de propagación
        self.ml_propagated = False
        
        # Reprogramar propagación
        self.propagate_ml_settings()
        
        logger.info(f"📡 ML settings updated: enabled={self.ML_ENABLED}, features={list(self.ML_FEATURES)}")
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Obtiene configuración ML actual."""
        return {
            'ML_ENABLED': self.ML_ENABLED,
            'ML_FEATURES': list(self.ML_FEATURES),
            'ML_EMBEDDING_MODEL': self.ML_EMBEDDING_MODEL,
            'ML_USE_GPU': self.ML_USE_GPU,
            'ML_CATEGORIES': self.ML_CATEGORIES,
            'ML_CACHE_SIZE': self.ML_CACHE_SIZE,
            'ML_CONFIDENCE_THRESHOLD': self.ML_CONFIDENCE_THRESHOLD,
            'ML_WEIGHT': self.ML_WEIGHT,
            'ML_MIN_SIMILARITY': self.ML_MIN_SIMILARITY
        }
    
    def get_local_llm_config(self) -> Dict[str, Any]:
        """Obtiene configuración de LLM local."""
        return {
            'LOCAL_LLM_ENABLED': self.LOCAL_LLM_ENABLED,
            'LOCAL_LLM_MODEL': self.LOCAL_LLM_MODEL,
            'LOCAL_LLM_ENDPOINT': self.LOCAL_LLM_ENDPOINT,
            'LOCAL_LLM_TIMEOUT': self.LOCAL_LLM_TIMEOUT,
            'LOCAL_LLM_TEMPERATURE': self.LOCAL_LLM_TEMPERATURE,
            'LOCAL_EMBEDDING_MODEL': self.LOCAL_EMBEDDING_MODEL,
            'LOCAL_EMBEDDING_DEVICE': self.LOCAL_EMBEDDING_DEVICE
        }
    
    def is_ml_feature_enabled(self, feature: str) -> bool:
        """Verifica si una feature ML específica está habilitada."""
        return feature in self.ML_FEATURES
    
    def is_local_llm_enabled(self) -> bool:
        """Verifica si el LLM local está habilitado."""
        return self.LOCAL_LLM_ENABLED
    
    def update_from_env(self):
        """Actualiza configuración desde variables de entorno."""
        # LOGGING
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", self.LOG_LEVEL)
        
        # SYSTEM LIMITS
        self.MAX_PRODUCTS_TO_LOAD = int(os.getenv("MAX_PRODUCTS_TO_LOAD", self.MAX_PRODUCTS_TO_LOAD))
        self.MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", self.MAX_QUERY_LENGTH))
        self.MAX_QUERY_RESULTS = int(os.getenv("MAX_QUERY_RESULTS", self.MAX_QUERY_RESULTS))
        
        # VECTOR STORE
        self.VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", self.VECTOR_INDEX_PATH)
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", self.EMBEDDING_MODEL)
        self.DEVICE = os.getenv("DEVICE", self.DEVICE)
        
        # RL
        self.RL_MIN_SAMPLES = int(os.getenv("RL_MIN_SAMPLES", self.RL_MIN_SAMPLES))
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", self.BATCH_SIZE))
        
        # TELEMETRY
        self.ANONYMIZED_TELEMETRY = os.getenv(
            "ANONYMIZED_TELEMETRY", 
            str(self.ANONYMIZED_TELEMETRY)
        ).lower() in {"true", "1", "yes"}
        
        # 🔥 CONFIGURACIÓN LOCAL LLM
        local_llm_env = os.getenv("LOCAL_LLM_ENABLED", "")
        if local_llm_env.lower() in {"true", "1", "yes"}:
            self.LOCAL_LLM_ENABLED = True
        elif local_llm_env.lower() in {"false", "0", "no"}:
            self.LOCAL_LLM_ENABLED = False
        
        self.LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", self.LOCAL_LLM_MODEL)
        self.LOCAL_LLM_ENDPOINT = os.getenv("LOCAL_LLM_ENDPOINT", self.LOCAL_LLM_ENDPOINT)
        self.LOCAL_LLM_TIMEOUT = int(os.getenv("LOCAL_LLM_TIMEOUT", self.LOCAL_LLM_TIMEOUT))
        
        temp_env = os.getenv("LOCAL_LLM_TEMPERATURE")
        if temp_env:
            self.LOCAL_LLM_TEMPERATURE = float(temp_env)
        
        self.LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", self.LOCAL_EMBEDDING_MODEL)
        self.LOCAL_EMBEDDING_DEVICE = os.getenv("LOCAL_EMBEDDING_DEVICE", self.LOCAL_EMBEDDING_DEVICE)
        
        # 🔥 CONFIGURACIÓN ML
        ml_enabled_env = os.getenv("ML_ENABLED", "")
        if ml_enabled_env.lower() in {"true", "1", "yes"}:
            self.ML_ENABLED = True
        elif ml_enabled_env.lower() in {"false", "0", "no"}:
            self.ML_ENABLED = False
        
        ml_features_env = os.getenv("ML_FEATURES", "")
        if ml_features_env:
            self.ML_FEATURES = set(f.strip() for f in ml_features_env.split(","))
        
        ml_model_env = os.getenv("ML_EMBEDDING_MODEL", "")
        if ml_model_env:
            self.ML_EMBEDDING_MODEL = ml_model_env
        
        # Propagar cambios de ML
        self.ml_propagated = False
        self.propagate_ml_settings()
    
    def __str__(self) -> str:
        """Representación legible de la configuración."""
        return (
            f"Settings(LOCAL_LLM_ENABLED={self.LOCAL_LLM_ENABLED}, "
            f"ML_ENABLED={self.ML_ENABLED}, "
            f"ML_FEATURES={list(self.ML_FEATURES)}, "
            f"EMBEDDING_MODEL={self.EMBEDDING_MODEL})"
        )
    
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True


# ============================================================
# INSTANCIA GLOBAL SINGLETON
# ============================================================

# Crear instancia global
settings = GlobalSettings()

# Actualizar desde entorno
settings.update_from_env()

# Imprimir configuración cargada
print(f"✅ Configuración LOCAL cargada:")
print(f"   - LOCAL_LLM_ENABLED: {settings.LOCAL_LLM_ENABLED}")
print(f"   - LOCAL_LLM_MODEL: {settings.LOCAL_LLM_MODEL}")
print(f"   - VECTOR_INDEX_PATH: {settings.VECTOR_INDEX_PATH}")
print(f"   - DEVICE: {settings.DEVICE}")
print(f"   - ML_ENABLED: {settings.ML_ENABLED}")
print(f"   - ML_FEATURES: {list(settings.ML_FEATURES)}")

# Asegurar que la propagación se ejecute
if settings.ML_ENABLED:
    settings.propagate_ml_settings()

# ============================================================
# CONSTANTES PARA COMPATIBILIDAD
# ============================================================

# Constantes para LLM local
LOCAL_LLM_ENABLED = settings.LOCAL_LLM_ENABLED
LOCAL_LLM_MODEL = settings.LOCAL_LLM_MODEL
LOCAL_LLM_ENDPOINT = settings.LOCAL_LLM_ENDPOINT
LOCAL_LLM_TIMEOUT = settings.LOCAL_LLM_TIMEOUT
LOCAL_LLM_TEMPERATURE = settings.LOCAL_LLM_TEMPERATURE
LOCAL_EMBEDDING_MODEL = settings.LOCAL_EMBEDDING_MODEL
LOCAL_EMBEDDING_DEVICE = settings.LOCAL_EMBEDDING_DEVICE

# Constantes originales para compatibilidad
LOG_LEVEL = settings.LOG_LEVEL
LOG_FILE = settings.LOG_FILE

BASE_DIR = settings.BASE_DIR
DATA_DIR = settings.DATA_DIR
RAW_DIR = settings.RAW_DIR
PROC_DIR = settings.PROC_DIR
LOG_DIR = settings.LOG_DIR
MODELS_DIR = settings.MODELS_DIR
FEEDBACK_DIR = settings.FEEDBACK_DIR
HISTORIAL_DIR = settings.HISTORIAL_DIR
LOCAL_MODELS_DIR = settings.LOCAL_MODELS_DIR

MAX_PRODUCTS_TO_LOAD = settings.MAX_PRODUCTS_TO_LOAD
MAX_QUERY_LENGTH = settings.MAX_QUERY_LENGTH
MAX_QUERY_RESULTS = settings.MAX_QUERY_RESULTS

CHROMA_DB_COLLECTION = settings.CHROMA_DB_COLLECTION
VECTOR_INDEX_PATH = settings.VECTOR_INDEX_PATH
CHROMA_DB_PATH = settings.CHROMA_DB_PATH
VECTOR_BACKEND = settings.VECTOR_BACKEND
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
DEVICE = settings.DEVICE
CHROMA_SETTINGS = settings.CHROMA_SETTINGS
BUILD_INDEX_IF_MISSING = settings.BUILD_INDEX_IF_MISSING

RL_MIN_SAMPLES = settings.RL_MIN_SAMPLES
BATCH_SIZE = settings.BATCH_SIZE
CACHE_ENABLED = settings.CACHE_ENABLED
RLHF_CHECKPOINT = settings.RLHF_CHECKPOINT

ANONYMIZED_TELEMETRY = settings.ANONYMIZED_TELEMETRY

# Constantes ML
ML_ENABLED = settings.ML_ENABLED
ML_FEATURES = list(settings.ML_FEATURES)
ML_EMBEDDING_MODEL = settings.ML_EMBEDDING_MODEL
ML_USE_GPU = settings.ML_USE_GPU
ML_CATEGORIES = settings.ML_CATEGORIES
ML_CACHE_SIZE = settings.ML_CACHE_SIZE
ML_CONFIDENCE_THRESHOLD = settings.ML_CONFIDENCE_THRESHOLD
ML_WEIGHT = settings.ML_WEIGHT
ML_MIN_SIMILARITY = settings.ML_MIN_SIMILARITY