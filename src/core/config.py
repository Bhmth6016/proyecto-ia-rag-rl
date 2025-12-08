# src/core/config.py - VERSIÓN COMPLETAMENTE CORREGIDA
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import List, Optional, Dict, Any, Set, ClassVar
import logging

# Configurar logger básico primero (sin settings)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# ============================================================
# CONFIGURACIÓN GLOBAL SINGLETON - SIN IMPORTS CIRCULARES
# ============================================================

class GlobalSettings(BaseModel):
    """
    Configuración central del sistema con gestión unificada de ML 100% local.
    Implementa patrón Singleton para asegurar una única instancia.
    """
    
    # Configuración Pydantic v2
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra='forbid',
        frozen=False
    )
    
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
        ge=10,
        le=300,
        description="Timeout en segundos para llamadas al LLM local (10-300s)"
    )
    
    LOCAL_LLM_TEMPERATURE: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperatura para generación del LLM local (0.0-2.0)"
    )
    
    LOCAL_LLM_MAX_TOKENS: int = Field(
        default=1024,
        ge=128,
        le=4096,
        description="Máximo de tokens para respuesta del LLM"
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
        description="Dispositivo para embeddings (cpu/cuda/mps)"
    )
    
    # ======================
    # LOGGING
    # ======================
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Formato del log"
    )
    
    LOG_FILE: Path = Field(
        default_factory=lambda: Path("./logs/amazon_recommendations.log"),
        description="Ruta del archivo de log"
    )
    
    # ======================
    # DIRECTORIOS
    # ======================
    BASE_DIR: Path = Field(
        default_factory=lambda: Path.cwd().resolve(),
        description="Directorio base del proyecto"
    )
    
    DATA_DIR: Path = Field(
        default_factory=lambda: Path.cwd().resolve() / "data",
        description="Directorio de datos"
    )
    
    RAW_DIR: Path = Field(
        default_factory=lambda: Path.cwd().resolve() / "data" / "raw",
        description="Datos crudos"
    )
    
    PROC_DIR: Path = Field(
        default_factory=lambda: Path.cwd().resolve() / "data" / "processed",
        description="Datos procesados"
    )
    
    LOG_DIR: Path = Field(
        default_factory=lambda: Path.cwd().resolve() / "logs",
        description="Directorio de logs"
    )
    
    MODELS_DIR: Path = Field(
        default_factory=lambda: Path.cwd().resolve() / "data" / "models",
        description="Directorio de modelos"
    )
    
    FEEDBACK_DIR: Path = Field(
        default_factory=lambda: Path.cwd().resolve() / "data" / "feedback",
        description="Directorio de feedback"
    )
    
    HISTORIAL_DIR: Path = Field(
        default_factory=lambda: Path.cwd().resolve() / "data" / "processed" / "historial",
        description="Directorio de historial"
    )
    
    LOCAL_MODELS_DIR: Path = Field(
        default_factory=lambda: Path.cwd().resolve() / "data" / "local_models",
        description="Directorio de modelos locales"
    )
    
    VECTOR_STORE_DIR: Path = Field(
        default_factory=lambda: Path.cwd().resolve() / "data" / "vector_store",
        description="Directorio de almacenamiento vectorial"
    )
    
    # ======================
    # LÍMITES DEL SISTEMA - CORREGIDO
    # ======================
    MAX_PRODUCTS_TO_LOAD: int = Field(
        default=1000,  # Cambiado de 1_000_000 a 1000 para pruebas
        ge=100,
        description="Máximo de productos a cargar"
    )
    
    MAX_QUERY_LENGTH: int = Field(
        default=20000,
        ge=100,
        le=100_000,
        description="Longitud máxima de consulta"
    )
    
    MAX_QUERY_RESULTS: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Máximo de resultados por consulta"
    )
    
    CACHE_ENABLED: bool = Field(
        default=True,
        description="Habilitar caché"
    )
    
    CACHE_TTL: int = Field(
        default=3600,
        ge=60,
        description="TTL de caché en segundos"
    )
    
    # ======================
    # ALMACENAMIENTO VECTORIAL
    # ======================
    VECTOR_BACKEND: str = Field(
        default="chroma",
        description="Backend vectorial (chroma/faiss)"
    )
    
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Modelo de embeddings principal"
    )
    
    DEVICE: str = Field(
        default="cpu",
        description="Dispositivo para cómputos"
    )
    
    SIMILARITY_METRIC: str = Field(
        default="cosine",
        description="Métrica de similitud (cosine/euclidean/dot)"
    )
    
    # Nuevas rutas para compatibilidad
    VECTOR_INDEX_PATH: Path = Field(
        default_factory=lambda: Path.cwd().resolve() / "data" / "vector",
        description="Ruta del índice vectorial"
    )
    
    CHROMA_DB_PATH: Path = Field(
        default_factory=lambda: Path.cwd().resolve() / "data" / "processed" / "chroma_db",
        description="Ruta de la base de datos Chroma"
    )
    
    # ======================
    # CONFIGURACIÓN ML (ÚNICA FUENTE DE VERDAD)
    # ======================
    ML_ENABLED: bool = Field(
        default=True,
        description="Master switch: Habilitar TODAS las características ML"
    )
    
    ML_FEATURES: Set[str] = Field(
        default_factory=lambda: {"category", "entities", "similarity"},
        description="Características ML a habilitar"
    )
    
    ML_EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Modelo de embeddings para búsqueda semántica"
    )
    
    ML_USE_GPU: bool = Field(
        default=False,
        description="Usar GPU para modelos ML"
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
        description="Categorías predefinidas para clasificación"
    )
    
    ML_CACHE_SIZE: int = Field(
        default=1000,
        ge=100,
        description="Tamaño máximo de caché de embeddings"
    )
    
    ML_CONFIDENCE_THRESHOLD: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Confianza mínima para predicciones ML"
    )
    
    ML_WEIGHT: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Peso para puntajes ML en ranking híbrido"
    )
    
    ML_MIN_SIMILARITY: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Umbral mínimo de similitud para recomendaciones ML"
    )
    
    ML_LAZY_LOAD: bool = Field(
        default=True,
        description="Carga diferida de modelos ML"
    )
    
    # Flag interno para control de propagación
    ml_propagated: bool = Field(
        default=False,
        description="Flag interno: ¿configuración ML propagada?",
        exclude=True
    )
    
    # ======================
    # RL / RLHF
    # ======================
    RL_ENABLED: bool = Field(
        default=False,
        description="Habilitar Reinforcement Learning"
    )
    
    RL_MIN_SAMPLES: int = Field(
        default=10,
        ge=1,
        description="Mínimo de muestras para RL"
    )
    
    BATCH_SIZE: int = Field(
        default=5000,
        ge=1,
        le=10000,
        description="Tamaño de batch"
    )
    
    RLHF_CHECKPOINT: Optional[str] = Field(
        default=None,
        description="Ruta del checkpoint RLHF"
    )
    
    # ======================
    # TELEMETRÍA
    # ======================
    ANONYMIZED_TELEMETRY: bool = Field(
        default=False,
        description="Telemetría anonimizada"
    )
    
    # ======================
    # MÉTODOS DE INICIALIZACIÓN
    # ======================
    
    @model_validator(mode='after')
    def initialize_directories(self) -> 'GlobalSettings':
        """Crear directorios necesarios al inicializar."""
        directories = [
            self.DATA_DIR,
            self.RAW_DIR,
            self.PROC_DIR,
            self.LOG_DIR,
            self.MODELS_DIR,
            self.FEEDBACK_DIR,
            self.HISTORIAL_DIR,
            self.LOCAL_MODELS_DIR,
            self.VECTOR_STORE_DIR,
            self.VECTOR_INDEX_PATH.parent,
            self.CHROMA_DB_PATH.parent
        ]
        
        for directory in directories:
            try:
                if directory:
                    directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"⚠️ No se pudo crear directorio {directory}: {e}")
        
        return self
    
    @model_validator(mode='after')
    def setup_logging(self) -> 'GlobalSettings':
        """Configurar logging después de la inicialización."""
        import logging.config
        
        log_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': self.LOG_FORMAT,
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'standard',
                    'level': self.LOG_LEVEL,
                    'stream': sys.stdout
                },
                'file': {
                    'class': 'logging.FileHandler',
                    'formatter': 'standard',
                    'level': self.LOG_LEVEL,
                    'filename': str(self.LOG_FILE),
                    'mode': 'a',
                    'encoding': 'utf-8'
                }
            },
            'loggers': {
                '': {
                    'handlers': ['console', 'file'],
                    'level': self.LOG_LEVEL,
                    'propagate': True
                }
            }
        }
        
        try:
            logging.config.dictConfig(log_config)
        except Exception as e:
            print(f"⚠️ Error configurando logging: {e}")
        
        return self
    
    @model_validator(mode='after')
    def propagate_ml_settings(self) -> 'GlobalSettings':
        """Propagar configuración ML (ejecutado después de inicialización completa)."""
        if self.ml_propagated:
            return self
        
        # Esta función se llamará después de que todo esté inicializado
        # La propagación real se hará cuando se necesite
        print(f"🤖 Configuración ML lista: habilitado={self.ML_ENABLED}, características={list(self.ML_FEATURES)}")
        self.ml_propagated = True
        return self
    
    # ======================
    # MÉTODOS PÚBLICOS
    # ======================
    
    def update_ml_settings(
        self,
        ml_enabled: Optional[bool] = None,
        ml_features: Optional[List[str]] = None,
        ml_embedding_model: Optional[str] = None,
        ml_categories: Optional[List[str]] = None
    ) -> bool:
        """
        Actualizar configuración ML dinámicamente.
        
        Args:
            ml_enabled: Habilitar/deshabilitar ML
            ml_features: Lista de características ML
            ml_embedding_model: Modelo de embeddings
            ml_categories: Categorías para clasificación
        
        Returns:
            True si hubo cambios, False en caso contrario
        """
        updates_made = False
        
        # Actualizar valores
        if ml_enabled is not None and ml_enabled != self.ML_ENABLED:
            self.ML_ENABLED = ml_enabled
            updates_made = True
            
        if ml_features is not None:
            if ml_features == ['all']:
                new_features = {"category", "entities", "tags", "embedding", "similarity"}
            else:
                new_features = set(ml_features)
            
            if new_features != self.ML_FEATURES:
                self.ML_FEATURES = new_features
                updates_made = True
        
        if ml_embedding_model is not None and ml_embedding_model != self.ML_EMBEDDING_MODEL:
            self.ML_EMBEDDING_MODEL = ml_embedding_model
            updates_made = True
            
        if ml_categories is not None and ml_categories != self.ML_CATEGORIES:
            self.ML_CATEGORIES = ml_categories
            updates_made = True
        
        # Reprogramar propagación si hubo cambios
        if updates_made:
            self.ml_propagated = False
            print(f"📡 Configuración ML actualizada: habilitado={self.ML_ENABLED}, características={list(self.ML_FEATURES)}")
        
        return updates_made
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Obtener configuración ML actual."""
        return {
            'ML_ENABLED': self.ML_ENABLED,
            'ML_FEATURES': list(self.ML_FEATURES),
            'ML_EMBEDDING_MODEL': self.ML_EMBEDDING_MODEL,
            'ML_USE_GPU': self.ML_USE_GPU,
            'ML_CATEGORIES': self.ML_CATEGORIES,
            'ML_CACHE_SIZE': self.ML_CACHE_SIZE,
            'ML_CONFIDENCE_THRESHOLD': self.ML_CONFIDENCE_THRESHOLD,
            'ML_WEIGHT': self.ML_WEIGHT,
            'ML_MIN_SIMILARITY': self.ML_MIN_SIMILARITY,
            'ML_LAZY_LOAD': self.ML_LAZY_LOAD
        }
    
    def get_local_llm_config(self) -> Dict[str, Any]:
        """Obtener configuración de LLM local."""
        return {
            'LOCAL_LLM_ENABLED': self.LOCAL_LLM_ENABLED,
            'LOCAL_LLM_MODEL': self.LOCAL_LLM_MODEL,
            'LOCAL_LLM_ENDPOINT': self.LOCAL_LLM_ENDPOINT,
            'LOCAL_LLM_TIMEOUT': self.LOCAL_LLM_TIMEOUT,
            'LOCAL_LLM_TEMPERATURE': self.LOCAL_LLM_TEMPERATURE,
            'LOCAL_LLM_MAX_TOKENS': self.LOCAL_LLM_MAX_TOKENS,
            'LOCAL_EMBEDDING_MODEL': self.LOCAL_EMBEDDING_MODEL,
            'LOCAL_EMBEDDING_DEVICE': self.LOCAL_EMBEDDING_DEVICE
        }
    
    def get_vector_config(self) -> Dict[str, Any]:
        """Obtener configuración de almacenamiento vectorial."""
        return {
            'VECTOR_BACKEND': self.VECTOR_BACKEND,
            'EMBEDDING_MODEL': self.EMBEDDING_MODEL,
            'DEVICE': self.DEVICE,
            'SIMILARITY_METRIC': self.SIMILARITY_METRIC,
            'VECTOR_STORE_DIR': str(self.VECTOR_STORE_DIR),
            'VECTOR_INDEX_PATH': str(self.VECTOR_INDEX_PATH),
            'CHROMA_DB_PATH': str(self.CHROMA_DB_PATH)
        }
    
    def is_ml_feature_enabled(self, feature: str) -> bool:
        """Verificar si una característica ML está habilitada."""
        return feature in self.ML_FEATURES
    
    def is_local_llm_enabled(self) -> bool:
        """Verificar si el LLM local está habilitado."""
        return self.LOCAL_LLM_ENABLED
    
    def is_ml_enabled(self) -> bool:
        """Verificar si ML está habilitado."""
        return self.ML_ENABLED
    
    def update_from_env(self) -> None:
        """Actualizar configuración desde variables de entorno."""
        # Helper function
        def get_env_bool(key: str, default: bool) -> bool:
            value = os.getenv(key, "").lower()
            return value in {"true", "1", "yes", "y"} if value else default
        
        def get_env_int(key: str, default: int) -> int:
            value = os.getenv(key)
            return int(value) if value else default
        
        def get_env_float(key: str, default: float) -> float:
            value = os.getenv(key)
            return float(value) if value else default
        
        # ======================
        # ACTUALIZAR VALORES
        # ======================
        
        # LLM Local
        self.LOCAL_LLM_ENABLED = get_env_bool("LOCAL_LLM_ENABLED", self.LOCAL_LLM_ENABLED)
        self.LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", self.LOCAL_LLM_MODEL)
        self.LOCAL_LLM_ENDPOINT = os.getenv("LOCAL_LLM_ENDPOINT", self.LOCAL_LLM_ENDPOINT)
        self.LOCAL_LLM_TIMEOUT = get_env_int("LOCAL_LLM_TIMEOUT", self.LOCAL_LLM_TIMEOUT)
        self.LOCAL_LLM_TEMPERATURE = get_env_float("LOCAL_LLM_TEMPERATURE", self.LOCAL_LLM_TEMPERATURE)
        self.LOCAL_LLM_MAX_TOKENS = get_env_int("LOCAL_LLM_MAX_TOKENS", self.LOCAL_LLM_MAX_TOKENS)
        
        # Embeddings Locales
        self.LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", self.LOCAL_EMBEDDING_MODEL)
        self.LOCAL_EMBEDDING_DEVICE = os.getenv("LOCAL_EMBEDDING_DEVICE", self.LOCAL_EMBEDDING_DEVICE)
        
        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", self.LOG_LEVEL)
        self.LOG_FORMAT = os.getenv("LOG_FORMAT", self.LOG_FORMAT)
        log_file_env = os.getenv("LOG_FILE")
        if log_file_env:
            self.LOG_FILE = Path(log_file_env)
        
        # Límites del Sistema
        max_products_env = os.getenv("MAX_PRODUCTS_TO_LOAD")
        if max_products_env:
            max_products = get_env_int("MAX_PRODUCTS_TO_LOAD", self.MAX_PRODUCTS_TO_LOAD)
            # Asegurar que cumple con la validación
            if max_products >= 100:  # Cambiado de 1000 a 100
                self.MAX_PRODUCTS_TO_LOAD = max_products
            else:
                print(f"⚠️ MAX_PRODUCTS_TO_LOAD={max_products} es muy pequeño, usando valor por defecto: {self.MAX_PRODUCTS_TO_LOAD}")
        
        self.MAX_QUERY_LENGTH = get_env_int("MAX_QUERY_LENGTH", self.MAX_QUERY_LENGTH)
        self.MAX_QUERY_RESULTS = get_env_int("MAX_QUERY_RESULTS", self.MAX_QUERY_RESULTS)
        self.CACHE_ENABLED = get_env_bool("CACHE_ENABLED", self.CACHE_ENABLED)
        self.CACHE_TTL = get_env_int("CACHE_TTL", self.CACHE_TTL)
        
        # Almacenamiento Vectorial
        self.VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", self.VECTOR_BACKEND)
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", self.EMBEDDING_MODEL)
        self.DEVICE = os.getenv("DEVICE", self.DEVICE)
        self.SIMILARITY_METRIC = os.getenv("SIMILARITY_METRIC", self.SIMILARITY_METRIC)
        
        # Configuración ML
        self.ML_ENABLED = get_env_bool("ML_ENABLED", self.ML_ENABLED)
        
        ml_features_env = os.getenv("ML_FEATURES")
        if ml_features_env:
            self.ML_FEATURES = set(f.strip() for f in ml_features_env.split(","))
        
        self.ML_EMBEDDING_MODEL = os.getenv("ML_EMBEDDING_MODEL", self.ML_EMBEDDING_MODEL)
        self.ML_USE_GPU = get_env_bool("ML_USE_GPU", self.ML_USE_GPU)
        
        ml_categories_env = os.getenv("ML_CATEGORIES")
        if ml_categories_env:
            self.ML_CATEGORIES = [c.strip() for c in ml_categories_env.split(",")]
        
        self.ML_CACHE_SIZE = get_env_int("ML_CACHE_SIZE", self.ML_CACHE_SIZE)
        self.ML_CONFIDENCE_THRESHOLD = get_env_float("ML_CONFIDENCE_THRESHOLD", self.ML_CONFIDENCE_THRESHOLD)
        self.ML_WEIGHT = get_env_float("ML_WEIGHT", self.ML_WEIGHT)
        self.ML_MIN_SIMILARITY = get_env_float("ML_MIN_SIMILARITY", self.ML_MIN_SIMILARITY)
        self.ML_LAZY_LOAD = get_env_bool("ML_LAZY_LOAD", self.ML_LAZY_LOAD)
        
        # RL/RLHF
        self.RL_ENABLED = get_env_bool("RL_ENABLED", self.RL_ENABLED)
        self.RL_MIN_SAMPLES = get_env_int("RL_MIN_SAMPLES", self.RL_MIN_SAMPLES)
        self.BATCH_SIZE = get_env_int("BATCH_SIZE", self.BATCH_SIZE)
        self.RLHF_CHECKPOINT = os.getenv("RLHF_CHECKPOINT", self.RLHF_CHECKPOINT)
        
        # Telemetría
        self.ANONYMIZED_TELEMETRY = get_env_bool("ANONYMIZED_TELEMETRY", self.ANONYMIZED_TELEMETRY)
        
        # Reprogramar propagación de ML
        self.ml_propagated = False
        
        print("✅ Configuración actualizada desde variables de entorno")
    SYSTEM_MODES: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "enhanced": {
                "name": "Enhanced Mode",
                "description": "Usa NER, Zero-Shot, embeddings ML, filtro colaborativo y RLHF",
                "features": ["ner", "zero_shot", "ml_embeddings", "collaborative_filter", "rlhf"],
                "ml_enabled": True,
                "ner_enabled": True,
                "zero_shot_enabled": True
            },
            "basic": {
                "name": "Basic Mode",
                "description": "Búsqueda semántica simple sin ML avanzado",
                "features": ["semantic_search"],
                "ml_enabled": False,
                "ner_enabled": False,
                "zero_shot_enabled": False
            },
            "balanced": {
                "name": "Balanced Mode",
                "description": "ML básico sin NLP avanzado",
                "features": ["ml_embeddings", "collaborative_filter"],
                "ml_enabled": True,
                "ner_enabled": False,
                "zero_shot_enabled": False
            }
        }
    )
    PRODUCT_REF_ENABLED: bool = Field(
        default=True,
        description="Habilitar sistema ProductReference"
    )

    CURRENT_MODE: str = Field(
        default="enhanced",
        description="Modo actual del sistema"
    )

    # 🔥 NUEVO: Configuración NLP
    NLP_ENABLED: bool = Field(
        default=True,
        description="Habilitar procesamiento NLP"
    )

    NER_MODEL: str = Field(
        default="dslim/distilbert-NER",
        description="Modelo para Named Entity Recognition"
    )

    ZERO_SHOT_MODEL: str = Field(
        default="typeform/distilbert-base-uncased-mnli",
        description="Modelo para Zero-Shot Classification"
    )
    def apply_mode_config(self, mode: str):
        """
        Aplica la configuración de un modo específico.
        """
        if mode not in self.SYSTEM_MODES:
            print(f"⚠️ Modo '{mode}' no encontrado, usando 'enhanced'")
            mode = "enhanced"
        
        mode_config = self.SYSTEM_MODES[mode]
        
        # Aplicar configuración
        self.CURRENT_MODE = mode
        self.ML_ENABLED = mode_config.get('ml_enabled', True)
        self.NLP_ENABLED = mode_config.get('ner_enabled', False) and mode_config.get('zero_shot_enabled', False)
        
        # 🔥 Mantener ProductReference siempre habilitado (es esencial)
        self.PRODUCT_REF_ENABLED = True
        
        # Actualizar características ML según el modo
        if mode == "basic":
            self.ML_FEATURES = set()  # Sin características ML
        elif mode == "balanced":
            self.ML_FEATURES = {"embedding", "category"}  # ML básico
        elif mode == "enhanced":
            self.ML_FEATURES = {"embedding", "category", "entities", "similarity", "ner", "zero_shot"}
        
        print(f"🔧 Modo '{mode}' aplicado:")
        print(f"   • ML: {'✅' if self.ML_ENABLED else '❌'}")
        print(f"   • NLP: {'✅' if self.NLP_ENABLED else '❌'}")
        print(f"   • Características ML: {list(self.ML_FEATURES)}")
    def get_current_mode_config(self) -> Dict[str, Any]:
        """Obtener configuración del modo actual."""
        mode_config = self.SYSTEM_MODES.get(self.CURRENT_MODE, {})
        return {
            **mode_config,
            "current_mode": self.CURRENT_MODE,
            "nlp_enabled": self.NLP_ENABLED
        }
    def __str__(self) -> str:
        """Representación legible de la configuración."""
        return (
            f"GlobalSettings(\n"
            f"  LOCAL_LLM_ENABLED={self.LOCAL_LLM_ENABLED},\n"
            f"  ML_ENABLED={self.ML_ENABLED},\n"
            f"  ML_FEATURES={list(self.ML_FEATURES)},\n"
            f"  EMBEDDING_MODEL={self.EMBEDDING_MODEL},\n"
            f"  DEVICE={self.DEVICE},\n"
            f"  LOG_LEVEL={self.LOG_LEVEL}\n"
            f")"
        )
    
    def __repr__(self) -> str:
        return self.__str__()


# ============================================================
# INSTANCIA GLOBAL SINGLETON
# ============================================================

# Crear instancia global
settings = GlobalSettings()

# Actualizar desde entorno (después de crear la instancia)
settings.update_from_env()

# ============================================================
# FUNCIONES DE CONVENIENCIA
# ============================================================

def get_settings() -> GlobalSettings:
    """Obtener la instancia global de configuración."""
    return settings

def update_ml_config(
    ml_enabled: Optional[bool] = None,
    ml_features: Optional[List[str]] = None,
    ml_embedding_model: Optional[str] = None,
    ml_categories: Optional[List[str]] = None
) -> bool:
    """
    Función de conveniencia para actualizar configuración ML.
    
    Args:
        ml_enabled: Habilitar/deshabilitar ML
        ml_features: Lista de características ML
        ml_embedding_model: Modelo de embeddings
        ml_categories: Categorías para clasificación
    
    Returns:
        True si hubo cambios
    """
    return settings.update_ml_settings(
        ml_enabled=ml_enabled,
        ml_features=ml_features,
        ml_embedding_model=ml_embedding_model,
        ml_categories=ml_categories
    )

def is_ml_feature_enabled(feature: str) -> bool:
    """
    Verificar si una característica ML está habilitada.
    
    Args:
        feature: Nombre de la característica
    
    Returns:
        True si está habilitada
    """
    return settings.is_ml_feature_enabled(feature)

def get_ml_config() -> Dict[str, Any]:
    """Obtener configuración ML actual."""
    return settings.get_ml_config()

def get_local_llm_config() -> Dict[str, Any]:
    """Obtener configuración de LLM local."""
    return settings.get_local_llm_config()

# ============================================================
# INICIALIZACIÓN FINAL
# ============================================================

# Imprimir configuración cargada usando print para evitar logging circular
print("=" * 60)
print("✅ CONFIGURACIÓN LOCAL CARGADA:")
print(f"   • LLM Local: {settings.LOCAL_LLM_ENABLED} ({settings.LOCAL_LLM_MODEL})")
print(f"   • ML Habilitado: {settings.ML_ENABLED}")
print(f"   • Características ML: {list(settings.ML_FEATURES)}")
print(f"   • Embedding Model: {settings.EMBEDDING_MODEL}")
print(f"   • Dispositivo: {settings.DEVICE}")
print(f"   • Log Level: {settings.LOG_LEVEL}")
print(f"   • Directorio Datos: {settings.DATA_DIR}")
print("=" * 60)


def apply_system_mode(mode: str = "enhanced"):
    """
    Función de conveniencia para aplicar un modo del sistema.
    
    Args:
        mode: Modo a aplicar (basic, balanced, enhanced)
    """
    settings.apply_mode_config(mode)