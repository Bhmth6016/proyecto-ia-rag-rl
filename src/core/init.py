# src/core/init.py
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any
from src.core.data.loader import DataLoader
from src.core.data.product import Product
from src.core.rag.basic.retriever import Retriever
from src.core.config import settings
import logging
import google.generativeai as genai
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv()

class SystemInitializer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._products = None
            self._retriever = None
            self._initialized = True
            self.llm_model = genai.GenerativeModel("gemini-1.5-flash")
            
            # 🔥 NUEVO: Configuración ML
            self.ml_enabled = getattr(settings, "ML_ENABLED", False)
            self.ml_features = getattr(settings, "ML_FEATURES", ["category", "entities"])
            self.ml_models = {}
            
            # 🔥 NUEVO: Configuración específica para CollaborativeFilter
            self.collaborative_ml_config = {
                'use_ml_features': getattr(settings, "COLLABORATIVE_ML_ENABLED", True),
                'ml_weight': getattr(settings, "ML_WEIGHT", 0.3),
                'min_similar_users': getattr(settings, "MIN_SIMILAR_USERS", 3),
                'ml_embedding_dim': getattr(settings, "ML_EMBEDDING_DIM", 768)
            }
            
            # 🔥 NUEVO: Configuración para embeddings
            self.embedding_config = {
                'use_sentence_transformers': getattr(settings, "USE_SENTENCE_TRANSFORMERS", True),
                'embedding_model': getattr(settings, "EMBEDDING_MODEL_ML", "all-MiniLM-L6-v2"),
                'cache_embeddings': getattr(settings, "CACHE_EMBEDDINGS", True)
            }
            
            # 🔥 NUEVO: Inicializar modelos ML si están habilitados
            if self.ml_enabled:
                self._initialize_ml_components()
            
            logger.info(f"✅ SystemInitializer creado - ML Enabled: {self.ml_enabled}")
            logger.info(f"🔧 ML Features: {self.ml_features}")
            logger.info(f"🤝 Collaborative ML Config: {self.collaborative_ml_config}")

    def _initialize_ml_components(self) -> None:
        """Inicializa componentes ML si están habilitados en configuración"""
        try:
            logger.info("🚀 Inicializando componentes ML...")
            
            # 🔥 NUEVO: Inicializar embeddings para ML si es necesario
            if self.embedding_config['use_sentence_transformers']:
                self._initialize_sentence_transformer()
            
            # 🔥 NUEVO: Verificar si hay modelos ML pre-entrenados para cargar
            self._load_pretrained_models()
            
            # 🔥 NUEVO: Inicializar caché de embeddings
            if self.embedding_config['cache_embeddings']:
                self._initialize_embedding_cache()
                
            logger.info("✅ Componentes ML inicializados correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando componentes ML: {e}")
            # Desactivar ML si hay error en inicialización
            self.ml_enabled = False

    def _initialize_sentence_transformer(self) -> None:
        """Inicializa modelo de Sentence Transformers para embeddings"""
        try:
            # Importación condicional para evitar dependencias innecesarias
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"🔧 Cargando Sentence Transformer: {self.embedding_config['embedding_model']}")
            self.ml_models['sentence_transformer'] = SentenceTransformer(
                self.embedding_config['embedding_model']
            )
            logger.info(f"✅ Sentence Transformer cargado: {self.embedding_config['embedding_model']}")
            
        except ImportError:
            logger.warning("⚠️  Sentence Transformers no está instalado. Usando embeddings básicos.")
            self.embedding_config['use_sentence_transformers'] = False
        except Exception as e:
            logger.error(f"❌ Error cargando Sentence Transformer: {e}")
            self.embedding_config['use_sentence_transformers'] = False

    def _load_pretrained_models(self) -> None:
        """Carga modelos ML pre-entrenados si existen"""
        models_path = Path(settings.MODELS_DIR) if hasattr(settings, 'MODELS_DIR') else Path("models")
        
        if models_path.exists():
            logger.info(f"🔍 Buscando modelos pre-entrenados en: {models_path}")
            
            # Lista de modelos a buscar
            model_files = {
                'category_classifier': models_path / "category_classifier.pkl",
                'sentiment_analyzer': models_path / "sentiment_analyzer.pkl",
                'similarity_model': models_path / "similarity_model.pkl"
            }
            
            for model_name, model_path in model_files.items():
                if model_path.exists():
                    try:
                        import pickle
                        with open(model_path, 'rb') as f:
                            self.ml_models[model_name] = pickle.load(f)
                        logger.info(f"✅ Modelo {model_name} cargado desde {model_path}")
                    except Exception as e:
                        logger.error(f"❌ Error cargando modelo {model_name}: {e}")

    def _initialize_embedding_cache(self) -> None:
        """Inicializa caché de embeddings"""
        try:
            import hashlib
            import json
            from pathlib import Path
            
            cache_dir = Path(getattr(settings, "EMBEDDING_CACHE_DIR", "data/cache/embeddings"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            self.embedding_cache_path = cache_dir / "embeddings_cache.json"
            
            if self.embedding_cache_path.exists():
                with open(self.embedding_cache_path, 'r', encoding='utf-8') as f:
                    self.embedding_cache = json.load(f)
                logger.info(f"📁 Caché de embeddings cargada: {len(self.embedding_cache)} entradas")
            else:
                self.embedding_cache = {}
                logger.info("📁 Caché de embeddings inicializada vacía")
                
        except Exception as e:
            logger.error(f"❌ Error inicializando caché de embeddings: {e}")
            self.embedding_cache = {}

    def get_ml_embedding(self, text: str) -> Optional[List[float]]:
        """Obtiene embedding para texto usando modelo ML configurado"""
        if not self.ml_enabled or not self.embedding_config['use_sentence_transformers']:
            return None
            
        try:
            # 🔥 NUEVO: Verificar caché primero
            cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
            
            if self.embedding_config['cache_embeddings'] and cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
            
            # 🔥 NUEVO: Calcular embedding si no está en caché
            if 'sentence_transformer' in self.ml_models:
                embedding = self.ml_models['sentence_transformer'].encode(text).tolist()
                
                # 🔥 NUEVO: Guardar en caché
                if self.embedding_config['cache_embeddings']:
                    self.embedding_cache[cache_key] = embedding
                    self._save_embedding_cache()
                
                return embedding
                
        except Exception as e:
            logger.error(f"❌ Error obteniendo embedding ML: {e}")
            
        return None

    def _save_embedding_cache(self) -> None:
        """Guarda la caché de embeddings en disco"""
        try:
            import json
            
            with open(self.embedding_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.embedding_cache, f)
                
        except Exception as e:
            logger.error(f"❌ Error guardando caché de embeddings: {e}")

    @property
    def products(self) -> List[Product]:
        if self._products is None:
            self._load_products()
        return self._products

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._initialize_retriever()
        return self._retriever

    @property
    def loader(self) -> DataLoader:
        if not hasattr(self, '_loader'):
            self._loader = DataLoader(
                raw_dir=settings.RAW_DIR,
                processed_dir=settings.PROC_DIR
            )
        return self._loader

    def _load_products(self) -> None:
        """Load products with caching."""
        loader = DataLoader(
            raw_dir=settings.RAW_DIR,
            processed_dir=settings.PROC_DIR,
            cache_enabled=settings.CACHE_ENABLED
        )
        self._products = loader.load_data()

    def _initialize_retriever(self) -> None:
        """Initialize retriever and build index if needed."""
        logger.info(f"Initializing retriever at {settings.VECTOR_INDEX_PATH}")
        
        # Asegura que el directorio existe
        index_path = Path(settings.VECTOR_INDEX_PATH)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._retriever = Retriever(
            index_path=settings.VECTOR_INDEX_PATH,
            embedding_model=settings.EMBEDDING_MODEL,
            device=settings.DEVICE
        )
        
        # Verifica si el índice existe
        if not self._retriever.index_exists():
            logger.info("Index not found, building...")
            if not hasattr(self, '_products') or not self._products:
                self._load_products()
            self._retriever.build_index(self._products)
    
    # 🔥 NUEVO: Métodos para obtener configuración ML
    def get_ml_config(self) -> Dict[str, Any]:
        """Retorna la configuración ML completa"""
        return {
            'ml_enabled': self.ml_enabled,
            'ml_features': self.ml_features,
            'collaborative_ml_config': self.collaborative_ml_config,
            'embedding_config': self.embedding_config
        }
    
    def is_ml_feature_enabled(self, feature: str) -> bool:
        """Verifica si una feature ML específica está habilitada"""
        return feature in self.ml_features
    
    def update_ml_config(self, config_updates: Dict[str, Any]) -> None:
        """Actualiza configuración ML dinámicamente"""
        try:
            if 'ml_enabled' in config_updates:
                self.ml_enabled = config_updates['ml_enabled']
                
            if 'ml_features' in config_updates:
                self.ml_features = config_updates['ml_features']
                
            if 'collaborative_ml_config' in config_updates:
                self.collaborative_ml_config.update(config_updates['collaborative_ml_config'])
                
            if 'embedding_config' in config_updates:
                self.embedding_config.update(config_updates['embedding_config'])
                
            logger.info(f"🔧 Configuración ML actualizada: {self.get_ml_config()}")
            
        except Exception as e:
            logger.error(f"❌ Error actualizando configuración ML: {e}")


def get_system() -> SystemInitializer:
    """Global access point for initialized system."""
    return SystemInitializer()