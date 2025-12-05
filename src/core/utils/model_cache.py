# src/core/utils/model_cache.py
"""
Sistema de cache para modelos ML locales.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import huggingface_hub

logger = logging.getLogger(__name__)

class ModelCacheManager:
    """Gestiona el cache de modelos ML locales."""
    
    # Directorio de cache por defecto (estilo Hugging Face)
    DEFAULT_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
    
    # Modelos predefinidos con sus rutas locales
    LOCAL_MODEL_PATHS = {
        "embedding": {
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "all-MiniLM-L12-v2": "sentence-transformers/all-MiniLM-L12-v2",
            "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        },
        "zero_shot": {
            "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
            "facebook/bart-large-mnli": "facebook/bart-large-mnli",
        },
        "ner": {
            "Davlan/bert-base-multilingual-cased-ner-hrl": "Davlan/bert-base-multilingual-cased-ner-hrl",
            "dslim/bert-base-NER": "dslim/bert-base-NER",
        }
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar variables de entorno para Hugging Face
        os.environ['HF_HOME'] = str(self.cache_dir)
        os.environ['TRANSFORMERS_CACHE'] = str(self.cache_dir / "transformers")
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(self.cache_dir / "sentence_transformers")
        
        # Crear subdirectorios
        (self.cache_dir / "transformers").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "sentence_transformers").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Cache de modelos configurado en: {self.cache_dir}")
    
    def ensure_model_downloaded(self, model_type: str, model_name: str) -> bool:
        """Asegura que el modelo esté descargado localmente."""
        try:
            from huggingface_hub import snapshot_download
            
            if model_type not in self.LOCAL_MODEL_PATHS:
                logger.error(f"Tipo de modelo desconocido: {model_type}")
                return False
            
            if model_name not in self.LOCAL_MODEL_PATHS[model_type]:
                logger.warning(f"Modelo {model_name} no en lista predefinida")
                # Intentar descargar de todos modos
                hf_model_name = model_name
            else:
                hf_model_name = self.LOCAL_MODEL_PATHS[model_type][model_name]
            
            logger.info(f"🔍 Verificando modelo: {hf_model_name}")
            
            # Verificar si ya está en cache
            model_path = self.cache_dir / "models" / hf_model_name.replace("/", "--")
            
            if model_path.exists() and any(model_path.iterdir()):
                logger.info(f"✅ Modelo ya en cache: {model_path}")
                return True
            
            # Descargar modelo
            logger.info(f"⬇️ Descargando modelo: {hf_model_name}")
            
            snapshot_download(
                repo_id=hf_model_name,
                cache_dir=self.cache_dir,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            logger.info(f"✅ Modelo descargado: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error descargando modelo {model_name}: {e}")
            return False
    
    def pre_download_essential_models(self):
        """Pre-descarga los modelos esenciales al inicio."""
        essential_models = [
            ("embedding", "all-MiniLM-L6-v2"),
            ("zero_shot", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"),
            ("ner", "Davlan/bert-base-multilingual-cased-ner-hrl"),
        ]
        
        logger.info("🚀 Pre-descargando modelos esenciales...")
        
        for model_type, model_name in essential_models:
            success = self.ensure_model_downloaded(model_type, model_name)
            if success:
                logger.info(f"✅ {model_type}: {model_name}")
            else:
                logger.warning(f"⚠️ Falló: {model_type}: {model_name}")
    
    def get_model_path(self, model_type: str, model_name: str) -> Optional[Path]:
        """Obtiene la ruta local del modelo."""
        if model_type not in self.LOCAL_MODEL_PATHS:
            return None
        
        if model_name not in self.LOCAL_MODEL_PATHS[model_type]:
            return None
        
        hf_model_name = self.LOCAL_MODEL_PATHS[model_type][model_name]
        model_path = self.cache_dir / "models" / hf_model_name.replace("/", "--")
        
        return model_path if model_path.exists() else None
    
    def clear_cache(self):
        """Limpia el cache de modelos."""
        import shutil
        try:
            shutil.rmtree(self.cache_dir / "models", ignore_errors=True)
            logger.info("✅ Cache de modelos limpiado")
        except Exception as e:
            logger.error(f"❌ Error limpiando cache: {e}")

# Instancia global
model_cache = ModelCacheManager()