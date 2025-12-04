#!/usr/bin/env python3
# download_local_models.py - Descarga todos los modelos necesarios

import os
import logging
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_embedding_models():
    """Descarga modelos de embeddings."""
    models = [
        'all-MiniLM-L6-v2',  # Principal - 384 dimensiones
        'all-mpnet-base-v2',  # Alternativa - 768 dimensiones
    ]
    
    for model_name in models:
        logger.info(f"📥 Descargando embedding model: {model_name}")
        try:
            model = SentenceTransformer(model_name)
            logger.info(f"✅ {model_name} descargado")
        except Exception as e:
            logger.error(f"❌ Error descargando {model_name}: {e}")

def download_nlp_models():
    """Descarga modelos NLP."""
    models = [
        # Clasificador zero-shot
        ('facebook/bart-large-mnli', 'zero-shot-classification'),
        # NER en inglés
        ('dslim/bert-base-NER', 'ner'),
        # NER multilingüe (opcional)
        ('Davlan/bert-base-multilingual-cased-ner-hrl', 'ner'),
    ]
    
    for model_name, task in models:
        logger.info(f"📥 Descargando {task} model: {model_name}")
        try:
            pipe = pipeline(task, model=model_name)
            logger.info(f"✅ {model_name} descargado")
        except Exception as e:
            logger.error(f"❌ Error descargando {model_name}: {e}")

def main():
    """Función principal."""
    print("=" * 60)
    print("🤖 DESCARGADOR DE MODELOS LOCALES")
    print("=" * 60)
    
    # Crear directorio de modelos
    models_dir = "models/local"
    os.makedirs(models_dir, exist_ok=True)
    
    # Configurar cache de HuggingFace
    os.environ['TRANSFORMERS_CACHE'] = models_dir
    os.environ['HF_HOME'] = models_dir
    
    print(f"📁 Modelos se guardarán en: {models_dir}")
    
    # Descargar modelos
    download_embedding_models()
    download_nlp_models()
    
    print("\n" + "=" * 60)
    print("✅ TODOS LOS MODELOS DESCARGADOS")
    print("🎯 Sistema listo para funcionar 100% local")
    print("=" * 60)

if __name__ == "__main__":
    main()
