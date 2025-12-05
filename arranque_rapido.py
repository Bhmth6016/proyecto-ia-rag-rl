# arranque_rapido.py
import sys
import os
from pathlib import Path

# 🔥 SOLUCIÓN: Configurar Python path y prevenir imports circulares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Crear directorios críticos primero
for d in ['data/raw', 'data/processed/historial', 'data/feedback', 'logs']:
    Path(d).mkdir(parents=True, exist_ok=True)

# Monkey patch para evitar recursividad
import src.core.data.product as product_module

# Desactivar temporalmente ML durante carga inicial
original_configure_ml = product_module.Product.configure_ml

def safe_configure_ml(enabled=False, features=None, categories=None):
    """Versión segura que no activa ML durante inicialización"""
    print(f"🔧 ML configurado de forma segura: enabled={enabled}")
    return original_configure_ml(enabled=enabled, features=features, categories=categories)

product_module.Product.configure_ml = safe_configure_ml

# Ahora importar main
from main import parse_arguments, initialize_system

# Configurar argumentos
args = type('Args', (), {
    'command': 'rag',
    'ml_enabled': True,
    'max_products_to_load': 20,
    'local_llm_enabled': True,
    'verbose': True,
    'data_dir': "./data/raw",
    'log_level': 'INFO',
    'device': None,
    'ml_features': None,
    'ml_batch_size': 32,
    'use_product_embeddings': False,
    'track_ml_metrics': True,
    'log_file': None,
    'ui': False,
    'top_k': 5,
    'user_age': 25,
    'user_gender': 'male',
    'user_country': 'Spain',
    'user_language': 'es',
    'user_id': None,
    'show_ml_info': False,
    'enable_rlhf': True,
    'memory_window': 3,
    'domain': 'general'
})()

print("🚀 Iniciando sistema de forma segura...")

# Inicializar con ML desactivado temporalmente
products, rag_agent, user_manager, ml_config = initialize_system(
    data_dir=args.data_dir,
    ml_enabled=False,  # 🔥 ML desactivado durante carga
    ml_features=None,
    ml_batch_size=args.ml_batch_size,
    use_product_embeddings=args.use_product_embeddings,
    track_ml_metrics=args.track_ml_metrics,
    args=args
)

# Ahora activar ML
print("✅ Sistema base cargado, activando ML...")
product_module.Product.configure_ml(
    enabled=True,
    features=['category', 'entities', 'similarity', 'embedding']
)

print("🎉 ¡Sistema listo! Puedes continuar con el RAG loop...")