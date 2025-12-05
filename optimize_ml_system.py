# optimize_ml_system.py
"""
Script para optimizar y probar el sistema ML mejorado.
"""

import sys
import os
from pathlib import Path
import time
import logging

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_model_cache():
    """Configura y pre-descarga modelos."""
    logger.info("🔄 Configurando cache de modelos...")
    
    try:
        from src.core.utils.model_cache import model_cache
        
        # Pre-descargar modelos esenciales
        model_cache.pre_download_essential_models()
        
        # Verificar modelos descargados
        models_to_check = [
            ("embedding", "all-MiniLM-L6-v2"),
            ("zero_shot", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"),
            ("ner", "Davlan/bert-base-multilingual-cased-ner-hrl"),
        ]
        
        logger.info("📋 Modelos en cache:")
        for model_type, model_name in models_to_check:
            path = model_cache.get_model_path(model_type, model_name)
            status = "✅" if path else "❌"
            logger.info(f"  {status} {model_type}: {model_name}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error configurando cache: {e}")
        return False

def test_optimized_loader():
    """Prueba el DataLoader optimizado."""
    logger.info("\n🧪 Probando DataLoader optimizado...")
    
    try:
        from src.core.data.loader import FastDataLoader
        
        # Crear datos de prueba rápidos
        test_data = [
            {
                "title": "iPhone 15 Pro Max",
                "description": "Smartphone Apple con cámara 48MP, chip A17 Pro, pantalla 6.7\"",
                "price": 1199.99,
                "main_category": "Electronics"
            },
            {
                "title": "Sony WH-1000XM5",
                "description": "Audífonos inalámbricos con cancelación de ruido líder",
                "price": 399.99,
                "main_category": "Electronics"
            }
        ]
        
        # Guardar datos
        test_dir = Path("./test_optimized")
        test_dir.mkdir(exist_ok=True)
        
        import json
        with open(test_dir / "test.jsonl", "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
        
        # Probar con cache de modelos
        start_time = time.time()
        
        loader = FastDataLoader(
            raw_dir=test_dir,
            processed_dir=test_dir / "processed",
            ml_enabled=True,
            ml_features=["category", "entities"],
            max_products_per_file=10,
            use_progress_bar=False
        )
        
        products = loader.load_data()
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Carga completada en {elapsed:.1f} segundos")
        logger.info(f"📦 Productos cargados: {len(products)}")
        
        if products:
            p = products[0]
            logger.info(f"🔍 Producto 1:")
            logger.info(f"   Título: {p.title}")
            logger.info(f"   ML Procesado: {getattr(p, 'ml_processed', False)}")
            
            if hasattr(p, 'predicted_category') and p.predicted_category:
                logger.info(f"   Categoría ML: {p.predicted_category}")
            
            if hasattr(p, 'ml_tags') and p.ml_tags:
                logger.info(f"   Tags ML: {p.ml_tags}")
        
        # Limpiar
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def benchmark_ml_components():
    """Benchmark de componentes ML individuales."""
    logger.info("\n⚡ Benchmark de componentes ML...")
    
    results = []
    
    try:
        from src.core.data.ml_processor import ProductDataPreprocessor
        
        # Inicializar con cache
        start = time.time()
        processor = ProductDataPreprocessor(verbose=False)
        init_time = time.time() - start
        results.append(("Inicialización", init_time))
        logger.info(f"  • Inicialización: {init_time:.2f}s")
        
        # Benchmark embeddings
        start = time.time()
        test_text = "Smartphone de última generación con cámara profesional"
        embedding = processor._generate_embedding(test_text)
        embed_time = time.time() - start
        results.append(("Embedding", embed_time))
        logger.info(f"  • Embedding (384-dim): {embed_time:.2f}s")
        
        # Benchmark categorización
        if processor.zero_shot_classifier:
            start = time.time()
            category = processor._predict_category_zero_shot(test_text)
            category_time = time.time() - start
            results.append(("Categorización", category_time))
            logger.info(f"  • Categorización: {category_time:.2f}s")
            logger.info(f"    → Resultado: {category}")
        
        # Benchmark NER
        if processor.ner_pipeline:
            start = time.time()
            entities = processor._extract_entities_ner(test_text)
            ner_time = time.time() - start
            results.append(("NER", ner_time))
            logger.info(f"  • Extracción entidades: {ner_time:.2f}s")
        
        # Resumen
        total_time = sum(t for _, t in results)
        logger.info(f"\n📊 Total benchmark: {total_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en benchmark: {e}")
        return False

def create_config_file():
    """Crea archivo de configuración optimizado."""
    config_content = (
        "# config_optimized.py\n"
        '"""Configuración optimizada para sistema ML local."""\n\n'
        "from pathlib import Path\n\n"
        "# Directorios\n"
        "RAW_DIR = Path('./data/raw')\n"
        "PROC_DIR = Path('./data/processed')\n"
        "VECTOR_DIR = Path('./data/vector')\n\n"
        "# Configuración ML optimizada\n"
        "ML_ENABLED = True\n"
        "ML_FEATURES = ['category', 'entities', 'embedding']  # Tags removido temporalmente para velocidad\n"
        "ML_CATEGORIES = [\n"
        "    'Electronics','Home & Kitchen','Clothing & Accessories','Books & Media','Sports & Outdoors',\n"
        "    'Health & Beauty','Automotive','Office Supplies','Toys & Games','Other'\n"
        "]\n\n"
        "# Modelos optimizados\n"
        "ML_MODELS = {\n"
        "    'embedding': 'all-MiniLM-L6-v2',\n"
        "    'zero_shot': 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7',\n"
        "    'ner': 'Davlan/bert-base-multilingual-cased-ner-hrl',\n"
        "}\n\n"
        "# Performance\n"
        "ML_BATCH_SIZE = 8\n"
        "ML_MAX_MEMORY_MB = 1024\n"
        "ML_USE_CACHE = True\n"
        "ML_CACHE_DIR = Path.home() / '.cache' / 'proyecto_ml'\n\n"
        "# DataLoader optimizado\n"
        "LOADER_MAX_PRODUCTS = 200\n"
        "LOADER_USE_PROGRESS = True\n"
        "LOADER_AUTO_CATEGORIES = True\n\n"
        "# Configuración local\n"
        "LOCAL_LLM_ENABLED = False\n"
        "LOCAL_LLM_MODEL = 'llama-3.2-3b-instruct'\n"
        "DEVICE = 'cpu'\n"
    )

    config_path = Path('./config_optimized.py')
    config_path.write_text(config_content, encoding='utf-8')
    logger.info(f'✅ Archivo de configuración creado: {config_path}')
    return config_path


def main():
    """Función principal."""
    print("\n🚀 OPTIMIZACIÓN DEL SISTEMA ML")
    print("="*50)
    
    logger.info("Iniciando optimización...")
    
    # 1. Crear configuración optimizada
    config_file = create_config_file()
    
    # 2. Configurar cache de modelos
    cache_success = setup_model_cache()
    
    # 3. Benchmark
    benchmark_success = benchmark_ml_components()
    
    # 4. Probar loader optimizado
    loader_success = test_optimized_loader()
    
    # Resumen
    print("\n" + "="*50)
    print("🎯 RESUMEN DE OPTIMIZACIÓN")
    print("="*50)
    
    results = [
        ("Configuración", True),
        ("Cache Modelos", cache_success),
        ("Benchmark", benchmark_success),
        ("Loader", loader_success)
    ]
    
    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    print("\n💡 Recomendaciones:")
    print("  1. Usa config_optimized.py para producción")
    print("  2. Los modelos ahora están en cache (~/.cache/)")
    print("  3. Para mayor velocidad, considera:")
    print("     • Reducir ML_FEATURES a ['category', 'embedding']")
    print("     • Usar batch_size más pequeño (4-8)")
    print("     • Deshabilitar TF-IDF si no es esencial")

if __name__ == "__main__":
    main()