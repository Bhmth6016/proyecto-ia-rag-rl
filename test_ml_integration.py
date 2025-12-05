# test_ml_integration.py
"""
Script de prueba completo para verificar la integración ML.
"""

import sys
import os
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

import json
import logging
from typing import List, Dict, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_product_creation():
    """Prueba la creación de productos con y sin ML."""
    logger.info("🧪 TEST 1: Creación de productos básica")
    
    try:
        from src.core.data.product import Product, AutoProductConfig
        
        # Deshabilitar ML inicialmente
        Product.configure_ml(enabled=False)
        
        # Producto simple
        product_data = {
            "title": "Wireless Bluetooth Headphones Premium",
            "description": "High-quality wireless headphones with noise cancellation",
            "price": 129.99,
            "main_category": "Electronics",
            "categories": ["Audio", "Electronics"],
            "tags": ["wireless", "bluetooth", "audio"]
        }
        
        product = Product.from_dict(product_data, ml_enrich=False)
        logger.info(f"✅ Producto básico creado: {product.title}")
        logger.info(f"   ID: {product.id}")
        logger.info(f"   Precio: ${product.price}")
        logger.info(f"   Categoría: {product.main_category}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error en test_product_creation: {e}")
        return False

def test_ml_enricher():
    """Prueba el enriquecedor ML."""
    logger.info("\n🧪 TEST 2: Enriquecimiento ML")
    
    try:
        from src.core.data.product import MLProductEnricher, AutoProductConfig
        
        # Habilitar ML
        AutoProductConfig.ML_ENABLED = True
        
        # Datos de prueba
        test_product = {
            "title": "Smartphone Samsung Galaxy S23 Ultra 5G",
            "description": "El último smartphone Samsung con cámara de 200MP, pantalla AMOLED 6.8'', 12GB RAM, 512GB almacenamiento",
            "price": 1299.99,
            "main_category": "Electronics"
        }
        
        # Enriquecer con ML
        enriched = MLProductEnricher.enrich_product(test_product)
        
        logger.info(f"✅ Producto enriquecido con ML")
        logger.info(f"   Título: {enriched.get('title')}")
        
        if enriched.get('predicted_category'):
            logger.info(f"   📊 Categoría predicha: {enriched['predicted_category']}")
        
        if enriched.get('extracted_entities'):
            entities = enriched['extracted_entities']
            logger.info(f"   🔍 Entidades extraídas: {len(entities)} grupos")
            for entity_type, entity_list in entities.items():
                logger.info(f"     - {entity_type}: {entity_list[:3]}")
        
        if enriched.get('ml_tags'):
            logger.info(f"   🏷️  Tags ML: {enriched['ml_tags']}")
        
        if enriched.get('embedding'):
            logger.info(f"   📐 Embedding generado: {len(enriched['embedding'])} dimensiones")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error en test_ml_enricher: {e}")
        return False

def test_ml_processor():
    """Prueba el preprocesador ML directamente."""
    logger.info("\n🧪 TEST 3: Procesador ML directo")
    
    try:
        from src.core.data.ml_processor import ProductDataPreprocessor
        
        # Inicializar preprocesador con verbose
        processor = ProductDataPreprocessor(
            verbose=True,
            use_gpu=False
        )
        
        # Obtener información del modelo
        model_info = processor.get_model_info()
        logger.info("📋 Información del procesador ML:")
        logger.info(f"   • Modelo de embeddings: {model_info.get('embedding_model_name')}")
        logger.info(f"   • Embeddings cargados: {model_info.get('embedding_model_loaded')}")
        logger.info(f"   • Zero-shot cargado: {model_info.get('zero_shot_classifier_loaded')}")
        logger.info(f"   • NER cargado: {model_info.get('ner_pipeline_loaded')}")
        
        # Verificar dependencias
        deps = processor.check_dependencies()
        logger.info("📦 Dependencias disponibles:")
        for dep, available in deps.items():
            status = "✅" if available else "❌"
            logger.info(f"   {status} {dep}: {available}")
        
        # Validar disponibilidad de modelos
        availability = processor.validate_model_availability()
        logger.info("🔍 Disponibilidad de modelos:")
        for model, available in availability.items():
            status = "✅" if available else "⚠️"
            logger.info(f"   {status} {model}: {available}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error en test_ml_processor: {e}")
        return False

def test_data_loader_ml():
    """Prueba el DataLoader con ML integrado."""
    logger.info("\n🧪 TEST 4: DataLoader con ML")
    
    try:
        from src.core.data.loader import FastDataLoader
        from src.core.config import settings
        
        # Crear directorios de prueba si no existen
        test_raw_dir = Path("./test_data/raw")
        test_processed_dir = Path("./test_data/processed")
        test_raw_dir.mkdir(parents=True, exist_ok=True)
        test_processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear datos de prueba
        test_products = [
            {
                "title": "Laptop Dell XPS 15",
                "description": "Laptop profesional con procesador Intel i7, 16GB RAM, SSD 1TB, pantalla 4K",
                "price": 1899.99,
                "main_category": "Electronics",
                "tags": ["laptop", "dell", "professional"]
            },
            {
                "title": "Libro Python Avanzado",
                "description": "Libro sobre programación avanzada en Python con ejemplos prácticos",
                "price": 45.99,
                "main_category": "Books",
                "tags": ["python", "programming", "book"]
            },
            {
                "title": "Cámara Sony Alpha 7 IV",
                "description": "Cámara mirrorless profesional full frame 33MP con estabilización 5 ejes",
                "price": 2499.99,
                "main_category": "Electronics",
                "tags": ["camera", "sony", "photography"]
            }
        ]
        
        # Guardar datos de prueba
        test_file = test_raw_dir / "test_products.jsonl"
        with open(test_file, "w", encoding="utf-8") as f:
            for product in test_products:
                f.write(json.dumps(product, ensure_ascii=False) + "\n")
        
        logger.info(f"📁 Datos de prueba creados en: {test_file}")
        
        # Probar DataLoader SIN ML
        logger.info("\n   📊 Probando DataLoader SIN ML...")
        loader_no_ml = FastDataLoader(
            raw_dir=test_raw_dir,
            processed_dir=test_processed_dir / "no_ml",
            ml_enabled=False,
            max_products_per_file=10,
            use_progress_bar=False
        )
        
        products_no_ml = loader_no_ml.load_data()
        logger.info(f"   ✅ Productos cargados sin ML: {len(products_no_ml)}")
        
        # Probar DataLoader CON ML
        logger.info("\n   🤖 Probando DataLoader CON ML...")
        loader_with_ml = FastDataLoader(
            raw_dir=test_raw_dir,
            processed_dir=test_processed_dir / "with_ml",
            ml_enabled=True,
            ml_features=["category", "entities", "tags"],
            max_products_per_file=10,
            use_progress_bar=False
        )
        
        products_with_ml = loader_with_ml.load_data()
        logger.info(f"   ✅ Productos cargados con ML: {len(products_with_ml)}")
        
        # Comparar resultados
        if products_with_ml and len(products_with_ml) > 0:
            ml_product = products_with_ml[0]
            logger.info(f"\n   🔍 Producto procesado con ML:")
            logger.info(f"      Título: {ml_product.title}")
            logger.info(f"      ML Procesado: {getattr(ml_product, 'ml_processed', False)}")
            
            if hasattr(ml_product, 'predicted_category') and ml_product.predicted_category:
                logger.info(f"      Categoría ML: {ml_product.predicted_category}")
            
            if hasattr(ml_product, 'ml_tags') and ml_product.ml_tags:
                logger.info(f"      Tags ML: {ml_product.ml_tags}")
        
        # Limpiar archivos de prueba
        import shutil
        if Path("./test_data").exists():
            shutil.rmtree("./test_data")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error en test_data_loader_ml: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_batch_processing():
    """Prueba el procesamiento por lotes."""
    logger.info("\n🧪 TEST 5: Procesamiento por lotes")
    
    try:
        from src.core.data.product import Product, MLProductEnricher
        from src.core.data.ml_processor import ProductDataPreprocessor
        
        # Habilitar ML
        Product.configure_ml(enabled=True)
        
        # Crear lote de productos de prueba
        batch_products = []
        for i in range(5):
            product_data = {
                "title": f"Producto de prueba {i+1}",
                "description": f"Descripción detallada del producto de prueba número {i+1} para evaluación del sistema ML",
                "price": 99.99 + (i * 50),
                "main_category": "Test Products",
                "tags": [f"test{i+1}", "sample", "demo"]
            }
            batch_products.append(product_data)
        
        logger.info(f"📦 Creando lote de {len(batch_products)} productos...")
        
        # Procesar por lotes
        enriched_batch = MLProductEnricher.enrich_batch(batch_products)
        
        logger.info(f"✅ Lote procesado: {len(enriched_batch)} productos")
        
        # Verificar algunos resultados
        if enriched_batch:
            sample = enriched_batch[0]
            logger.info(f"\n   📊 Producto de muestra del lote:")
            logger.info(f"      Título: {sample.get('title')}")
            logger.info(f"      Tiene categoría ML: {'predicted_category' in sample}")
            logger.info(f"      Tiene entidades: {'extracted_entities' in sample}")
            logger.info(f"      Tiene embedding: {'embedding' in sample}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error en test_batch_processing: {e}")
        return False

def test_end_to_end():
    """Prueba completa de extremo a extremo."""
    logger.info("\n" + "="*60)
    logger.info("🚀 TEST COMPLETO: Flujo de extremo a extremo")
    logger.info("="*60)
    
    results = []
    
    # Ejecutar todas las pruebas
    results.append(("Product Creation", test_product_creation()))
    results.append(("ML Enricher", test_ml_enricher()))
    results.append(("ML Processor", test_ml_processor()))
    results.append(("DataLoader ML", test_data_loader_ml()))
    results.append(("Batch Processing", test_batch_processing()))
    
    # Resumen
    logger.info("\n" + "="*60)
    logger.info("📊 RESUMEN DE PRUEBAS")
    logger.info("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASÓ" if success else "❌ FALLÓ"
        logger.info(f"{status} {test_name}")
        if success:
            passed += 1
    
    logger.info("\n" + "="*60)
    logger.info(f"🎯 Resultado: {passed}/{total} pruebas exitosas ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ¡TODOS LOS TESTS PASARON! El sistema ML está funcionando correctamente.")
    else:
        logger.warning("⚠️ Algunos tests fallaron. Revisar los logs para más detalles.")
    
    logger.info("="*60)

def main():
    """Función principal."""
    print("\n🧪 SISTEMA DE PRUEBA DE INTEGRACIÓN ML")
    print("="*50)
    
    try:
        test_end_to_end()
    except KeyboardInterrupt:
        logger.info("Prueba interrumpida por el usuario")
    except Exception as e:
        logger.error(f"Error no esperado: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()