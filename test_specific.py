# test_specific.py
"""
Pruebas específicas de funcionalidades ML.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_individual_components():
    """Prueba cada componente individualmente."""
    
    print("\n🔬 PRUEBA DE COMPONENTES INDIVIDUALES")
    print("="*50)
    
    # 1. Probar solo el preprocesador ML
    print("\n1. Probando ProductDataPreprocessor...")
    try:
        from src.core.data.ml_processor import ProductDataPreprocessor
        
        # Crear preprocesador en modo verbose
        processor = ProductDataPreprocessor(verbose=True)
        
        # Procesar un producto de prueba
        test_product = {
            "title": "Nintendo Switch OLED Model - White",
            "description": "Consola de videojuegos Nintendo Switch con pantalla OLED de 7 pulgadas, 64GB almacenamiento, controles Joy-Con",
            "price": 349.99,
            "main_category": "Gaming"
        }
        
        result = processor.preprocess_product(test_product)
        print(f"✅ Preprocesador funcionando")
        print(f"   Título procesado: {result.get('title')}")
        print(f"   ML procesado: {result.get('ml_processed', False)}")
        
        if result.get('predicted_category'):
            print(f"   Categoría ML: {result['predicted_category']}")
        
        return True
    except Exception as e:
        print(f"❌ Error en preprocesador: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_ml_features():
    """Prueba características ML específicas."""
    
    print("\n2. Probando características ML...")
    
    try:
        from src.core.data.product import MLProductEnricher
        
        # Habilitar features específicos
        product_data = {
            "title": "Microsoft Surface Laptop 5",
            "description": "Laptop ultradelgada con Windows 11, procesador Intel Core i7, 16GB RAM, SSD 512GB",
            "price": 1399.99
        }
        
        # Probar solo categorías
        print("\n   Probando solo categorización...")
        result1 = MLProductEnricher.enrich_product(
            product_data,
            enable_features=['category']
        )
        print(f"   Categoría predicha: {result1.get('predicted_category', 'Ninguna')}")
        
        # Probar solo entidades
        print("\n   Probando solo extracción de entidades...")
        result2 = MLProductEnricher.enrich_product(
            product_data,
            enable_features=['entities']
        )
        if result2.get('extracted_entities'):
            print(f"   Entidades encontradas: {result2['extracted_entities']}")
        
        # Probar todo
        print("\n   Probando todas las features...")
        result3 = MLProductEnricher.enrich_product(product_data)
        print(f"   Categoría: {result3.get('predicted_category', 'Ninguna')}")
        print(f"   Tiene embedding: {'embedding' in result3}")
        
        return True
    except Exception as e:
        print(f"❌ Error en características ML: {e}")
        return False

def test_memory_management():
    """Prueba la gestión de memoria."""
    
    print("\n3. Probando gestión de memoria...")
    
    try:
        from src.core.data.ml_processor import ProductDataPreprocessor
        
        # Configurar límites bajos de memoria
        processor = ProductDataPreprocessor(
            max_memory_mb=256,  # Solo 256MB
            verbose=True
        )
        
        print(f"✅ Preprocesador configurado con límites de memoria")
        print(f"   Tamaño máximo de batch: {processor.max_batch_size}")
        print(f"   Tamaño máximo de cache: {processor.max_cache_size}")
        
        return True
    except Exception as e:
        print(f"❌ Error en gestión de memoria: {e}")
        return False

def main():
    """Función principal."""
    print("\n🧪 PRUEBAS ESPECÍFICAS DE ML")
    print("="*50)
    
    results = []
    results.append(("Componentes", test_individual_components()))
    results.append(("Características ML", test_ml_features()))
    results.append(("Memoria", test_memory_management()))
    
    print("\n" + "="*50)
    print("📊 RESULTADOS DE PRUEBAS ESPECÍFICAS")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 {passed}/{total} pruebas específicas exitosas")

if __name__ == "__main__":
    main()