# scripts/repair_ml_embeddings.py
import json
import logging
from pathlib import Path
import sys
from typing import List, Dict, Any

# Añadir el directorio raíz al path para importaciones
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import settings
from src.core.data.product import Product
from src.core.data.ml_processor import ProductDataPreprocessor

logger = logging.getLogger(__name__)

def repair_ml_embeddings():
    """Repara embeddings ML para productos existentes."""
    print("\n🔧 REPARANDO EMBEDDINGS ML")
    print("="*50)
    
    try:
        # Ruta a productos procesados
        products_file = settings.PROC_DIR / "products.json"
        
        if not products_file.exists():
            print(f"❌ Archivo no encontrado: {products_file}")
            return
        
        # Cargar productos
        with open(products_file, 'r', encoding='utf-8') as f:
            products_data = json.load(f)
        
        print(f"📦 Productos cargados: {len(products_data)}")
        
        # Crear preprocesador ML
        ml_processor = ProductDataPreprocessor(
            verbose=True,
            use_gpu=settings.DEVICE == "cuda"
        )
        
        # Procesar productos con ML
        print("🤖 Aplicando procesamiento ML...")
        repaired_count = 0
        
        for i, product_data in enumerate(products_data):
            if i % 100 == 0:
                print(f"  Procesados {i}/{len(products_data)} productos...")
            
            try:
                # Verificar si ya tiene ML
                if not product_data.get('ml_processed', False):
                    # Aplicar procesamiento ML
                    ml_enhanced = ml_processor.preprocess_product(
                        product_data, 
                        enable_ml=True
                    )
                    
                    if ml_enhanced.get('ml_processed', False):
                        products_data[i] = ml_enhanced
                        repaired_count += 1
            except Exception as e:
                logger.debug(f"Error procesando producto {i}: {e}")
        
        # Guardar productos reparados
        with open(products_file, 'w', encoding='utf-8') as f:
            json.dump(products_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Embeddings reparados: {repaired_count}/{len(products_data)} productos")
        print(f"💾 Guardado en: {products_file}")
        
    except Exception as e:
        print(f"❌ Error reparando embeddings: {e}")
        import traceback
        traceback.print_exc()

def test_rag_with_ml():
    """Prueba el sistema RAG con ML habilitado."""
    print("\n🧪 PROBANDO RAG CON ML")
    print("="*50)
    
    try:
        from src.core.rag.advanced.WorkingRAGAgent import create_rag_agent
        
        # Crear agente con ML habilitado
        agent = create_rag_agent(mode="ml_enhanced")
        
        # Probar algunas consultas
        test_queries = [
            "smartphone económico",
            "libro de programación",
            "audífonos inalámbricos"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Consulta: '{query}'")
            result = agent.process_query(query)
            
            if result.get('products'):
                print(f"✅ Encontrados: {len(result['products'])} productos")
                for i, product in enumerate(result['products'][:3], 1):
                    title = getattr(product, 'title', 'Sin título')[:50]
                    print(f"  {i}. {title}")
            else:
                print("❌ No se encontraron productos")
        
        print("\n🎉 Prueba completada exitosamente")
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_rag_with_ml()
    else:
        repair_ml_embeddings()