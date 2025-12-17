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
        
        # Verificar configuración ML
        if not settings.ML_ENABLED:
            print("⚠️  ML está deshabilitado en configuración")
            print("💡 Habilitando ML temporalmente...")
            settings.update_ml_settings(ml_enabled=True)
        
        print(f"🔧 Configuración ML:")
        print(f"   • ML habilitado: {settings.ML_ENABLED}")
        print(f"   • Device: {settings.DEVICE}")
        print(f"   • ML use GPU: {settings.ML_USE_GPU}")
        
        # Crear preprocesador ML - CORREGIDO: usar ml_use_gpu según configuración
        ml_processor = ProductDataPreprocessor(
            verbose=True,
            use_gpu=settings.ML_USE_GPU  # CORREGIDO: usar ML_USE_GPU de configuración
        )
        
        # Procesar productos con ML
        print("🤖 Aplicando procesamiento ML...")
        repaired_count = 0
        
        batch_size = 50  # Procesar en lotes para mejor rendimiento
        
        for batch_start in range(0, len(products_data), batch_size):
            batch_end = min(batch_start + batch_size, len(products_data))
            batch = products_data[batch_start:batch_end]
            
            print(f"  Procesando lote {batch_start+1}-{batch_end}/{len(products_data)}...")
            
            for i in range(len(batch)):
                idx = batch_start + i
                product_data = batch[i]
                
                try:
                    # Verificar si ya tiene ML
                    if not product_data.get('ml_processed', False):
                        # Aplicar procesamiento ML
                        ml_enhanced = ml_processor.preprocess_product(
                            product_data, 
                            enable_ml=True
                        )
                        
                        if ml_enhanced.get('ml_processed', False):
                            products_data[idx] = ml_enhanced
                            repaired_count += 1
                            
                            # Mostrar progreso cada 10 productos reparados
                            if repaired_count % 10 == 0:
                                print(f"    ✅ Reparados: {repaired_count}")
                except Exception as e:
                    logger.debug(f"Error procesando producto {idx}: {e}")
                    # Continuar con siguiente producto
                    continue
        
        # Verificar si hay cambios para guardar
        if repaired_count > 0:
            # Crear backup primero
            backup_file = products_file.with_suffix('.json.backup_ml')
            try:
                with open(products_file, 'r', encoding='utf-8') as src:
                    with open(backup_file, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
                print(f"\n📦 Backup creado: {backup_file}")
            except Exception as e:
                print(f"⚠️  Error creando backup: {e}")
            
            # Guardar productos reparados
            with open(products_file, 'w', encoding='utf-8') as f:
                json.dump(products_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ Embeddings reparados: {repaired_count}/{len(products_data)} productos")
            print(f"💾 Guardado en: {products_file}")
            
            # Estadísticas
            ml_enabled_count = sum(1 for p in products_data if p.get('ml_processed', False))
            print(f"📊 Estadísticas:")
            print(f"   • Productos con ML: {ml_enabled_count}/{len(products_data)}")
            print(f"   • Nuevos ML agregados: {repaired_count}")
            
            # Mostrar algunos ejemplos de productos reparados
            print(f"\n🎯 Ejemplos de productos reparados:")
            ml_products = [p for p in products_data if p.get('ml_processed', False)]
            for i, product in enumerate(ml_products[:3], 1):
                title = product.get('title', 'Sin título')[:60]
                category = product.get('main_category', 'Sin categoría')
                print(f"  {i}. '{title}...' - {category}")
        else:
            print(f"\n💡 No se necesitaron reparaciones: {len(products_data)} productos ya tienen ML")
            
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
        
        # Asegurarse de que ML esté habilitado
        if not settings.ML_ENABLED:
            settings.update_ml_settings(ml_enabled=True)
            print("✅ ML habilitado para la prueba")
        
        # Crear agente con ML habilitado
        print("🤖 Creando agente RAG...")
        agent = create_rag_agent(mode="ml_enhanced")
        
        # Probar algunas consultas
        test_queries = [
            "smartphone económico",
            "libro de programación",
            "audífonos inalámbricos",
            "videojuego de aventura"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Consulta: '{query}'")
            try:
                result = agent.process_query(query)
                
                if result.get('products'):
                    print(f"✅ Encontrados: {len(result['products'])} productos")
                    for i, product in enumerate(result['products'][:3], 1):
                        if isinstance(product, dict):
                            title = product.get('title', 'Sin título')[:50]
                            ml = product.get('ml_processed', False)
                            ml_mark = "🤖" if ml else "  "
                        else:
                            title = getattr(product, 'title', 'Sin título')[:50]
                            ml_mark = "🤖" if hasattr(product, 'ml_processed') and getattr(product, 'ml_processed', False) else "  "
                        print(f"  {i}. {ml_mark} {title}")
                else:
                    print("❌ No se encontraron productos")
            except Exception as e:
                print(f"⚠️  Error procesando consulta: {e}")
        
        print("\n🎉 Prueba completada exitosamente")
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        import traceback
        traceback.print_exc()

def check_ml_status():
    """Verifica el estado de ML en los productos."""
    print("\n🔍 VERIFICANDO ESTADO ML")
    print("="*50)
    
    try:
        products_file = settings.PROC_DIR / "products.json"
        
        if not products_file.exists():
            print(f"❌ Archivo no encontrado: {products_file}")
            return
        
        with open(products_file, 'r', encoding='utf-8') as f:
            products_data = json.load(f)
        
        total_products = len(products_data)
        ml_processed = sum(1 for p in products_data if p.get('ml_processed', False))
        has_embeddings = sum(1 for p in products_data if p.get('ml_embedding') is not None)
        has_entities = sum(1 for p in products_data if p.get('ml_entities'))
        
        print(f"📊 Estado ML de {total_products} productos:")
        print(f"   • Con ML procesado: {ml_processed} ({ml_processed/total_products*100:.1f}%)")
        print(f"   • Con embeddings: {has_embeddings} ({has_embeddings/total_products*100:.1f}%)")
        print(f"   • Con entidades: {has_entities} ({has_entities/total_products*100:.1f}%)")
        
        # Mostrar algunos productos sin ML
        if ml_processed < total_products:
            no_ml = [p for p in products_data if not p.get('ml_processed', False)]
            print(f"\n⚠️  {len(no_ml)} productos SIN ML:")
            for i, product in enumerate(no_ml[:5], 1):
                title = product.get('title', 'Sin título')[:60]
                category = product.get('main_category', 'Sin categoría')
                print(f"  {i}. '{title}...' - {category}")
            
            if len(no_ml) > 5:
                print(f"  ... y {len(no_ml)-5} más")
                
    except Exception as e:
        print(f"❌ Error verificando estado: {e}")

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_rag_with_ml()
        elif sys.argv[1] == "check":
            check_ml_status()
        elif sys.argv[1] == "help":
            print("Uso: python repair_ml_embeddings.py [comando]")
            print("\nComandos:")
            print("  (sin comando)   Repara embeddings ML")
            print("  test            Prueba RAG con ML")
            print("  check           Verifica estado ML")
            print("  help            Muestra esta ayuda")
    else:
        repair_ml_embeddings()