#!/usr/bin/env python3
"""
Reparación definitiva del Retriever
"""
import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

def fix_retriever_completely():
    print("🔧 REPARACIÓN DEFINITIVA DEL RETRIEVER")
    print("=" * 50)
    
    # Limpiar índice existente
    index_path = Path("data/vector_index")
    if index_path.exists():
        shutil.rmtree(index_path)
        print("🗑️  Índice anterior eliminado")
    
    # Cargar datos
    from src.core.data.loader import DataLoader
    loader = DataLoader()
    products = loader.load_data()
    print(f"📦 Productos cargados: {len(products)}")
    
    # Reconstruir índice
    from src.core.rag.basic.retriever import Retriever
    retriever = Retriever()
    
    print("🔧 Construyendo nuevo índice...")
    retriever.build_index(products)
    print("✅ Índice reconstruido")
    
    return retriever

def test_fixed_retriever(retriever):
    print("\n🔍 VERIFICANDO REPARACIÓN")
    print("=" * 50)
    
    test_cases = [
        ("game", "Término general"),
        ("software", "Productos de software"), 
        ("professional", "Palabra en títulos"),
        ("music", "Contenido multimedia"),
        ("beauty", "Productos belleza")
    ]
    
    success_count = 0
    
    for query, description in test_cases:
        try:
            print(f"\n🎯 '{query}' ({description}):")
            results = retriever.retrieve(query, k=3, min_similarity=0.05)
            
            if results:
                print(f"   ✅ {len(results)} resultados")
                success_count += 1
                # Mostrar primer resultado
                product = results[0]
                title = getattr(product, 'title', 'N/A')
                score = getattr(product, 'score', 0)
                print(f"   📝 Ejemplo: {title[:60]}... (score: {score:.3f})")
            else:
                print(f"   ⚠️  0 resultados")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 RESUMEN: {success_count}/{len(test_cases)} búsquedas exitosas")
    return success_count >= 3

if __name__ == "__main__":
    try:
        retriever = fix_retriever_completely()
        
        if test_fixed_retriever(retriever):
            print("\n🎉 ¡RETRIEVER REPARADO EXITOSAMENTE!")
            print("✅ Puedes ejecutar ahora: python test_complete_system.py")
        else:
            print("\n❌ El retriever aún tiene problemas")
            
    except Exception as e:
        print(f"\n💥 Error durante la reparación: {e}")
        import traceback
        traceback.print_exc()