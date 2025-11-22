#!/usr/bin/env python3
"""
Verificación después de la reparación
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_fix():
    print("🔍 VERIFICANDO REPARACIÓN DEL RETRIEVER")
    print("=" * 50)
    
    from src.core.rag.basic.retriever import Retriever
    
    retriever = Retriever()
    
    # Test con queries que deberían funcionar
    test_cases = [
        ("game", "Término general"),
        ("software", "Productos de software"), 
        ("professional", "Palabra en títulos"),
        ("add-on", "Productos adicionales"),
        ("music", "Contenido multimedia")
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
    
    if success_count >= 3:
        print("🎉 ¡Retriever funcionando correctamente!")
        return True
    else:
        print("❌ El retriever aún tiene problemas")
        return False

if __name__ == "__main__":
    if verify_fix():
        print("\n✅ Puedes ejecutar ahora: python test_complete_system.py")
    else:
        print("\n🔧 Ejecuta primero: python fix_retriever.py")