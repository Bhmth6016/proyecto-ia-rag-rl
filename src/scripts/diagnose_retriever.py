#!/usr/bin/env python3
# scripts/diagnose_retriever.py

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.core.rag.basic.retriever import Retriever
from src.core.config import settings
from src.core.utils.logger import get_logger

logger = get_logger(__name__)

def diagnose_retriever():
    """Diagnóstico completo del retriever"""
    print("🔧 DIAGNÓSTICO DEL RETRIEVER")
    print("=" * 50)
    
    # 1. Verificar configuración
    print("1. 📋 Verificando configuración...")
    print(f"   • Chroma path: {settings.VECTOR_INDEX_PATH}")
    print(f"   • Existe: {Path(settings.VECTOR_INDEX_PATH).exists()}")
    print(f"   • Embedding model: {settings.EMBEDDING_MODEL}")
    
    # 2. Inicializar retriever
    print("2. 🔄 Inicializando retriever...")
    try:
        retriever = Retriever()
        print("   ✅ Retriever inicializado")
        
        # 3. Verificar índice
        print("3. 🔍 Verificando índice...")
        if retriever.index_exists():
            print("   ✅ Índice existe")
            
            # 4. Probar búsqueda
            print("4. 🧪 Probando búsquedas...")
            test_queries = [
                "laptop", 
                "auriculares bluetooth",
                "libro python"
            ]
            
            for query in test_queries:
                try:
                    results = retriever.retrieve(query, k=2)
                    print(f"   • '{query}': {len(results)} resultados")
                    
                    if results:
                        for i, product in enumerate(results[:1]):
                            print(f"     {i+1}. {product.title[:50]}...")
                    
                except Exception as e:
                    print(f"   ❌ Error en '{query}': {e}")
            
            # 5. Estadísticas
            print("5. 📊 Obteniendo estadísticas...")
            try:
                stats = retriever.get_index_stats() if hasattr(retriever, 'get_index_stats') else {}
                print(f"   • Estadísticas: {stats}")
            except Exception as e:
                print(f"   ⚠️  No se pudieron obtener estadísticas: {e}")
                
        else:
            print("   ❌ Índice no existe")
            print("   💡 Ejecuta: python main.py index --force")
            
    except Exception as e:
        print(f"   ❌ Error inicializando retriever: {e}")
        return False
    
    print("=" * 50)
    print("🎉 DIAGNÓSTICO COMPLETADO")
    return True

if __name__ == "__main__":
    diagnose_retriever()