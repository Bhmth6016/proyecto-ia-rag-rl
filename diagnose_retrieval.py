# diagnose_retrieval.py
import sys
import os
sys.path.append(os.getcwd())

from src.core.rag.basic.retriever import Retriever

def diagnose_retrieval():
    print("🔍 DIAGNÓSTICO DEL SISTEMA DE RECUPERACIÓN")
    
    # 1. Inicializar retriever
    retriever = Retriever()
    print("✅ Retriever inicializado")
    
    # 2. Verificar si el índice existe
    if retriever.index_exists():
        print("✅ Índice Chroma encontrado")
        
        # 3. Probar consultas simples
        test_queries = [
            "playstation",
            "xbox", 
            "nintendo switch",
            "auriculares gaming",
            "teclado mecánico"
        ]
        
        for query in test_queries:
            results = retriever.retrieve(query, k=5)
            print(f"🔍 Query: '{query}' -> {len(results)} resultados: {results[:3]}")
            
    else:
        print("❌ NO se encontró índice Chroma")
        print("💡 Ejecuta primero: python -m src.core.data.loader")

if __name__ == "__main__":
    diagnose_retrieval()