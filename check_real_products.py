# check_real_products.py
import json
from pathlib import Path

def check_real_products():
    """Verifica qué productos reales existen en el sistema"""
    print("🔍 VERIFICANDO PRODUCTOS REALES")
    
    # Verificar archivo de productos procesados
    products_file = Path("data/processed/products.json")
    if products_file.exists():
        with open(products_file, 'r', encoding='utf-8') as f:
            products = json.load(f)
        
        print(f"✅ {len(products)} productos encontrados en products.json")
        print("📋 Primeros 5 productos:")
        for i, product in enumerate(products[:5]):
            print(f"   {i+1}. ID: {product.get('id', 'N/A')}")
            print(f"      Title: {product.get('title', 'N/A')}")
    else:
        print("❌ No se encontró products.json")
        
    # Verificar qué IDs devuelve el retriever
    print("\n🔍 PROBANDO RETRIEVER CON CONSULTAS REALES")
    
    from src.core.rag.basic.retriever import Retriever
    retriever = Retriever()
    
    test_queries = ["playstation", "xbox", "nintendo"]
    for query in test_queries:
        results = retriever.retrieve(query, k=3)
        print(f"📋 Query: '{query}' -> {len(results)} resultados")
        for i, result in enumerate(results[:3]):
            print(f"   {i+1}. {result}")

if __name__ == "__main__":
    check_real_products()