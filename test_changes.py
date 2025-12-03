#!/usr/bin/env python3
"""
Retriever simple para pruebas sin depender de config.py roto.
"""
import json
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any
from difflib import SequenceMatcher

from src.core.data.product import Product
from src.core.data.product_reference import ProductReference

class SimpleRetriever:
    """Retriever simplificado para pruebas."""
    
    def __init__(self):
        self.mock_products = self._create_mock_products()
        print("✅ SimpleRetriever inicializado con productos mock")
    
    def _create_mock_products(self) -> List[Product]:
        """Crea productos mock para pruebas."""
        products = []
        
        mock_data = [
            {
                "id": "PROD001",
                "title": "iPhone 15 Pro Max 256GB",
                "price": 1299.99,
                "description": "Smartphone Apple con cámara profesional",
                "category": "Electronics"
            },
            {
                "id": "PROD002", 
                "title": "Samsung Galaxy S24 Ultra",
                "price": 1199.99,
                "description": "Teléfono Android con S-Pen",
                "category": "Electronics"
            },
            {
                "id": "PROD003",
                "title": "Google Pixel 8 Pro",
                "price": 999.99,
                "description": "Teléfono Google con IA integrada",
                "category": "Electronics"
            },
            {
                "id": "PROD004",
                "title": "Auriculares Sony WH-1000XM5",
                "price": 349.99,
                "description": "Auriculares noise cancelling premium",
                "category": "Electronics"
            },
            {
                "id": "PROD005",
                "title": "MacBook Air M3",
                "price": 1099.99,
                "description": "Laptop Apple ultra delgada",
                "category": "Computers"
            }
        ]
        
        for data in mock_data:
            try:
                product = Product.from_dict(data)
                products.append(product)
            except Exception as e:
                print(f"⚠️  Error creando producto mock {data['id']}: {e}")
        
        return products
    
    def retrieve(self, query: str, k: int = 5) -> List[Product]:
        """
        Retrieve products based on query.
        DEVUELVE objetos Product, no strings.
        """
        print(f"\n🔍 SimpleRetriever.retrieve(query='{query}', k={k})")
        
        if not self.mock_products:
            print("⚠️  No hay productos mock disponibles")
            return []
        
        # Simular scoring basado en similitud de texto
        scored = []
        for product in self.mock_products:
            score = self._score_query(query, product)
            scored.append((score, product))
        
        # Ordenar por score
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Tomar los k mejores
        results = [p for _, p in scored[:k]]
        
        print(f"✅ Devolviendo {len(results)} objetos Product:")
        for i, product in enumerate(results):
            print(f"   {i+1}. [{product.id}] {product.title} (score: {scored[i][0]:.2f})")
        
        return results
    
    def _score_query(self, query: str, product: Product) -> float:
        """Calcula score de similitud entre query y producto."""
        query_lower = query.lower()
        title_lower = product.title.lower() if hasattr(product, 'title') else ""
        
        # Similitud basada en título
        title_sim = SequenceMatcher(None, query_lower, title_lower).ratio()
        
        # Bonus si la categoría coincide
        category_bonus = 0.0
        if hasattr(product, 'category') and product.category:
            if any(word in query_lower for word in product.category.lower().split()):
                category_bonus = 0.2
        
        return min(1.0, title_sim + category_bonus)
    
    def retrieve_as_product_reference(self, query: str, k: int = 5) -> List[ProductReference]:
        """Versión que devuelve ProductReference."""
        products = self.retrieve(query, k)
        
        references = []
        for product in products:
            # Simular score (en realidad viene de retrieve)
            score = self._score_query(query, product)
            pref = ProductReference.from_product(product, score=score, source="simple")
            references.append(pref)
        
        print(f"✅ Convertidos a {len(references)} ProductReference")
        return references

def test_retriever_fix():
    """Test del problema original: debe devolver Products, no strings."""
    print("🚀 TEST: Retriever debe devolver Products, no strings")
    print("=" * 70)
    
    try:
        retriever = SimpleRetriever()
        
        # Test 1: Devuelve objetos Product
        print("\n🧪 Test 1: Retriever.retrieve() devuelve Product objects")
        results = retriever.retrieve("iPhone smartphone", k=3)
        
        assert len(results) > 0, "No se devolvieron resultados"
        
        for i, result in enumerate(results):
            print(f"  Resultado {i+1}:")
            print(f"    Tipo: {type(result).__name__}")
            print(f"    Es string? {isinstance(result, str)}")
            print(f"    Tiene 'id': {hasattr(result, 'id')}")
            print(f"    Tiene 'title': {hasattr(result, 'title')}")
            print(f"    ID: {getattr(result, 'id', 'N/A')}")
            print(f"    Título: {getattr(result, 'title', 'N/A')}")
            
            # Verificaciones críticas
            assert not isinstance(result, str), f"ERROR: Resultado {i+1} es string: {result}"
            assert hasattr(result, 'id'), f"ERROR: Resultado {i+1} no tiene 'id'"
            assert hasattr(result, 'title'), f"ERROR: Resultado {i+1} no tiene 'title'"
        
        print("✅ Test 1 PASADO: Retriever devuelve objetos Product")
        
        # Test 2: Convertir a ProductReference
        print("\n🧪 Test 2: Conversión a ProductReference")
        references = retriever.retrieve_as_product_reference("laptop computer", k=2)
        
        for i, pref in enumerate(references):
            print(f"  ProductReference {i+1}:")
            print(f"    ID: {pref.id}")
            print(f"    Title: {pref.title}")
            print(f"    Score: {pref.score:.2f}")
            print(f"    Source: {pref.source}")
        
        print("✅ Test 2 PASADO: Conversión a ProductReference funciona")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        return False

def inspect_real_retriever():
    """Inspecciona el retriever real si se puede importar."""
    print("\n🔍 Intentando importar retriever real...")
    
    try:
        # Intentar arreglar config primero
        import sys
        import os
        
        # Añadir configuración temporal al entorno
        os.environ['VECTOR_INDEX_PATH'] = 'data/processed/chroma_db'
        
        from src.core.rag.basic.retriever import Retriever
        
        print("✅ Retriever real importado correctamente")
        
        # Verificar si hay índice
        retriever = Retriever()
        
        if retriever.index_exists():
            print("✅ Índice Chroma existe")
            
            # Probar consulta simple
            results = retriever.retrieve("test", k=2)
            print(f"Retriever real devolvió {len(results)} resultados")
            
            if results:
                print("Tipo del primer resultado:", type(results[0]).__name__)
        else:
            print("⚠️  No hay índice Chroma")
            
    except Exception as e:
        print(f"❌ No se pudo usar retriever real: {e}")
        print("💡 Usando SimpleRetriever para pruebas")

if __name__ == "__main__":
    print("🚀 PRUEBA DE CORRECCIÓN DEL PROBLEMA 1")
    print("=" * 70)
    
    # Ejecutar test con SimpleRetriever
    if test_retriever_fix():
        print("\n" + "=" * 70)
        print("🎉 ¡PROBLEMA 1 RESUELTO CONCEPTUALMENTE!")
        print("\n📌 CONCLUSIÓN:")
        print("1. Retriever DEBE devolver objetos Product (no strings)")
        print("2. Tu código YA lo hace: 'return [p for _, p in scored[:k]]'")
        print("3. El problema real es config.py (Settings no tiene atributos)")
        
        # Preguntar si probar retriever real
        response = input("\n¿Probar a arreglar config.py e importar retriever real? (s/n): ")
        if response.lower() in ['s', 'si', 'sí', 'y', 'yes']:
            inspect_real_retriever()
    else:
        print("\n" + "=" * 70)
        print("❌ Hay problemas con el test")
    
    print("\n" + "=" * 70)
    print("📌 SIGUIENTES PASOS:")
    print("1. Arregla config.py (usa el código que te di)")
    print("2. Luego WorkingRAGAgent.py (UserManager dummy)")
    print("3. Luego collaborative_filter.py (pesos dinámicos)")