#!/usr/bin/env python3
"""
Prueba del RAGAgent sin el retriever problemático
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_rag_with_mock_retriever():
    """Prueba el RAGAgent con un retriever simulado"""
    print("🧠 PROBANDO RAGAgent CON RETRIEVER SIMULADO")
    print("=" * 50)
    
    from src.core.rag.advanced.RAGAgent import RAGAgent
    from src.core.data.product import Product
    from src.core.data.loader import DataLoader
    
    # Cargar algunos productos reales para las pruebas
    loader = DataLoader()
    real_products = loader.load_data()[:20]
    
    # Crear agente
    agent = RAGAgent(user_id="direct_test")
    
    # Sobrescribir el retriever problemático con uno simulado
    class MockRetriever:
        def retrieve(self, query, k=5, min_similarity=0.0):
            # Devolver productos de ejemplo basados en la query
            filtered = []
            query_lower = query.lower()
            
            for product in real_products:
                title = getattr(product, 'title', '').lower()
                category = getattr(product, 'main_category', '').lower()
                
                # Simular matching básico
                if any(term in title or term in category 
                      for term in query_lower.split()):
                    # Simular score
                    product.score = 0.8
                    filtered.append(product)
            
            return filtered[:k]
    
    # Reemplazar el retriever
    agent.retriever = MockRetriever()
    
    # Probar consultas
    test_queries = [
        "electronics",
        "games", 
        "music",
        "books"
    ]
    
    for query in test_queries:
        print(f"\n📝 Consulta: '{query}'")
        print("-" * 40)
        
        try:
            respuesta = agent.ask(query)
            print(f"🤖 Respuesta:\n{respuesta}")
            
            # Registrar feedback
            agent._log_feedback(query, respuesta, 4)
            print("✅ Feedback registrado")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_rag_with_mock_retriever()