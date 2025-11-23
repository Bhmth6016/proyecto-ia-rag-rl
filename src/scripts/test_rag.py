# scripts/test_rag.py
from src.core.rag.advanced.WorkingRAGAgent import WorkingAdvancedRAGAgent

def test_rag_agent():
    print("🧪 Testing WorkingAdvancedRAGAgent...")
    
    agent = WorkingAdvancedRAGAgent()
    
    test_queries = [
        "laptop para programar",
        "auriculares inalámbricos",
        "libros de python",
        "productos de belleza",
        "something that doesn't exist"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"🔍 Testing: '{query}'")
        
        response = agent.process_query(query)
        
        print(f"✅ Quality: {response.quality_score:.2f}")
        print(f"✅ Products: {len(response.products)}")
        print(f"✅ Answer length: {len(response.answer)} chars")
        
        # Mostrar primeros 200 caracteres de la respuesta
        preview = response.answer[:200] + "..." if len(response.answer) > 200 else response.answer
        print(f"📝 Preview: {preview}")

if __name__ == "__main__":
    test_rag_agent()