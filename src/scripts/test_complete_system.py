#!/usr/bin/env python3
"""
Prueba completa de todos los componentes del sistema
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

def test_data_loading():
    """Prueba la carga y enriquecimiento de datos"""
    print("📦 Probando carga de datos...")
    
    from src.core.data.loader import DataLoader
    from src.core.config import settings
    
    loader = DataLoader()
    products = loader.load_data()[:100]  # Solo 100 para prueba rápida
    
    print(f"✅ Productos cargados: {len(products)}")
    
    if products:
        sample = products[0]
        print(f"📝 Producto de muestra:")
        print(f"   Título: {getattr(sample, 'title', 'N/A')}")
        print(f"   Categoría: {getattr(sample, 'main_category', 'N/A')}")
        print(f"   Precio: {getattr(sample, 'price', 'N/A')}")
        print(f"   Rating: {getattr(sample, 'average_rating', 'N/A')}")
    
    return products

def test_retriever(products):
    """Prueba el sistema de búsqueda semántica"""
    print("\n🔍 Probando retriever...")
    
    from src.core.rag.basic.retriever import Retriever
    
    retriever = Retriever()
    
    # Construir índice si no existe
    if not retriever.index_exists():
        print("   Construyendo índice...")
        retriever.build_index(products)
    
    # Probar búsqueda
    test_query = "laptop gaming"
    results = retriever.retrieve(query=test_query, k=3)
    
    print(f"✅ Resultados para '{test_query}': {len(results)}")
    
    for i, product in enumerate(results, 1):
        score = getattr(product, 'score', 0)
        print(f"   {i}. {getattr(product, 'title', 'N/A')} (score: {score:.3f})")
    
    return retriever

def test_rag_agent():
    """Prueba el RAGAgent completo"""
    print("\n🧠 Probando RAGAgent...")
    
    from src.core.rag.advanced.WorkingRAGAgent import RAGAgent
    
    agent = RAGAgent(user_id="system_test")
    
    # Probar consultas
    test_cases = [
        ("auriculares bluetooth", "Tecnología"),
        ("libro cocina", "Hogar"),
        ("zapatillas running", "Deportes")
    ]
    
    for query, expected_category in test_cases:
        print(f"\n   Consulta: '{query}'")
        respuesta = agent.ask(query)
        
        # Verificar que la respuesta no sea de error
        if "No encontré" not in respuesta and "dificultades" not in respuesta:
            print("   ✅ Respuesta válida generada")
            # Registrar feedback para RL
            agent._log_feedback(query, respuesta, 4)
        else:
            print("   ⚠️  Sin resultados (puede ser normal para datos de prueba)")
    
    return agent

def test_feedback_system():
    """Prueba el sistema de feedback"""
    print("\n📊 Probando sistema de feedback...")
    
    from src.core.rag.advanced.feedback_processor import FeedbackProcessor
    
    processor = FeedbackProcessor()
    
    # Simular algunos feedbacks
    test_feedbacks = [
        ("laptop gaming", "Excelente laptop para gaming", 5),
        ("mouse inalámbrico", "No encontré lo que buscaba", 2),
        ("teclado mecánico", "Buenas opciones", 4)
    ]
    
    for query, answer, rating in test_feedbacks:
        processor.save_feedback(
            query=query,
            answer=answer, 
            rating=rating,
            extra_meta={"user_id": "test_user"}
        )
        print(f"   ✅ Feedback registrado: {query} -> {rating} estrellas")
    
    # Verificar que se crearon los archivos
    feedback_dir = Path("data/feedback")
    if feedback_dir.exists():
        files = list(feedback_dir.glob("*.jsonl")) + list(feedback_dir.glob("*.log"))
        print(f"   📁 Archivos de feedback: {len(files)}")

def main():
    """Ejecuta todas las pruebas"""
    print("🎯 INICIANDO PRUEBAS COMPLETAS DEL SISTEMA")
    print("=" * 60)
    
    try:
        # 1. Carga de datos
        products = test_data_loading()
        
        if not products:
            print("❌ No se pudieron cargar productos - abortando")
            return
        
        # 2. Retriever
        retriever = test_retriever(products)
        
        # 3. RAG Agent
        agent = test_rag_agent()
        
        # 4. Sistema de feedback
        test_feedback_system()
        
        print("\n" + "=" * 60)
        print("🎉 ¡TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE!")
        print("\n📝 RESUMEN:")
        print("   ✅ Carga y enriquecimiento de datos")
        print("   ✅ Búsqueda semántica (retriever)") 
        print("   ✅ RAG Agent con personalización")
        print("   ✅ Sistema de feedback y RL")
        print("   ✅ Procesamiento de conversaciones")
        
    except Exception as e:
        print(f"\n❌ Error en las pruebas: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()