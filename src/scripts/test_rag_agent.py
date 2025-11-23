#!/usr/bin/env python3
"""
Script de prueba para el RAGAgent sin interfaz gráfica
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.rag.advanced.WorkingRAGAgent import RAGAgent
from src.core.data.loader import DataLoader
from src.core.init import get_system

def test_basic_queries():
    """Prueba consultas básicas con el RAGAgent"""
    print("🚀 Iniciando prueba del RAGAgent...")
    
    try:
        # Inicializar sistema
        system = get_system()
        
        # Crear agente
        agent = RAGAgent(user_id="test_user")
        
        # Consultas de prueba
        test_queries = [
            "videojuegos",
            "juegos pc", 
            "simulador vuelo",
            "addon profesional",
            "juegos descargables"
        ]
        
        print("\n" + "="*50)
        print("🧪 PROBANDO CONSULTAS:")
        print("="*50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Consulta {i}: '{query}'")
            print("-" * 40)
            
            respuesta = agent.ask(query)
            print(f"🤖 Respuesta:\n{respuesta}")
            
            # Simular feedback positivo
            agent._log_feedback(query, respuesta, 5)
            print("✅ Feedback registrado")
            
        print("\n🎉 ¡Prueba completada!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_chat_loop():
    """Prueba el bucle de chat interactivo"""
    print("\n💬 Iniciando modo chat interactivo...")
    
    try:
        agent = RAGAgent(user_id="test_user")
        agent.chat_loop()
    except Exception as e:
        print(f"❌ Error en chat loop: {e}")

if __name__ == "__main__":
    print("🎯 PRUEBAS DEL SISTEMA RAG")
    print("1. Prueba automática de consultas")
    print("2. Chat interactivo")
    
    opcion = input("\nSelecciona opción (1/2): ").strip()
    
    if opcion == "1":
        test_basic_queries()
    elif opcion == "2":
        test_chat_loop()
    else:
        print("❌ Opción no válida")