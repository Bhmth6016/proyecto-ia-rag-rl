#!/usr/bin/env python3
"""
Genera feedback rápido interactuando con el sistema
"""

import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).parent))

from src.core.rag.advanced.WorkingRAGAgent import WorkingAdvancedRAGAgent

def generate_quick_feedback():
    """Genera feedback rápido con consultas variadas"""
    
    print("🚀 Generando feedback rápido para reentrenamiento...")
    
    agent = WorkingAdvancedRAGAgent()
    
    # Consultas para generar feedback diverso
    test_queries = [
        ("juegos nintendo switch", 5),
        ("videojuegos de deportes", 3), 
        ("rpg para pc", 4),
        ("juegos de acción baratos", 2),
        ("nuevos lanzamientos ps5", 5)
    ]
    
    for query, rating in test_queries:
        print(f"\n🔍 Consulta: '{query}'")
        try:
            response = agent.process_query(query, "training_user")
            print(f"🤖 Respuesta: {len(response.products)} productos | Calidad: {response.quality_score}")
            
            # Guardar feedback
            agent.log_feedback(query, response.answer, rating, "training_user")
            print(f"📝 Feedback guardado: {rating}/5")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎯 Feedback generado: {len(test_queries)} ejemplos")
    print("✅ El sistema se reentrenará automáticamente")

if __name__ == "__main__":
    generate_quick_feedback()