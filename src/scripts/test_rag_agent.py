# src/scripts/test_final_system.py
from src.core.rag.advanced.WorkingRAGAgent import WorkingAdvancedRAGAgent
import logging

logging.basicConfig(level=logging.INFO)

def test_final_system():
    print("🎮 SISTEMA RAG FINAL - VIDEOJUEGOS")
    print("=" * 60)
    print("✅ Sin dependencias externas | ✅ Optimizado para gaming")
    print("=" * 60)
    
    agent = WorkingAdvancedRAGAgent()
    
    test_cases = [
        # (consulta, descripción)
        ("juegos de playstation 5", "Búsqueda por plataforma específica"),
        ("nintendo switch aventura rpg", "Búsqueda por plataforma y género"),
        ("xbox one shooters", "Búsqueda por plataforma y género"),
        ("minecraft edition", "Búsqueda por título específico"), 
        ("zelda breath of the wild", "Búsqueda por título famoso"),
        ("juegos de deportes baratos", "Búsqueda con filtro de precio"),
        ("acción y aventura", "Búsqueda por múltiples géneros"),
    ]
    
    for query, description in test_cases:
        print(f"\n🎯 '{query}'")
        print(f"📝 {description}")
        print("-" * 50)
        
        agent.clear_memory()
        response = agent.process_query(query)
        
        print(f"✅ Calidad: {response.quality_score:.2f}")
        print(f"📦 Juegos encontrados: {len(response.products)}")
        print(f"🤖 LLM externo: {response.used_llm}")
        
        if response.products:
            print("\n🎮 PLATAFORMAS ENCONTRADAS:")
            platforms = {}
            for product in response.products:
                title = getattr(product, 'title', '')
                if 'playstation' in title.lower():
                    platforms['PlayStation'] = platforms.get('PlayStation', 0) + 1
                elif 'xbox' in title.lower():
                    platforms['Xbox'] = platforms.get('Xbox', 0) + 1  
                elif 'nintendo' in title.lower():
                    platforms['Nintendo'] = platforms.get('Nintendo', 0) + 1
                elif 'pc' in title.lower():
                    platforms['PC'] = platforms.get('PC', 0) + 1
                else:
                    platforms['Otras'] = platforms.get('Otras', 0) + 1
            
            for platform, count in platforms.items():
                print(f"   {platform}: {count} juegos")
        
        print(f"\n💬 RESPUESTA (primeras 2 líneas):")
        lines = response.answer.split('\n')[:2]
        for line in lines:
            print(f"   {line}")

if __name__ == "__main__":
    test_final_system()