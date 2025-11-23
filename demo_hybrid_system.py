#!/usr/bin/env python3
"""
Demo MEJORADA del sistema híbrido con datos simulados realistas
"""

import logging
import random
from datetime import datetime, timedelta
from src.core.rag.advanced.WorkingRAGAgent import WorkingAdvancedRAGAgent
from src.core.data.user_manager import UserManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simulate_user_feedback(user_manager, user_id, query, response, rating, days_ago=0):
    """Simula feedback histórico para un usuario"""
    try:
        profile = user_manager.get_user_profile(user_id)
        if not profile:
            return False
            
        # Crear timestamp en el pasado
        timestamp = datetime.now() - timedelta(days=days_ago)
        
        # Simular productos seleccionados (en sistema real vendrían de la respuesta)
        sample_products = ["B082XY23D3", "B08H75RTZ8", "B08P3XHZK8"]  # IDs de videojuegos reales
        
        # Actualizar perfil con feedback simulado
        profile.add_feedback_event(
            query=query,
            response=response,
            rating=rating,
            products_shown=sample_products,
            selected_product=random.choice(sample_products)
        )
        
        # Guardar perfil actualizado
        user_manager.save_user_profile(profile)
        return True
        
    except Exception as e:
        logger.error(f"Error simulando feedback: {e}")
        return False

def demo_hybrid_system_enhanced():
    """Demostración MEJORADA con datos simulados realistas"""
    
    print("🎮 DEMO MEJORADA - SISTEMA HÍBRIDO CON DATOS REALISTAS")
    print("=" * 60)
    
    # Inicializar sistema
    rag_agent = WorkingAdvancedRAGAgent()
    user_manager = UserManager()
    
    # Crear usuarios de prueba con diferentes perfiles
    print("\n👥 CREANDO USUARIOS DE PRUEBA CON HISTORIAL...")
    
    # Usuario joven - ama los shooters
    usuario_joven = user_manager.create_user_profile(
        age=20,
        gender="male", 
        country="Spain",
        preferred_categories=["action", "shooter", "fps"],
        preferred_brands=["Activision", "Electronic Arts", "Ubisoft"]
    )
    
    # Usuario adulto - prefiere RPGs
    usuario_adulto = user_manager.create_user_profile(
        age=35,
        gender="male",
        country="Spain", 
        preferred_categories=["rpg", "strategy", "adventure"],
        preferred_brands=["Nintendo", "Square Enix", "CD Projekt"]
    )
    
    print(f"✅ Usuario joven creado: {usuario_joven.user_id}")
    print(f"✅ Usuario adulto creado: {usuario_adulto.user_id}")
    
    # 🔥 NUEVO: Simular historial de feedback REALISTA
    print("\n📝 SIMULANDO HISTORIAL DE FEEDBACK REALISTA...")
    
    # Usuario joven - feedback positivo en shooters
    shooters_queries = [
        "call of duty modern warfare",
        "juegos de guerra moderna", 
        "fps multijugador online",
        "battlefield 2042",
        "juegos de disparos en primera persona"
    ]
    
    for i, query in enumerate(shooters_queries):
        simulate_user_feedback(
            user_manager, usuario_joven.user_id,
            query=query,
            response=f"Recomendaciones de shooters para {query}",
            rating=random.randint(4, 5),  # Feedback positivo
            days_ago=random.randint(1, 30)  # En último mes
        )
    
    # Usuario adulto - feedback positivo en RPGs
    rpg_queries = [
        "final fantasy xvi",
        "juegos de rol aventura",
        "the witcher 3",
        "zelda breath of the wild", 
        "rpg mundo abierto"
    ]
    
    for i, query in enumerate(rpg_queries):
        simulate_user_feedback(
            user_manager, usuario_adulto.user_id,
            query=query,
            response=f"Recomendaciones de RPGs para {query}",
            rating=random.randint(4, 5),  # Feedback positivo
            days_ago=random.randint(1, 30)
        )
    
    print("✅ Historial de feedback simulado exitosamente")
    
    # Test del sistema híbrido con usuario nuevo similar al joven
    print("\n🎯 TEST 1: NUEVO USUARIO SIMILAR AL JOVEN (BUSCA SHOOTERS)")
    
    test_user_1 = user_manager.create_user_profile(
        age=22, 
        gender="male",
        country="Spain",
        preferred_categories=["action", "fps", "shooter"]
    )
    
    response_1 = rag_agent.process_query("juegos de guerra modernos", test_user_1.user_id)
    
    print(f"🤖 RESPUESTA TEST 1:")
    print(f"   Productos recomendados: {len(response_1.products)}")
    print(f"   Score de calidad: {response_1.quality_score}")
    
    # Mostrar algunos productos recomendados
    if response_1.products:
        print(f"   Ejemplo producto: {response_1.products[0].title}")
    
    # Test 2: Usuario similar al adulto busca RPGs
    print("\n🎯 TEST 2: NUEVO USUARIO SIMILAR AL ADULTO (BUSCA RPGS)")
    
    test_user_2 = user_manager.create_user_profile(
        age=32,
        gender="male", 
        country="Spain",
        preferred_categories=["rpg", "adventure"]
    )
    
    response_2 = rag_agent.process_query("juegos de rol aventura", test_user_2.user_id)
    
    print(f"🤖 RESPUESTA TEST 2:")
    print(f"   Productos recomendados: {len(response_2.products)}")
    print(f"   Score de calidad: {response_2.quality_score}")
    
    if response_2.products:
        print(f"   Ejemplo producto: {response_2.products[0].title}")
    
    # Estadísticas finales mejoradas
    print("\n📊 ESTADÍSTICAS FINALES MEJORADAS:")
    stats = user_manager.get_demographic_stats()
    print(f"   Total usuarios: {stats['total_users']}")
    print(f"   Distribución por edad: {stats['age_distribution']}")
    print(f"   Distribución por género: {stats['gender_distribution']}")
    print(f"   Búsquedas totales: {stats['total_searches']}")
    print(f"   Feedbacks totales: {stats['total_feedbacks']}")
    
    # Métricas del sistema híbrido
    print("\n⚡ MÉTRICAS DEL SISTEMA HÍBRIDO:")
    print(f"   Pesos actuales: {rag_agent.hybrid_weights}")
    print(f"   Umbral similitud: {rag_agent.min_similarity_threshold}")
    
    print("\n🎉 DEMO MEJORADA COMPLETADA - Sistema híbrido funcionando con datos realistas!")

if __name__ == "__main__":
    demo_hybrid_system_enhanced()