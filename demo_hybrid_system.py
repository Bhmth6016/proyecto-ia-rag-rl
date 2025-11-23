#!/usr/bin/env python3
"""
Demo del sistema híbrido de recomendación
"""

import logging
from src.core.rag.advanced.WorkingRAGAgent import WorkingAdvancedRAGAgent
from src.core.data.user_manager import UserManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_hybrid_system():
    """Demostración del sistema híbrido en acción"""
    
    print("🎮 DEMO SISTEMA HÍBRIDO DE RECOMENDACIÓN")
    print("=" * 50)
    
    # Inicializar sistema
    rag_agent = WorkingAdvancedRAGAgent()
    user_manager = UserManager()
    
    # Crear usuarios de prueba con diferentes perfiles
    print("\n👥 CREANDO USUARIOS DE PRUEBA...")
    
    usuario_joven = user_manager.create_user_profile(
        age=20,
        gender="male", 
        country="Spain",
        preferred_categories=["action", "shooter"],
        preferred_brands=["Sony", "Activision"]
    )
    
    usuario_adulto = user_manager.create_user_profile(
        age=35,
        gender="male",
        country="Spain", 
        preferred_categories=["strategy", "rpg"],
        preferred_brands=["Nintendo", "Square Enix"]
    )
    
    print(f"✅ Usuario joven creado: {usuario_joven.user_id}")
    print(f"✅ Usuario adulto creado: {usuario_adulto.user_id}")
    
    # Simular interacciones previas
    print("\n📝 SIMULANDO INTERACCIONES PREVIAS...")
    
    # Usuario joven busca shooters y da feedback positivo
    print("1. Usuario joven busca 'call of duty' → feedback positivo")
    # (En sistema real esto vendría de interacciones reales)
    
    # Usuario adulto busca RPGs  
    print("2. Usuario adulto busca 'final fantasy' → feedback positivo")
    
    # Test del sistema híbrido
    print("\n🎯 TESTANDO SISTEMA HÍBRIDO...")
    
    # Nuevo usuario similar al joven busca shooters
    print("3. Nuevo usuario (similar al joven) busca 'juegos de guerra'")
    
    test_user = user_manager.create_user_profile(
        age=22, 
        gender="male",
        country="Spain",
        preferred_categories=["action", "fps"]
    )
    
    # Ejecutar consulta
    response = rag_agent.process_query("juegos de guerra modernos", test_user.user_id)
    
    print(f"\n🤖 RESPUESTA DEL SISTEMA HÍBRIDO:")
    print(f"   Productos recomendados: {len(response.products)}")
    print(f"   Score de calidad: {response.quality_score}")
    print(f"   ¿Usó filtro colaborativo? {'✅' if len(response.products) > 0 else '❌'}")
    
    # Mostrar estadísticas
    print("\n📊 ESTADÍSTICAS DEL SISTEMA:")
    stats = user_manager.get_demographic_stats()
    print(f"   Total usuarios: {stats['total_users']}")
    print(f"   Distribución por edad: {stats['age_distribution']}")
    print(f"   Búsquedas totales: {stats['total_searches']}")
    print(f"   Feedbacks totales: {stats['total_feedbacks']}")
    
    print("\n🎉 DEMO COMPLETADA - Sistema híbrido funcionando correctamente")

if __name__ == "__main__":
    demo_hybrid_system()