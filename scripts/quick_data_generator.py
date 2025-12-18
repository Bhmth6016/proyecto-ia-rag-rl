#!/usr/bin/env python3
# scripts/quick_data_generator.py
"""
Generador rápido de datos de prueba.
"""

import json
import random
from pathlib import Path
from datetime import datetime, timedelta
import uuid

def generate_quick_data():
    """Genera datos de prueba rápidamente."""
    
    print("🚀 GENERANDO DATOS DE PRUEBA RÁPIDOS")
    print("="*60)
    
    # Crear directorios
    data_dir = Path("large_test_data")
    (data_dir / "processed/historial").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed/user_profiles").mkdir(parents=True, exist_ok=True)
    (data_dir / "feedback").mkdir(parents=True, exist_ok=True)
    (data_dir / "feedback/user_feedback").mkdir(parents=True, exist_ok=True)
    
    # Generar 30 usuarios
    users = []
    for i in range(30):
        user_id = f"test_user_{i+1:03d}"
        user = {
            "user_id": user_id,
            "name": f"Usuario {i+1}",
            "email": f"user{i+1}@test.com",
            "age": random.randint(18, 65),
            "country": random.choice(["Spain", "Mexico", "Argentina"]),
            "preferred_categories": random.sample(["Electronics", "Books", "Clothing"], k=2),
            "feedback_history": [],
            "created_at": datetime.now().isoformat()
        }
        users.append(user)
        
        # Guardar perfil individual
        user_file = data_dir / "processed/user_profiles" / f"{user_id}.json"
        with open(user_file, 'w', encoding='utf-8') as f:
            json.dump(user, f, indent=2)
    
    print(f"✅ {len(users)} usuarios generados")
    
    # Generar conversaciones
    all_conversations = []
    queries = [
        "smartphone económico", "laptop gaming", "libros de programación",
        "zapatillas deportivas", "cámara digital", "auriculares bluetooth",
        "consola videojuegos", "tablet android", "smartwatch", "impresora láser"
    ]
    
    success_log = data_dir / "feedback" / "success_queries.log"
    failed_log = data_dir / "feedback" / "failed_queries.log"
    
    success_entries = []
    failed_entries = []
    
    for user in users:
        for _ in range(15):  # 15 conversaciones por usuario
            # Crear conversación
            conv = {
                "timestamp": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
                "session_id": f"session_{random.randint(1000, 9999)}",
                "user_id": user["user_id"],
                "query": random.choice(queries),
                "response": f"Respuesta para {user['user_id']} sobre productos relevantes.",
                "feedback": random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.15, 0.2, 0.25, 0.3])[0],
                "products_shown": [f"prod_{random.randint(100, 999)}" for _ in range(3)],
                "selected_product": f"prod_{random.randint(100, 999)}",
                "source": "quick_generator",
                "processed": False
            }
            
            all_conversations.append(conv)
            
            # Agregar a feedback_history del usuario
            user["feedback_history"].append({
                "timestamp": conv["timestamp"],
                "query": conv["query"],
                "response": conv["response"],
                "rating": conv["feedback"],
                "products_shown": conv["products_shown"],
                "selected_product": conv["selected_product"]
            })
            
            # Agregar a logs
            log_entry = {
                "timestamp": conv["timestamp"],
                "user_id": user["user_id"],
                "query": conv["query"],
                "response": conv["response"],
                "feedback": conv["feedback"],
                "source": "quick_generator"
            }
            
            if conv["feedback"] >= 4:
                success_entries.append(log_entry)
            else:
                failed_entries.append(log_entry)
    
    print(f"✅ {len(all_conversations)} conversaciones generadas")
    
    # Guardar historial por fecha
    from collections import defaultdict
    conversations_by_date = defaultdict(list)
    
    for conv in all_conversations:
        try:
            date = datetime.fromisoformat(conv["timestamp"]).date().isoformat()
            conversations_by_date[date].append(conv)
        except:
            date = datetime.now().date().isoformat()
            conversations_by_date[date].append(conv)
    
    for date_str, convs in conversations_by_date.items():
        hist_file = data_dir / "processed/historial" / f"conversation_{date_str}.json"
        with open(hist_file, 'w', encoding='utf-8') as f:
            json.dump(convs, f, indent=2)
    
    print(f"✅ {len(conversations_by_date)} archivos de historial creados")
    
    # Guardar logs
    with open(success_log, 'w', encoding='utf-8') as f:
        for entry in success_entries:
            f.write(json.dumps(entry) + "\n")
    
    with open(failed_log, 'w', encoding='utf-8') as f:
        for entry in failed_entries:
            f.write(json.dumps(entry) + "\n")
    
    print(f"✅ Logs de feedback: {len(success_entries)} exitosos, {len(failed_entries)} fallidos")
    
    # Guardar feedback por usuario
    for user in users:
        if user["feedback_history"]:
            fb_file = data_dir / "feedback/user_feedback" / f"feedback_{user['user_id']}.json"
            with open(fb_file, 'w', encoding='utf-8') as f:
                json.dump(user["feedback_history"], f, indent=2)
    
    print(f"✅ Archivos de feedback por usuario creados")
    
    # Estadísticas
    total_feedbacks = len(all_conversations)
    positive = sum(1 for c in all_conversations if c["feedback"] >= 4)
    negative = sum(1 for c in all_conversations if c["feedback"] < 4)
    
    print("\n📊 ESTADÍSTICAS FINALES:")
    print(f"   • Usuarios: {len(users)}")
    print(f"   • Conversaciones totales: {total_feedbacks}")
    print(f"   • Feedback positivo (>=4): {positive} ({positive/total_feedbacks*100:.1f}%)")
    print(f"   • Feedback negativo (<4): {negative} ({negative/total_feedbacks*100:.1f}%)")
    print(f"   • Directorio: {data_dir}")
    print("\n🎯 ¡Datos listos para usar!")

if __name__ == "__main__":
    generate_quick_data()