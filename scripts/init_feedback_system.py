#!/usr/bin/env python3
# scripts/init_feedback_system.py

import json
from pathlib import Path

def init_feedback_system():
    """Inicializa el sistema de feedback."""
    
    # 1. Crear estructura de directorios
    directories = [
        "data/processed/historial",
        "data/feedback",
        "data/models/rlhf_model"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Directorio creado/verificado: {dir_path}")
    
    # 2. Crear archivo de historial inicial si no existe
    historial_file = Path("data/processed/historial/conversation_init.json")
    if not historial_file.exists():
        initial_data = [
            {
                "timestamp": "2024-01-01T00:00:00",
                "session_id": "session_init",
                "user_id": "system",
                "query": "ejemplo de consulta",
                "response": "esta es una respuesta de ejemplo",
                "feedback": 5,
                "products_shown": ["example_product_1"],
                "source": "init",
                "processed": True
            }
        ]
        
        with open(historial_file, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)
        
        print("📄 Archivo de historial inicial creado")
    
    # 3. Verificar que FeedbackProcessor pueda cargar datos
    try:
        from src.core.rag.advanced.feedback_processor import FeedbackProcessor
        
        processor = FeedbackProcessor()
        print("✅ FeedbackProcessor inicializado correctamente")
        
        # Verificar logs
        failed_log = Path("data/feedback/failed_queries.log")
        success_log = Path("data/feedback/success_queries.log")
        
        if failed_log.exists():
            with open(failed_log, 'r', encoding='utf-8') as f:
                failed_count = sum(1 for _ in f)
            print(f"📉 Consultas fallidas en logs: {failed_count}")
        
        if success_log.exists():
            with open(success_log, 'r', encoding='utf-8') as f:
                success_count = sum(1 for _ in f)
            print(f"📈 Consultas exitosas en logs: {success_count}")
        
        processor.close()
        
    except Exception as e:
        print(f"⚠️  Error inicializando FeedbackProcessor: {e}")
    
    print("\n✅ Sistema de feedback inicializado")

if __name__ == "__main__":
    init_feedback_system()