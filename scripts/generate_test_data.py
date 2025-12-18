#!/usr/bin/env python3
"""
Verifica los datos generados por el test data generator.
"""

import json
from pathlib import Path
from datetime import datetime
import sys

def check_generated_data():
    """Verifica la estructura y contenido de los datos generados."""
    
    print("🔍 VERIFICANDO DATOS GENERADOS")
    print("="*60)
    
    base_dirs = ["data", "test_data"]  # Directorios a verificar
    
    for base_dir in base_dirs:
        data_path = Path(base_dir)
        if not data_path.exists():
            print(f"\n📁 Directorio {base_dir} no existe.")
            continue
        
        print(f"\n📁 DIRECTORIO: {base_dir}")
        print("-"*40)
        
        # 1. Verificar estructura básica
        required_dirs = [
            "processed/historial",
            "processed/user_profiles", 
            "feedback",
            "feedback/user_feedback"
        ]
        
        for req_dir in required_dirs:
            dir_path = data_path / req_dir
            if dir_path.exists():
                file_count = len(list(dir_path.glob("*")))
                print(f"✅ {req_dir}: {file_count} archivos")
            else:
                print(f"❌ {req_dir}: NO EXISTE")
        
        # 2. Contar usuarios
        user_dir = data_path / "processed" / "user_profiles"
        if user_dir.exists():
            user_files = list(user_dir.glob("*.json"))
            print(f"\n👤 USUARIOS: {len(user_files)} perfiles")
            
            # Analizar feedbacks por usuario
            total_feedbacks = 0
            for user_file in user_files[:3]:  # Mostrar primeros 3
                try:
                    with open(user_file, 'r', encoding='utf-8') as f:
                        user = json.load(f)
                        fb_count = len(user.get('feedback_history', []))
                        total_feedbacks += fb_count
                        
                        if fb_count > 0:
                            ratings = [fb.get('rating', 0) for fb in user['feedback_history']]
                            avg_rating = sum(ratings) / len(ratings)
                            print(f"   📄 {user_file.stem}: {fb_count} feedbacks (avg: {avg_rating:.1f}/5)")
                except:
                    pass
        
        # 3. Verificar historial
        historial_dir = data_path / "processed" / "historial"
        if historial_dir.exists():
            historial_files = list(historial_dir.glob("conversation_*.json"))
            print(f"\n📚 HISTORIAL: {len(historial_files)} archivos")
            
            total_conversations = 0
            for hist_file in historial_files[:2]:  # Mostrar primeros 2
                try:
                    with open(hist_file, 'r', encoding='utf-8') as f:
                        convs = json.load(f)
                        total_conversations += len(convs)
                        
                        # Calcular distribución de feedback
                        if convs:
                            ratings = [c.get('feedback', 0) for c in convs]
                            positive = sum(1 for r in ratings if r >= 4)
                            negative = sum(1 for r in ratings if r < 4 and r > 0)
                            neutral = sum(1 for r in ratings if r == 0)
                            
                            print(f"   📄 {hist_file.name}: {len(convs)} convs | "
                                  f"✅{positive} ❌{negative} ➖{neutral}")
                except:
                    pass
        
        # 4. Verificar logs de feedback
        feedback_dir = data_path / "feedback"
        if feedback_dir.exists():
            print(f"\n📝 LOGS DE FEEDBACK:")
            
            # Success log
            success_log = feedback_dir / "success_queries.log"
            if success_log.exists():
                with open(success_log, 'r', encoding='utf-8') as f:
                    success_lines = sum(1 for _ in f)
                print(f"   ✅ success_queries.log: {success_lines} líneas")
            
            # Failed log
            failed_log = feedback_dir / "failed_queries.log"
            if failed_log.exists():
                with open(failed_log, 'r', encoding='utf-8') as f:
                    failed_lines = sum(1 for _ in f)
                print(f"   ❌ failed_queries.log: {failed_lines} líneas")
            
            # Archivos individuales
            user_feedback_dir = feedback_dir / "user_feedback"
            if user_feedback_dir.exists():
                fb_files = list(user_feedback_dir.glob("*.json"))
                print(f"   📁 user_feedback/: {len(fb_files)} archivos")
    
    print("\n" + "="*60)
    print("✅ Verificación completada")

if __name__ == "__main__":
    check_generated_data()