#!/usr/bin/env python3
"""
Monitorea el estado del reentrenamiento
"""

import time
from pathlib import Path
import json

def monitor_retraining_status():
    """Monitorea el estado del sistema de reentrenamiento"""
    
    print("🔍 MONITOREO DE REENTRENAMIENTO")
    print("=" * 50)
    
    # Contar feedback actual
    feedback_dir = Path("data/feedback")
    total_feedback = 0
    
    for jsonl_file in feedback_dir.glob("*.jsonl"):
        if jsonl_file.exists():
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f)
            print(f"📊 {jsonl_file.name}: {count} ejemplos")
            total_feedback += count
    
    print(f"🎯 TOTAL FEEDBACK: {total_feedback} ejemplos")
    
    # Verificar modelos RLHF
    rlhf_dir = Path("models/rl_models")
    if rlhf_dir.exists():
        model_files = list(rlhf_dir.glob("*"))
        print(f"🤖 MODELOS RLHF: {len(model_files)} archivos")
        for f in model_files:
            print(f"   📁 {f.name}")
    else:
        print("🤖 MODELOS RLHF: No existen (primer entrenamiento pendiente)")
    
    # Umbral necesario
    min_feedback = 5
    status = "✅ LISTO" if total_feedback >= min_feedback else "⏳ PENDIENTE"
    print(f"🎯 ESTADO REENTRENAMIENTO: {status}")
    print(f"   (Se necesitan {min_feedback} feedbacks, hay {total_feedback})")

if __name__ == "__main__":
    monitor_retraining_status()