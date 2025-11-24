#!/usr/bin/env python3
"""
Fuerza reentrenamiento modificando el timestamp
"""

import time
import pickle
from pathlib import Path

def force_immediate_retrain():
    """Fuerza reentrenamiento inmediato"""
    
    # Ruta donde se guarda el estado del agente
    state_file = Path("data/feedback/agent_state.pkl")
    
    # Crear estado forzando reentrenamiento
    agent_state = {
        'last_retrain_time': 0,  # Hace mucho tiempo
        'feedback_count': 10,    # Suficiente feedback
        'force_retrain': True    # Bandera forzada
    }
    
    # Guardar estado
    with open(state_file, 'wb') as f:
        pickle.dump(agent_state, f)
    
    print("✅ Estado modificado - próximo reentrenamiento forzado")
    print("🕒 Último reentrenamiento: Nunca (forzado)")
    print("📊 Feedback count: 10 (artificial)")

if __name__ == "__main__":
    force_immediate_retrain()