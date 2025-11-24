#!/usr/bin/env python3
"""
Debug y forzado de reentrenamiento RLHF
"""

import sys
from pathlib import Path
import json

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).parent))

from src.core.rag.advanced.WorkingRAGAgent import WorkingAdvancedRAGAgent

def debug_and_force_retrain():
    """Debug del sistema y forzado de reentrenamiento"""
    
    print("🔧 DEBUG Y REENTRENAMIENTO FORZADO")
    print("=" * 50)
    
    # 1. Verificar feedback disponible
    feedback_dir = Path("data/feedback")
    success_log = feedback_dir / "success_queries.log"
    failed_log = feedback_dir / "failed_queries.log"
    
    print("📊 VERIFICANDO ARCHIVOS DE FEEDBACK:")
    print(f"   success_queries.log: {success_log.exists()}")
    print(f"   failed_queries.log: {failed_log.exists()}")
    
    if success_log.exists():
        with open(success_log, 'r', encoding='utf-8') as f:
            success_count = sum(1 for _ in f)
        print(f"   ✅ Ejemplos positivos: {success_count}")
    
    if failed_log.exists():
        with open(failed_log, 'r', encoding='utf-8') as f:
            failed_count = sum(1 for _ in f)
        print(f"   ❌ Ejemplos negativos: {failed_count}")
    
    # 2. Crear instancia del agente
    print("\n🤖 INICIALIZANDO AGENTE...")
    agent = WorkingAdvancedRAGAgent()
    
    # 3. Forzar verificación de reentrenamiento
    print("\n🔄 FORZANDO VERIFICACIÓN DE REENTRENAMIENTO...")
    try:
        # Llamar directamente al método de verificación
        agent._check_and_retrain()
        print("✅ Verificación de reentrenamiento ejecutada")
    except Exception as e:
        print(f"❌ Error en verificación: {e}")
    
    # 4. Intentar reentrenamiento manual
    print("\n🏋️ EJECUTANDO REENTRENAMIENTO MANUAL...")
    try:
        success = agent._retrain_with_feedback()
        if success:
            print("🎉 ¡REENTRENAMIENTO EXITOSO!")
        else:
            print("⚠️ Reentrenamiento falló o no hay datos suficientes")
    except Exception as e:
        print(f"❌ Error en reentrenamiento: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. Verificar resultado
    rlhf_dir = Path("models/rl_models")
    print(f"\n📁 VERIFICANDO MODELOS EN: {rlhf_dir}")
    if rlhf_dir.exists():
        model_files = list(rlhf_dir.glob("*"))
        print(f"   Archivos encontrados: {len(model_files)}")
        for f in model_files:
            print(f"   📄 {f.name}")
    else:
        print("   ❌ Directorio no existe")

if __name__ == "__main__":
    debug_and_force_retrain()