#!/usr/bin/env python3
"""
Script de verificación final del sistema completo
"""

import logging
from pathlib import Path
from src.core.rag.advanced.WorkingRAGAgent import WorkingAdvancedRAGAgent
from src.core.data.user_manager import UserManager

def verify_system_completeness():
    """Verifica que todos los componentes del sistema estén funcionando"""
    
    print("🔍 VERIFICACIÓN FINAL DEL SISTEMA HÍBRIDO RAG + RL")
    print("=" * 60)
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: Componentes del sistema
    print("\n1. VERIFICANDO COMPONENTES DEL SISTEMA...")
    try:
        rag_agent = WorkingAdvancedRAGAgent()
        print("   ✅ WorkingAdvancedRAGAgent - CARGADO")
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ WorkingAdvancedRAGAgent - ERROR: {e}")
    total_checks += 1
    
    try:
        user_manager = UserManager()
        print("   ✅ UserManager - CARGADO") 
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ UserManager - ERROR: {e}")
    total_checks += 1
    
    # Check 2: Archivos de datos
    print("\n2. VERIFICANDO ARCHIVOS DE DATOS...")
    required_dirs = [
        "data/feedback",
        "data/users", 
        "data/processed/historial",
        "data/processed/chroma_db"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"   ✅ {dir_path} - EXISTE")
            checks_passed += 1
        else:
            print(f"   ⚠️  {dir_path} - NO EXISTE")
        total_checks += 1
    
    # Check 3: Configuración híbrida
    print("\n3. VERIFICANDO CONFIGURACIÓN HÍBRIDA...")
    try:
        weights = rag_agent.hybrid_weights
        if weights['collaborative'] == 0.6 and weights['rag'] == 0.4:
            print("   ✅ Pesos híbridos configurados correctamente (0.6/0.4)")
            checks_passed += 1
        else:
            print(f"   ⚠️  Pesos híbridos incorrectos: {weights}")
        total_checks += 1
        
        if rag_agent.min_similarity_threshold == 0.6:
            print("   ✅ Umbral de similitud configurado correctamente (0.6)")
            checks_passed += 1
        else:
            print(f"   ⚠️  Umbral de similitud incorrecto: {rag_agent.min_similarity_threshold}")
        total_checks += 1
    
    except Exception as e:
        print(f"   ❌ Error verificando configuración: {e}")
    
    # Check 4: RLHF activo
    print("\n4. VERIFICANDO SISTEMA RLHF...")
    try:
        if rag_agent.min_feedback_for_retrain == 5:
            print("   ✅ RLHF configurado con umbral bajo (5 feedbacks)")
            checks_passed += 1
        else:
            print(f"   ⚠️  Umbral RLHF incorrecto: {rag_agent.min_feedback_for_retrain}")
        total_checks += 1
        
        if rag_agent.retrain_interval == 3600:
            print("   ✅ Intervalo de reentrenamiento configurado (1 hora)")
            checks_passed += 1
        else:
            print(f"   ⚠️  Intervalo RLHF incorrecto: {rag_agent.retrain_interval}")
        total_checks += 1
    
    except Exception as e:
        print(f"   ❌ Error verificando RLHF: {e}")
    
    # Resumen final
    print("\n" + "=" * 60)
    print(f"📊 RESUMEN DE VERIFICACIÓN:")
    print(f"   Checks pasados: {checks_passed}/{total_checks}")
    
    success_rate = (checks_passed / total_checks) * 100
    if success_rate >= 90:
        print(f"   🎉 ESTADO: EXCELENTE ({success_rate:.1f}%)")
        print("   El sistema híbrido está completamente operativo")
    elif success_rate >= 70:
        print(f"   ✅ ESTADO: BUENO ({success_rate:.1f}%)") 
        print("   El sistema funciona con algunas advertencias menores")
    else:
        print(f"   ⚠️  ESTADO: REQUIERE ATENCIÓN ({success_rate:.1f}%)")
        print("   Revisar los componentes marcados con error")
    
    print(f"\n🎯 SISTEMA HÍBRIDO RAG + RL - VERIFICACIÓN COMPLETADA")

if __name__ == "__main__":
    verify_system_completeness()