# pipeline_reviews_to_rlhf.py
#!/usr/bin/env python3
"""
Pipeline COMPLETO - Reviews → RLHF (SIMPLIFICADO)
=================================================

USO SIMPLE:
-----------
python run_complete_pipeline.py

Ejecuta automáticamente:
1. Genera pares RLHF de TODAS las categorías
2. Integra pares con el sistema
3. Muestra siguiente paso

Ventajas:
- Sin parámetros complicados
- Auto-detección de categorías
- Validación automática
- Reportes claros
"""

import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Formato simple
)
logger = logging.getLogger(__name__)


def print_banner(title: str):
    """Imprime un banner bonito"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def check_requirements():
    """Verifica que existan los datos necesarios"""
    print_banner("VERIFICANDO REQUISITOS")
    
    # Verificar directorios
    required_dirs = [
        Path("data/raw"),
        Path("data/reviews")
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not dir_path.exists():
            missing.append(str(dir_path))
            logger.error(f"❌ Faltante: {dir_path}")
        else:
            # Contar archivos
            files = list(dir_path.glob("*.jsonl"))
            logger.info(f"✓ {dir_path}: {len(files)} archivos")
    
    if missing:
        logger.error("\n❌ Directorios faltantes!")
        logger.error("   Asegúrate de tener:")
        logger.error("   • data/raw/meta_*.jsonl")
        logger.error("   • data/reviews/*.jsonl")
        return False
    
    return True


def step_1_generate_pairs():
    """Paso 1: Genera pares RLHF de todas las categorías"""
    print_banner("PASO 1: GENERAR PARES RLHF")
    
    logger.info("📊 Procesando TODAS las categorías automáticamente...")
    logger.info("   (Esto puede tomar 5-10 minutos)\n")
    
    try:
        # Importar y ejecutar
        from generate_rlhf_pairs_from_reviews import RLHFPairGenerator
        
        generator = RLHFPairGenerator(
            data_dir=Path("data"),
            output_dir=Path("data/rlhf_pairs"),
            min_reviews=5,
            pairs_per_query=3
        )
        
        # Procesar TODAS las categorías
        generator.run_all_categories(
            limit_products=10000,   # Todos disponibles
            limit_reviews=100000    # Primeras 100K reviews
        )
        
        logger.info("\n✅ Paso 1 completado exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Error en Paso 1: {e}")
        import traceback
        traceback.print_exc()
        return False


def step_2_integrate_pairs():
    """Paso 2: Integra pares con el sistema"""
    print_banner("PASO 2: INTEGRAR CON SISTEMA")
    
    logger.info("🔄 Integrando pares de todas las categorías...\n")
    
    try:
        from integrate_rlhf_pairs import RLHFPairsIntegrator
        
        integrator = RLHFPairsIntegrator(
            pairs_dir=Path("data/rlhf_pairs"),
            output_file=Path("data/interactions/rlhf_interactions_from_reviews.jsonl"),
            ground_truth_file=Path("data/interactions/ground_truth_from_reviews.json")
        )
        
        success = integrator.run()
        
        if success:
            logger.info("\n✅ Paso 2 completado exitosamente")
            return True
        else:
            logger.error("\n❌ Paso 2 falló")
            return False
        
    except Exception as e:
        logger.error(f"\n❌ Error en Paso 2: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_next_steps():
    """Muestra los próximos pasos"""
    print_banner("🎯 PRÓXIMOS PASOS")
    
    print("✅ Datos RLHF generados exitosamente!")
    print("\nAhora puedes ejecutar el experimento:")
    print("\n  python main.py experimento\n")
    
    print("Esto evaluará 4 métodos:")
    print("  1. Baseline (FAISS)")
    print("  2. NER-Enhanced")
    print("  3. RLHF (entrenado con reviews)")
    print("  4. Full Hybrid")
    
    print("\n" + "="*70)
    print("\n💡 TIPS:")
    print("   • El experimento puede tomar 10-20 minutos")
    print("   • Los resultados se guardarán en results/")
    print("   • Busca mejoras >15% en MRR para paper IEEE")
    print("\n" + "="*70)


def main():
    """Función principal"""
    print_banner("🚀 PIPELINE COMPLETO: Reviews → RLHF")
    
    print("Este script ejecutará automáticamente:")
    print("  1. Generación de pares RLHF (todas las categorías)")
    print("  2. Integración con el sistema")
    print("\n¿Continuar? (s/n): ", end='')
    
    response = input().strip().lower()
    if response != 's':
        print("\n❌ Cancelado")
        return 1
    
    # Paso 0: Verificar requisitos
    if not check_requirements():
        logger.error("\n❌ Requisitos no cumplidos")
        return 1
    
    # Paso 1: Generar pares
    if not step_1_generate_pairs():
        logger.error("\n❌ Pipeline interrumpido en Paso 1")
        return 1
    
    # Paso 2: Integrar pares
    if not step_2_integrate_pairs():
        logger.error("\n❌ Pipeline interrumpido en Paso 2")
        return 1
    
    # Mostrar próximos pasos
    show_next_steps()
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrumpido por usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)