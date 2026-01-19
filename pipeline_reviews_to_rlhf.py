# pipeline_reviews_to_rlhf.py
#!/usr/bin/env python3
"""
Pipeline Completo: Reviews → RLHF Training
==========================================

Orquesta el proceso completo de convertir reviews en datos RLHF.

Pasos:
1. Generar pares (chosen, rejected) desde reviews
2. Integrar pares con sistema existente
3. Entrenar modelo RLHF
4. Evaluar con experimento de 4 métodos

Uso:
    python pipeline_reviews_to_rlhf.py --category All_Beauty --limit 10000
"""

import argparse
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_files_exist(data_dir: Path, category: str) -> bool:
    """Verifica que existan los archivos necesarios"""
    # PRODUCTOS están en data/raw/
    products_file = data_dir / "raw" / f"meta_{category}.jsonl"
    
    # REVIEWS están en data/reviews/
    reviews_file = data_dir / "reviews" / f"{category}.jsonl"
    
    missing = []
    
    if not products_file.exists():
        missing.append(str(products_file))
    
    if not reviews_file.exists():
        missing.append(str(reviews_file))
    
    if missing:
        logger.error("❌ Archivos faltantes:")
        for file in missing:
            logger.error(f"   • {file}")
        return False
    
    logger.info("✅ Archivos encontrados:")
    logger.info(f"   • {products_file.name}")
    logger.info(f"   • {reviews_file.name}")
    
    return True


def step_1_generate_pairs(
    data_dir: Path,
    category: str,
    limit_products: int,
    limit_reviews: int
):
    """Paso 1: Generar pares RLHF desde reviews"""
    logger.info("\n" + "="*60)
    logger.info("PASO 1: Generar Pares RLHF desde Reviews")
    logger.info("="*60)
    
    try:
        from generate_rlhf_pairs_from_reviews import RLHFPairGenerator
        
        # Rutas CORRECTAS:
        products_file = data_dir / "raw" / f"meta_{category}.jsonl"
        reviews_file = data_dir / "reviews" / f"{category}.jsonl"
        output_file = Path("data/rlhf_pairs") / f"rlhf_pairs_{category.lower()}.jsonl"
        
        generator = RLHFPairGenerator(
            products_file=products_file,
            reviews_file=reviews_file,
            output_file=output_file,
            min_reviews=5,
            pairs_per_query=3
        )
        
        generator.run(
            limit_products=limit_products,
            limit_reviews=limit_reviews
        )
        
        logger.info(f"✅ Paso 1 completado: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"❌ Error en Paso 1: {e}")
        import traceback
        traceback.print_exc()
        return None


def step_2_integrate_pairs(pairs_file: Path):
    """Paso 2: Integrar pares con sistema"""
    logger.info("\n" + "="*60)
    logger.info("PASO 2: Integrar Pares con Sistema")
    logger.info("="*60)
    
    try:
        from integrate_rlhf_pairs import RLHFPairsIntegrator
        
        output_file = Path("data/interactions/rlhf_interactions_from_reviews.jsonl")
        ground_truth_file = Path("data/interactions/ground_truth_from_reviews.json")
        
        integrator = RLHFPairsIntegrator(
            pairs_file=pairs_file,
            output_file=output_file,
            ground_truth_file=ground_truth_file
        )
        
        integrator.run()
        
        logger.info(f"✅ Paso 2 completado")
        logger.info(f"   • Interacciones: {output_file}")
        logger.info(f"   • Ground truth: {ground_truth_file}")
        
        return output_file, ground_truth_file
        
    except Exception as e:
        logger.error(f"❌ Error en Paso 2: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def step_3_train_system(interactions_file: Path):
    """Paso 3: Entrenar sistema con RLHF"""
    logger.info("\n" + "="*60)
    logger.info("PASO 3: Entrenar Sistema RLHF")
    logger.info("="*60)
    
    try:
        # Verificar que el sistema esté inicializado
        system_cache = Path("data/cache/unified_system_v2.pkl")
        
        if not system_cache.exists():
            logger.warning("⚠️ Sistema no inicializado")
            logger.info("   Ejecutando: python main.py init")
            
            import subprocess
            result = subprocess.run(
                [sys.executable, "main.py", "init"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error("❌ Error inicializando sistema")
                logger.error(result.stderr)
                return False
            
            logger.info("✅ Sistema inicializado")
        
        # Cargar sistema
        from src.unified_system_v2 import UnifiedSystemV2
        
        logger.info("📂 Cargando sistema...")
        system = UnifiedSystemV2.load_from_cache()
        
        if not system:
            logger.error("❌ No se pudo cargar sistema")
            return False
        
        logger.info(f"✅ Sistema cargado: {len(system.canonical_products):,} productos")
        
        # Cargar interacciones para extraer queries de entrenamiento
        import json
        
        train_queries = set()
        
        with open(interactions_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    interaction = json.loads(line)
                    if interaction.get('interaction_type') == 'click':
                        query = interaction.get('context', {}).get('query')
                        if query:
                            train_queries.add(query)
                except json.JSONDecodeError:
                    continue
        
        train_queries = list(train_queries)
        logger.info(f"📝 Queries de entrenamiento: {len(train_queries)}")
        
        if not train_queries:
            logger.error("❌ No hay queries para entrenar")
            return False
        
        # Entrenar RLHF
        logger.info("🎓 Entrenando RLHF...")
        
        success = system.train_rlhf_with_queries(
            train_queries=train_queries,
            interactions_file=interactions_file
        )
        
        if success:
            logger.info("✅ RLHF entrenado exitosamente")
            
            stats = system.rl_ranker.get_stats()
            logger.info(f"   • Feedback procesado: {stats.get('feedback_count', 0)}")
            logger.info(f"   • Features aprendidas: {stats.get('weights_count', 0)}")
            
            return True
        else:
            logger.warning("⚠️ RLHF no pudo ser entrenado (datos insuficientes?)")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error en Paso 3: {e}")
        import traceback
        traceback.print_exc()
        return False


def step_4_run_experiment():
    """Paso 4: Ejecutar experimento de evaluación"""
    logger.info("\n" + "="*60)
    logger.info("PASO 4: Ejecutar Experimento de Evaluación")
    logger.info("="*60)
    
    try:
        logger.info("🧪 Ejecutando experimento de 4 métodos...")
        
        import subprocess
        result = subprocess.run(
            [sys.executable, "main.py", "experimento"],
            capture_output=False,  # Mostrar output en tiempo real
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ Experimento completado")
            return True
        else:
            logger.error("❌ Error en experimento")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error en Paso 4: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline completo: Reviews → RLHF Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Pipeline completo con 10K productos
  python pipeline_reviews_to_rlhf.py --category All_Beauty --limit 10000
  
  # Solo generar pares (sin entrenar)
  python pipeline_reviews_to_rlhf.py --category All_Beauty --limit 5000 --pairs-only
  
  # Pipeline completo desde el paso 2 (si ya tienes pares)
  python pipeline_reviews_to_rlhf.py --category All_Beauty --from-step 2
        """
    )
    
    parser.add_argument(
        '--category',
        type=str,
        required=True,
        help='Categoría de Amazon (ej: All_Beauty, Toys_and_Games)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=10000,
        help='Límite de productos a procesar (default: 10000)'
    )
    
    parser.add_argument(
        '--limit-reviews',
        type=int,
        default=None,
        help='Límite de reviews (default: None = todas)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),  # ✅ Solo hasta data/
        help='Directorio base de datos (default: data)'
    )
    
    parser.add_argument(
        '--pairs-only',
        action='store_true',
        help='Solo generar pares (no entrenar)'
    )
    
    parser.add_argument(
        '--from-step',
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help='Comenzar desde paso específico'
    )
    
    args = parser.parse_args()
    
    # Banner
    print("\n" + "="*60)
    print(" PIPELINE: REVIEWS → RLHF TRAINING")
    print("="*60)
    print(f"Categoría:        {args.category}")
    print(f"Límite productos: {args.limit:,}")
    print(f"Límite reviews:   {args.limit_reviews or 'Todas'}")
    print(f"Comenzar desde:   Paso {args.from_step}")
    print("="*60)
    
    # Verificar archivos
    if args.from_step == 1:
        if not check_files_exist(args.data_dir, args.category):
            logger.error("❌ Archivos necesarios no encontrados")
            return 1
    
    # Pipeline
    pairs_file = None
    interactions_file = None
    ground_truth_file = None
    
    # Paso 1: Generar pares
    if args.from_step <= 1:
        pairs_file = step_1_generate_pairs(
            data_dir=args.data_dir,
            category=args.category,
            limit_products=args.limit,
            limit_reviews=args.limit_reviews or args.limit * 10
        )
        
        if not pairs_file or not pairs_file.exists():
            logger.error("❌ Fallo en Paso 1")
            return 1
    else:
        # Buscar archivo existente
        pairs_file = Path("data/rlhf_pairs") / f"rlhf_pairs_{args.category.lower()}.jsonl"
        if not pairs_file.exists():
            logger.error(f"❌ Archivo de pares no existe: {pairs_file}")
            return 1
    
    if args.pairs_only:
        logger.info("\n✅ Pipeline completado (--pairs-only)")
        logger.info(f"   Pares generados: {pairs_file}")
        return 0
    
    # Paso 2: Integrar pares
    if args.from_step <= 2:
        interactions_file, ground_truth_file = step_2_integrate_pairs(pairs_file)
        
        if not interactions_file or not interactions_file.exists():
            logger.error("❌ Fallo en Paso 2")
            return 1
    else:
        interactions_file = Path("data/interactions/rlhf_interactions_from_reviews.jsonl")
        ground_truth_file = Path("data/interactions/ground_truth_from_reviews.json")
        
        if not interactions_file.exists() or not ground_truth_file.exists():
            logger.error("❌ Archivos de interacciones no existen")
            return 1
    
    # Paso 3: Entrenar RLHF
    if args.from_step <= 3:
        success = step_3_train_system(interactions_file)
        
        if not success:
            logger.error("❌ Fallo en Paso 3")
            return 1
    
    # Paso 4: Ejecutar experimento
    if args.from_step <= 4:
        success = step_4_run_experiment()
        
        if not success:
            logger.error("❌ Fallo en Paso 4")
            return 1
    
    # Resumen final
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*60)
    print("\n📊 Resultados:")
    print(f"   • Pares RLHF:     {pairs_file}")
    print(f"   • Interacciones:  {interactions_file}")
    print(f"   • Ground truth:   {ground_truth_file}")
    print(f"   • Experimento:    results/experimento_4_metodos_*.json")
    print("\n🎯 Próximos pasos:")
    print("   1. Revisar resultados en results/")
    print("   2. Ajustar hiperparámetros si es necesario")
    print("   3. Escalar a más categorías")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())