# test.py
#!/usr/bin/env python3
"""
Test Rápido del Pipeline
========================

Prueba el pipeline con un subset pequeño de datos.
Útil para verificar que todo funciona antes de procesar millones de reviews.

Uso:
    python test_pipeline_quick.py
"""

import json
from pathlib import Path
from typing import Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_review_file(reviews_dir: Path, category: str) -> Optional[Path]:
    """Encuentra el archivo de reviews, manejando nombres con _10000"""
    # Intentar diferentes patrones
    patterns = [
        f"{category}.jsonl",
        f"{category}_*.jsonl",
        f"*{category}*.jsonl"
    ]
    
    for pattern in patterns:
        matches = list(reviews_dir.glob(pattern))
        if matches:
            return matches[0]
    
    return None


def quick_test():
    """Test rápido del pipeline completo"""
    
    logger.info("="*60)
    logger.info("TEST RÁPIDO DEL PIPELINE")
    logger.info("="*60)
    
    # 1. Verificar archivos
    logger.info("\n1️⃣ Verificando archivos...")
    
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    reviews_dir = data_dir / "reviews"
    
    # Buscar cualquier categoría disponible
    meta_files = list(raw_dir.glob("meta_*.jsonl"))
    review_files = list(reviews_dir.glob("*.jsonl"))
    review_files = [f for f in review_files if not f.name.startswith("meta_")]
    
    if not meta_files:
        logger.error("❌ No se encontraron archivos meta_*.jsonl")
        logger.info("   Coloca tus archivos en data/raw/")
        return False
    
    if not review_files:
        logger.error("❌ No se encontraron archivos de reviews")
        return False
    
    logger.info(f"✅ Archivos meta encontrados: {len(meta_files)}")
    logger.info(f"✅ Archivos reviews encontrados: {len(review_files)}")
    
    # Usar primer archivo como ejemplo
    meta_file = meta_files[0]
    
    # Encontrar reviews correspondientes
    category = meta_file.name.replace("meta_", "").replace(".jsonl", "")
    # Remover _10000 si existe
    if "_10000" in category:
        category = category.replace("_10000", "")
    
    # Buscar archivo de reviews
    review_file = find_review_file(reviews_dir, category)
    
    if not review_file:
        logger.warning(f"⚠️ No se encontró archivo de reviews para {category}")
        # Buscar cualquier archivo de reviews
        if review_files:
            review_file = review_files[0]
            # Extraer categoría del nombre del archivo
            category = review_file.stem
            if "_10000" in category:
                category = category.replace("_10000", "")
        else:
            logger.error("❌ No hay archivos de reviews disponibles")
            return False
    
    logger.info(f"\n📂 Usando categoría: {category}")
    logger.info(f"   Meta:    {meta_file.name}")
    logger.info(f"   Reviews: {review_file.name}")
    
    # 2. Cargar muestra pequeña
    logger.info("\n2️⃣ Cargando muestra (100 productos, 1000 reviews)...")
    
    products = {}
    count = 0
    
    with open(meta_file, 'rt', encoding='utf-8') as f:  # ✅ CORRECTO: open, no json.open
        for line in f:
            if count >= 100:
                break
            try:
                product = json.loads(line)
                parent_asin = product.get('parent_asin')
                if parent_asin:
                    products[parent_asin] = product
                    count += 1
            except json.JSONDecodeError:
                continue
    
    logger.info(f"✅ Productos cargados: {len(products)}")
    
    # Mostrar ejemplo
    if products:
        sample = list(products.values())[0]
        logger.info(f"\nEjemplo de producto:")
        logger.info(f"   ASIN:  {sample.get('parent_asin', 'N/A')}")
        logger.info(f"   Título: {sample.get('title', 'N/A')[:50]}...")
        logger.info(f"   Categoría: {sample.get('main_category', 'N/A')}")
    
    # Cargar reviews
    reviews_by_product = defaultdict(list)
    count = 0
    
    with open(review_file, 'rt', encoding='utf-8') as f:  # ✅ CORRECTO: open, no json.open
        for line in f:
            if count >= 1000:
                break
            try:
                review = json.loads(line)
                parent_asin = review.get('parent_asin')
                if parent_asin:
                    reviews_by_product[parent_asin].append(review)
                    count += 1
            except json.JSONDecodeError:
                continue
    
    logger.info(f"✅ Reviews cargadas: {count}")
    logger.info(f"✅ Productos con reviews: {len(reviews_by_product)}")
    
    # Mostrar ejemplo de review
    if reviews_by_product:
        sample_reviews = list(reviews_by_product.values())[0]
        if sample_reviews:
            sample_review = sample_reviews[0]
            logger.info(f"\nEjemplo de review:")
            logger.info(f"   Rating: {sample_review.get('rating', 'N/A')}")
            logger.info(f"   Title: {sample_review.get('title', 'N/A')[:50]}...")
            logger.info(f"   Helpful: {sample_review.get('helpful_vote', 0)}")
            logger.info(f"   Verified: {sample_review.get('verified_purchase', False)}")
    
    # 3. Calcular rewards de ejemplo
    logger.info("\n3️⃣ Calculando rewards de ejemplo...")
    
    from generate_rlhf_pairs_from_reviews import ReviewRewardCalculator
    
    calculator = ReviewRewardCalculator()
    
    product_rewards = []
    for parent_asin, reviews in list(reviews_by_product.items())[:10]:
        if len(reviews) < 3:
            continue
        
        reward = calculator.calculate_product_reward(reviews)
        
        product_rewards.append({
            'parent_asin': parent_asin,
            'num_reviews': len(reviews),
            'reward': reward
        })
    
    # Ordenar por reward
    product_rewards.sort(key=lambda x: x['reward'], reverse=True)
    
    logger.info(f"✅ Rewards calculados para {len(product_rewards)} productos")
    
    if product_rewards:
        logger.info("\nTop 3 productos por reward:")
        for i, p in enumerate(product_rewards[:3], 1):
            logger.info(f"   {i}. ASIN {p['parent_asin'][:10]}... "
                       f"reward={p['reward']:.3f} ({p['num_reviews']} reviews)")
        
        logger.info("\nBottom 3 productos por reward:")
        for i, p in enumerate(product_rewards[-3:], 1):
            logger.info(f"   {i}. ASIN {p['parent_asin'][:10]}... "
                       f"reward={p['reward']:.3f} ({p['num_reviews']} reviews)")
    
    # 4. Generar par de ejemplo
    logger.info("\n4️⃣ Generando par de ejemplo...")
    
    if len(product_rewards) >= 2:
        chosen = product_rewards[0]
        rejected = product_rewards[-1]
        
        pair = {
            'query': f"{category.replace('_', ' ')} products",
            'chosen': chosen,
            'rejected': rejected,
            'margin': chosen['reward'] - rejected['reward']
        }
        
        logger.info("✅ Par de ejemplo generado:")
        logger.info(f"\n   Query: '{pair['query']}'")
        logger.info(f"\n   CHOSEN:")
        logger.info(f"      ASIN:   {pair['chosen']['parent_asin']}")
        logger.info(f"      Reward: {pair['chosen']['reward']:.3f}")
        logger.info(f"\n   REJECTED:")
        logger.info(f"      ASIN:   {pair['rejected']['parent_asin']}")
        logger.info(f"      Reward: {pair['rejected']['reward']:.3f}")
        logger.info(f"\n   MARGIN: {pair['margin']:.3f}")
        
        if pair['margin'] > 0.3:
            logger.info("\n   ✅ Margin suficiente (>0.3) - Buen par para RLHF")
        else:
            logger.info("\n   ⚠️ Margin bajo (<0.3) - Considerar filtrar")
    
    # 5. Verificar compatibilidad con sistema
    logger.info("\n5️⃣ Verificando compatibilidad con sistema...")
    
    try:
        from src.unified_system_v2 import UnifiedSystemV2
        logger.info("✅ unified_system_v2 importado correctamente")
        
        system_cache = Path("data/cache/unified_system_v2.pkl")
        if system_cache.exists():
            logger.info("✅ Sistema ya inicializado")
        else:
            logger.info("⚠️ Sistema no inicializado")
            logger.info("   Ejecutar: python main.py init")
        
    except ImportError as e:
        logger.error(f"❌ Error importando sistema: {e}")
        return False
    
    try:
        from integrate_rlhf_pairs import RLHFPairsIntegrator
        logger.info("✅ Integrador importado correctamente")
    except ImportError as e:
        logger.error(f"❌ Error importando integrador: {e}")
        return False
    
    # Resumen
    logger.info("\n" + "="*60)
    logger.info("✅ TEST COMPLETADO EXITOSAMENTE")
    logger.info("="*60)
    
    logger.info("\n📊 Resumen:")
    logger.info(f"   • Categoría:           {category}")
    logger.info(f"   • Productos muestra:   {len(products)}")
    logger.info(f"   • Reviews muestra:     {count}")
    logger.info(f"   • Con rewards:         {len(product_rewards)}")
    
    logger.info("\n🎯 Próximos pasos:")
    logger.info("   1. Si no lo has hecho:")
    logger.info("      python main.py init")
    logger.info("")
    logger.info("   2. Ejecutar pipeline completo:")
    logger.info(f"      python pipeline_reviews_to_rlhf.py --category {category} --limit 10000")
    logger.info("")
    logger.info("   3. O paso a paso:")
    logger.info("      python generate_rlhf_pairs_from_reviews.py")
    logger.info("      python integrate_rlhf_pairs.py")
    logger.info("      python main.py experimento")
    
    logger.info("="*60)
    
    return True


if __name__ == "__main__":
    import sys
    
    try:
        success = quick_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Interrumpido por usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)