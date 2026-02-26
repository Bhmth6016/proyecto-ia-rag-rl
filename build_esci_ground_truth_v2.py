"""
build_esci_ground_truth_v2.py
==============================
Construye ground truth ESCI con el corpus nuevo (100% intersección).

DIFERENCIA vs build_esci_ground_truth.py original:
    Antes: corpus original con 0.14% intersección -> 498 queries útiles
    Ahora: corpus ESCI con 100% intersección -> miles de queries útiles

GARANTÍAS:
    - Solo usa ESCI TEST para el ground truth (jamás ESCI TRAIN)
    - ESCI TRAIN fue usado para construir el corpus y entrenar el reward
    - No hay leakage: test queries nunca vistas en training

SALIDAS:
    data/esci/ground_truth_esci_v2.json
    data/esci/esci_stats_v2.json

Uso:
    python build_esci_ground_truth_v2.py
    python build_esci_ground_truth_v2.py --max-queries 2000
"""
import argparse
import json
import logging
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

LABEL_SCORE = {"Exact": 3, "Substitute": 2, "Complement": 1, "Irrelevant": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-queries', type=int, default=None,
                        help='Máximo de queries en el ground truth (default: todas)')
    parser.add_argument('--min-relevant', type=int, default=1,
                        help='Mínimo de productos Exact/Substitute por query (default: 1)')
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  GROUND TRUTH ESCI v2 (corpus con 100% intersección)")
    print("=" * 65 + "\n")

    # 1. Cargar ASINs del corpus nuevo
    corpus_manifest = Path("data/esci_corpus/corpus_manifest.json")
    if not corpus_manifest.exists():
        logger.error("Ejecuta primero: python build_esci_corpus.py && python init_esci_system.py")
        sys.exit(1)

    asins_file = Path("data/esci_corpus/asins.txt")
    if asins_file.exists():
        corpus_asins = set(asins_file.read_text().strip().split('\n'))
    else:
        import pandas as pd
        df_corpus = pd.read_parquet("data/esci_corpus/products.parquet")
        corpus_asins = set(df_corpus['asin'].astype(str).tolist())

    logger.info(f"ASINs en corpus nuevo: {len(corpus_asins):,}")

    # 2. Cargar ESCI TEST (nunca el train)
    test_path = Path("data/esci/esci_test.parquet")
    if not test_path.exists():
        logger.error("Ejecuta primero: python verify_and_split_esci.py")
        sys.exit(1)

    import pandas as pd
    logger.info("Cargando ESCI test...")
    df_test = pd.read_parquet(test_path)
    if 'product_id' in df_test.columns and 'asin' not in df_test.columns:
        df_test = df_test.rename(columns={'product_id': 'asin'})
    logger.info(f"  ESCI test: {len(df_test):,} filas, {df_test['query'].nunique():,} queries")

    # 3. Filtrar a ASINs en corpus
    df_test['score'] = df_test['esci_label'].map(LABEL_SCORE)
    df_intersect = df_test[df_test['asin'].isin(corpus_asins)].copy()

    n_asins_test = df_test['asin'].nunique()
    n_asins_intersect = df_intersect['asin'].nunique()
    pct = n_asins_intersect / n_asins_test * 100

    logger.info(f"\n  Intersección ESCI test ∩ corpus:")
    logger.info(f"    ASINs en ESCI test:  {n_asins_test:,}")
    logger.info(f"    ASINs en corpus:     {len(corpus_asins):,}")
    logger.info(f"    ASINs en común:      {n_asins_intersect:,} ({pct:.1f}%)")

    if n_asins_intersect == 0:
        logger.error("Intersección vacía. El corpus no contiene ASINs del test.")
        logger.error("Asegúrate de haber ejecutado build_esci_corpus.py con --size suficiente.")
        sys.exit(1)

    # 4. Construir ground truth por query
    logger.info("\nConstruyendo ground truth...")
    ground_truth = {}
    stats = defaultdict(int)

    queries = df_intersect['query'].unique().tolist()
    logger.info(f"  Queries con overlap: {len(queries):,}")

    for query in queries:
        qdf = df_intersect[df_intersect['query'] == query]
        relevance = {}
        for _, row in qdf.iterrows():
            asin  = str(row['asin'])
            score = int(row['score'])
            if asin in corpus_asins:
                relevance[asin] = max(relevance.get(asin, 0), score)

        # Filtro: necesita al menos min_relevant productos con score ≥ 2
        n_relevant = sum(1 for s in relevance.values() if s >= 2)
        if n_relevant < args.min_relevant:
            stats['queries_sin_relevantes'] += 1
            continue

        ground_truth[query] = relevance
        stats['queries_utiles'] += 1

        # Estadísticas de labels
        for score in relevance.values():
            label = {3: 'Exact', 2: 'Substitute', 1: 'Complement', 0: 'Irrelevant'}.get(score, '?')
            stats[f'n_{label.lower()}'] += 1

    # Aplicar límite si se especificó
    if args.max_queries and len(ground_truth) > args.max_queries:
        # Priorizar queries con más productos Exact
        def query_priority(q):
            return sum(1 for s in ground_truth[q].values() if s == 3)
        queries_sorted = sorted(ground_truth.keys(), key=query_priority, reverse=True)
        ground_truth = {q: ground_truth[q] for q in queries_sorted[:args.max_queries]}
        logger.info(f"  Limitado a {len(ground_truth):,} queries más ricas")

    # 5. Guardar
    Path("data/esci").mkdir(exist_ok=True)
    gt_path = Path("data/esci/ground_truth_esci_v2.json")
    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, ensure_ascii=False)
    logger.info(f"  [OK] {gt_path} ({len(ground_truth):,} queries)")

    # Estadísticas extendidas
    n_q = len(ground_truth)
    all_rel = [s for qr in ground_truth.values() for s in qr.values()]
    stats_output = {
        'timestamp': datetime.now().isoformat(),
        'corpus_size': len(corpus_asins),
        'esci_test_asins': n_asins_test,
        'intersection_asins': n_asins_intersect,
        'intersection_pct': pct,
        'queries_utiles': n_q,
        'queries_sin_relevantes': int(stats['queries_sin_relevantes']),
        'n_exact': int(stats.get('n_exact', 0)),
        'n_substitute': int(stats.get('n_substitute', 0)),
        'n_complement': int(stats.get('n_complement', 0)),
        'n_irrelevant': int(stats.get('n_irrelevant', 0)),
        'avg_relevant_per_query': float(np.mean([
            sum(1 for s in qr.values() if s >= 2)
            for qr in ground_truth.values()
        ])) if n_q > 0 else 0,
        'paper_note': (
            f"Ground truth construido intersectando Amazon ESCI Dataset (split=test) "
            f"con corpus ESCI ({len(corpus_asins):,} ASINs). "
            f"Intersección: {n_asins_intersect:,} ASINs ({pct:.1f}%). "
            f"Solo se incluyen queries con ≥{args.min_relevant} producto(s) Exact/Substitute."
        ),
    }
    stats_path = Path("data/esci/esci_stats_v2.json")
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)

    print("\n" + "=" * 65)
    print("  GROUND TRUTH v2 CONSTRUIDO")
    print("=" * 65)
    print(f"  Corpus ASINs:          {len(corpus_asins):,}")
    print(f"  Intersección test:     {n_asins_intersect:,} ({pct:.1f}%)")
    print(f"  Queries útiles:        {n_q:,}")
    print(f"  Labels en corpus:")
    print(f"    Exact (3):           {stats.get('n_exact', 0):,}")
    print(f"    Substitute (2):      {stats.get('n_substitute', 0):,}")
    print(f"    Complement (1):      {stats.get('n_complement', 0):,}")
    print(f"    Irrelevant (0):      {stats.get('n_irrelevant', 0):,}")
    print(f"  Media relevantes/q:    {stats_output['avg_relevant_per_query']:.1f}")
    print(f"\n  SIGUIENTE PASO:")
    print(f"    python pretrain_reward_esci.py")
    print(f"    python evaluate_esci_v2.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()