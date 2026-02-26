# -*- coding: utf-8 -*-
import os; os.environ.setdefault("PYTHONUTF8", "1")
"""
evaluate_esci_v2.py
====================
Evaluación con el corpus ESCI nuevo.
VERSIÓN CORREGIDA - FIX DEFINITIVO para error de dimensiones

CORRECCIONES CLAVE:
    1. Forzar todos los embeddings a shape [384] (no [1,384])
    2. Pasar pool_size correctamente a reward_only_rank
    3. Debug de shapes para verificar
"""
import argparse
import json
import logging
import random
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

METHODS = ['baseline', 'reward_only']

# Cache global
_EMB_CACHE: Optional[Dict[str, np.ndarray]] = None


# ---------------------------------------------------------------------
# Métricas IR
# ---------------------------------------------------------------------
def ndcg_at_k(ranked: List[str], relevance: Dict[str, int], k: int) -> float:
    top = ranked[:k]
    dcg = sum(relevance.get(a, 0) / np.log2(i + 2) for i, a in enumerate(top))
    idcg = sum(s / np.log2(i + 2)
               for i, s in enumerate(sorted(relevance.values(), reverse=True)[:k]) if s > 0)
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked: List[str], relevance: Dict[str, int]) -> float:
    for i, a in enumerate(ranked):
        if relevance.get(a, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def ap_at_k(ranked: List[str], relevance: Dict[str, int], k: int) -> float:
    top = ranked[:k]
    n_rel = sum(1 for s in relevance.values() if s > 0)
    if not n_rel:
        return 0.0
    hits, ap = 0, 0.0
    for i, a in enumerate(top):
        if relevance.get(a, 0) > 0:
            hits += 1
            ap += hits / (i + 1)
    return ap / min(n_rel, k)


def recall_at_k(ranked: List[str], relevance: Dict[str, int], k: int) -> float:
    n_rel = sum(1 for s in relevance.values() if s > 0)
    if not n_rel:
        return 0.0
    return sum(1 for a in ranked[:k] if relevance.get(a, 0) > 0) / n_rel


def all_metrics(ranked: List[str], relevance: Dict[str, int]) -> dict:
    return {
        'ndcg@5': ndcg_at_k(ranked, relevance, 5),
        'ndcg@10': ndcg_at_k(ranked, relevance, 10),
        'mrr': mrr(ranked, relevance),
        'map@10': ap_at_k(ranked, relevance, 10),
        'recall@10': recall_at_k(ranked, relevance, 10),
    }


# ---------------------------------------------------------------------
# Cache de embeddings
# ---------------------------------------------------------------------
def load_embedding_cache() -> Optional[Dict[str, np.ndarray]]:
    """Carga el cache de embeddings de productos"""
    global _EMB_CACHE
    if _EMB_CACHE is not None:
        return _EMB_CACHE

    cache_path = Path("data/cache/product_embeddings.npz")
    if not cache_path.exists():
        logger.warning(f"Cache de embeddings no encontrado: {cache_path}")
        return None

    try:
        data = np.load(cache_path, allow_pickle=True)
        _EMB_CACHE = {str(pid): emb for pid, emb in zip(data['ids'], data['embeddings'])}
        logger.info(f"Cache de embeddings cargado: {len(_EMB_CACHE):,} productos")
        return _EMB_CACHE
    except Exception as e:
        logger.error(f"Error cargando cache: {e}")
        return None


def get_product_embedding(pid: str, cache: Optional[Dict] = None) -> Optional[np.ndarray]:
    """Obtiene embedding de producto, con cache opcional"""
    if cache is not None and pid in cache:
        return cache[pid]
    return None


# ---------------------------------------------------------------------
# Reward-only ranker - VERSIÓN CORREGIDA CON FIX DE SHAPES
# ---------------------------------------------------------------------
def _is_pointwise_model(pipeline) -> bool:
    """Detecta si el reward model es pointwise"""
    model = pipeline.reward_model
    model_type = type(model).__name__.lower()

    if 'pointwise' in model_type:
        return True

    try:
        device = pipeline.device
        q_test = torch.randn(2, 384, device=device)
        p_test = torch.randn(2, 384, device=device)
        with torch.no_grad():
            _ = model(q_test, p_test)
        return True
    except Exception:
        return False

def get_ranked(system, pipeline, method: str, query: str, k: int = 20,
               pool_size: Optional[int] = None) -> List[str]:
    cache_key = f"{method}|{query}|{k}|{pool_size}"
    if cache_key in _rank_cache:
        return _rank_cache[cache_key]

    try:
        if method == 'baseline':
            asins = baseline_rank(system, pipeline, query, k)
        elif method == 'reward_only':
            asins = reward_only_rank(system, pipeline, query, k,
                                     pool_size=pool_size or 50)
        else:
            asins = baseline_rank(system, pipeline, query, k)
    except Exception as e:
        logger.debug(f"Error en get_ranked ({method}): {e}")
        asins = []

    _rank_cache[cache_key] = asins
    return asins

def baseline_rank(system, pipeline, query: str, k: int) -> List[str]:
    """Baseline FAISS"""
    try:
        q_emb = pipeline.emb_model.encode(query, normalize_embeddings=True)
        candidates = system.vector_store.search(q_emb, k=k)
        return _asins(candidates)
    except Exception as e:
        logger.debug(f"Error en baseline_rank: {e}")
        return []


def _asins(products) -> List[str]:
    """Extrae ASINs de productos"""
    asins = []
    for p in products:
        pid = str(getattr(p, 'id', '') or getattr(p, 'product_id', ''))
        if pid:
            asins.append(pid)
    return asins


# ---------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------
_rank_cache: Dict[str, List[str]] = {}


def get_ranked(system, pipeline, method: str, query: str, k: int = 20,
               pool_size: Optional[int] = None) -> List[str]:
    """
    Obtiene ranking para un método, con cache y parámetros específicos

    Args:
        pool_size: Solo se usa para reward_only, controla cuántos candidatos rerankear
    """
    cache_key = f"{method}|{query}|{k}|{pool_size}"
    if cache_key in _rank_cache:
        return _rank_cache[cache_key]

    try:
        if method == 'reward_only':
            # 🔥 PASAR POOL_SIZE EXPLÍCITAMENTE 🔥
            if pool_size is None:
                pool_size = 50  # default
            asins = reward_only_rank(system, pipeline, query, k, pool_size=pool_size)
        else:
            res = system.query_four_methods(query, k=k)
            prods = res.get('methods', {}).get(method, [])
            asins = _asins(prods)
    except Exception as e:
        logger.debug(f"Error en get_ranked ({method}): {e}")
        asins = []

    _rank_cache[cache_key] = asins
    return asins

def evaluate(system, pipeline, ground_truth: Dict[str, Dict[str, int]],
             queries: List[str], pool_size: int = 50) -> Dict[str, List[dict]]:
    results = {m: [] for m in METHODS}
    n = len(queries)
    logger.info(f"Evaluando {n:,} queries — métodos: {METHODS}")
    logger.info(f"Pool size para reward_only: {pool_size}")

    for i, query in enumerate(queries, 1):
        if i % 100 == 0 or i == 1:
            logger.info(f"  [{i}/{n}] {query[:50]}...")
        relevance = ground_truth.get(query, {})
        for method in METHODS:
            ranked = get_ranked(system, pipeline, method, query,
                                k=20, pool_size=pool_size)
            metrics = all_metrics(ranked, relevance)
            metrics.update({'query': query, 'method': method,
                            'ranked_count': len(ranked)})
            results[method].append(metrics)

    return results
# ---------------------------------------------------------------------
# Agregación y estadísticas
# ---------------------------------------------------------------------
def aggregate(results: Dict[str, List[dict]]) -> dict:
    """Agrega resultados por método"""
    summary = {}

    for method, rows in results.items():
        if not rows:
            continue

        df = pd.DataFrame(rows)
        summary[method] = {'n_queries': len(df)}

        for metric in ['ndcg@5', 'ndcg@10', 'mrr', 'map@10', 'recall@10']:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    summary[method][f'{metric}_mean'] = float(values.mean())
                    summary[method][f'{metric}_std'] = float(values.std())
                else:
                    summary[method][f'{metric}_mean'] = 0.0
                    summary[method][f'{metric}_std'] = 0.0

    return summary


def stat_tests(results: Dict[str, List[dict]]) -> dict:
    """Tests estadísticos vs baseline"""
    try:
        from scipy import stats
    except ImportError:
        logger.warning("scipy no instalado, saltando tests estadísticos")
        return {}

    baseline_key = 'baseline'
    if baseline_key not in results:
        return {}

    base_vals = [r.get('ndcg@10', 0) for r in results[baseline_key]]
    if not base_vals:
        return {}

    tests = {}
    base_mean = np.mean(base_vals)

    for method in METHODS[1:]:
        if method not in results:
            continue

        method_vals = [r.get('ndcg@10', 0) for r in results[method]]
        if len(method_vals) != len(base_vals):
            continue

        try:
            t_stat, p_val = stats.ttest_rel(base_vals, method_vals)
            method_mean = np.mean(method_vals)

            tests[method] = {
                'p_value': float(p_val),
                'significant': bool(p_val < 0.05),
                'improvement': float(method_mean - base_mean),
                'improvement_pct': float((method_mean - base_mean) / base_mean * 100) if base_mean > 0 else 0.0,
            }
        except Exception as e:
            logger.debug(f"Error en test estadístico para {method}: {e}")

    return tests


# ---------------------------------------------------------------------
# Tabla de resultados
# ---------------------------------------------------------------------
def print_table(summary: dict, tests: dict, n_queries: int, phase: str,
                pool_size: int) -> str:
    bl = summary.get('baseline', {}).get('ndcg@10_mean', 0)
    ro = summary.get('reward_only', {}).get('ndcg@10_mean', 0)
    delta = ro - bl
    delta_pct = (delta / bl * 100) if bl > 0 else 0

    lines = [
        "",
        "=" * 70,
        f"  FASE 1 — baseline vs reward_only  ({n_queries:,} queries, pool={pool_size})",
        "=" * 70,
        f"{'Método':<22} {'nDCG@5':>9} {'nDCG@10':>9} {'MRR':>9} {'MAP@10':>9} {'R@10':>9}",
        "-" * 70,
    ]

    for method in METHODS:
        s = summary.get(method, {})
        n5  = s.get('ndcg@5_mean', 0)
        n10 = s.get('ndcg@10_mean', 0)
        mrr = s.get('mrr_mean', 0)
        mp  = s.get('map@10_mean', 0)
        r10 = s.get('recall@10_mean', 0)
        label = 'Baseline (FAISS)' if method == 'baseline' else 'Reward-Only'
        lines.append(f"{label:<22} {n5:>9.4f} {n10:>9.4f} {mrr:>9.4f} {mp:>9.4f} {r10:>9.4f}")

    lines += [
        "-" * 70,
        f"  Δ nDCG@10:   {delta:+.4f}  ({delta_pct:+.1f}%)",
    ]

    sig = tests.get('reward_only', {})
    if sig:
        p_val = sig.get('p_value', 1.0)
        lines.append(f"  p-value:     {p_val:.4f} {'(sig. p<0.05)' if p_val < 0.05 else '(no sig.)'}")

    lines += [""]
    if ro > bl:
        lines += [
            "  ✅ FASE 1 SUPERADA — reward mejora baseline.",
            "  → Siguiente: aumentar preferencias y/o proceder a Fase 2.",
        ]
    else:
        lines += [
            "  ❌ FASE 1 NO SUPERADA.",
            "  → Acciones: más preferencias, más epochs, revisar corpus.",
        ]

    lines.append("=" * 70)
    table = "\n".join(lines)
    print(table)
    return table

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluación ESCI v2")
    parser.add_argument('--sample', type=int, default=None,
                        help='Muestra aleatoria de N queries')
    parser.add_argument('--quick', action='store_true',
                        help='Rápido: primeras 100 queries')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true',
                        help='Modo debug con más logging')
    parser.add_argument('--pool-size', type=int, default=50,
                        help='Tamaño del pool para reranking (default: 50)')
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print("\n" + "=" * 65)
    print("  🔥 EVALUACIÓN ESCI v2 (FIX DE SHAPES) 🔥")
    print("=" * 65 + "\n")

    # Verificar ground truth
    gt_path = Path("data/esci/ground_truth_esci_v2.json")
    if not gt_path.exists():
        logger.error("❌ Ground truth no encontrado")
        logger.error("Ejecuta primero: python build_esci_ground_truth_v2.py")
        sys.exit(1)

    # Cargar ground truth
    with open(gt_path, encoding='utf-8') as f:
        ground_truth = json.load(f)
    logger.info(f"📊 Ground truth v2: {len(ground_truth):,} queries")

    # Cargar sistema
    try:
        from src.unified_system_v2 import UnifiedSystemV2
        from src.rlhf_integration import add_rlhf_to_system

        system = UnifiedSystemV2.load_from_cache()
        pipeline = add_rlhf_to_system(system)

        logger.info(f"📦 Sistema: {len(system.canonical_products):,} productos")
        logger.info(f"🤖 Reward model: {type(pipeline.reward_model).__name__}")
        logger.info(f"🎯 Reward trained: {pipeline.reward_trained}")

        # Detectar tipo de modelo
        is_pointwise = _is_pointwise_model(pipeline)
        logger.info(f"📐 Modelo pointwise: {is_pointwise}")
        logger.info(f"🎚️  Pool size: {args.pool_size}")

    except Exception as e:
        logger.error(f"❌ Error cargando sistema: {e}")
        sys.exit(1)

    # Seleccionar queries
    all_queries = list(ground_truth.keys())
    if args.quick:
        queries = all_queries[:100]
        logger.info("⚡ Modo quick: primeras 100 queries")
    elif args.sample:
        random.seed(args.seed)
        queries = random.sample(all_queries, min(args.sample, len(all_queries)))
        logger.info(f"🎲 Muestra aleatoria: {len(queries):,} queries")
    else:
        queries = all_queries
        logger.info(f"📋 Todas las queries: {len(queries):,}")

    # Precargar cache de embeddings
    load_embedding_cache()

    # 🔥 PASAR POOL_SIZE A LA EVALUACIÓN 🔥
    logger.info(f"🚀 Iniciando evaluación con pool_size={args.pool_size}...")
    results = evaluate(system, pipeline, ground_truth, queries,
                       pool_size=args.pool_size)

    # Agregar resultados
    summary = aggregate(results)
    tests = stat_tests(results)

    # Determinar fase
    phase = "FASE 1 (pre-training)" if not pipeline.policy_trained else "FASE 3 (PPO activo)"

    # Mostrar tabla
    table = print_table(summary, tests, len(queries), phase, args.pool_size)

    # Guardar resultados
    Path("results").mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    output = {
        'timestamp': ts,
        'phase': phase,
        'n_queries': len(queries),
        'model_type': type(pipeline.reward_model).__name__,
        'reward_trained': pipeline.reward_trained,
        'policy_trained': pipeline.policy_trained,
        'summary': summary,
        'tests': tests,
        'config': {
            'pool_size': args.pool_size,
            'sample': args.sample,
            'quick': args.quick,
            'seed': args.seed,
        }
    }

    with open(f"results/esci_v2_evaluation_{ts}.json", 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    # CSV
    rows = []
    for method, metrics_list in results.items():
        for metrics in metrics_list:
            row = {'method': method}
            row.update({k: v for k, v in metrics.items()
                       if k not in ['query', 'method']})
            rows.append(row)

    if rows:
        pd.DataFrame(rows).to_csv(f"results/esci_v2_evaluation_{ts}.csv",
                                  index=False, float_format='%.4f')

    # TXT
    with open(f"results/esci_v2_evaluation_{ts}.txt", 'w', encoding='utf-8', errors='replace') as f:
        f.write(table)

    print(f"\n  💾 Resultados guardados en results/esci_v2_evaluation_{ts}.*")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()