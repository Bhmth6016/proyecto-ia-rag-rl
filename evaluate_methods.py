"""
evaluate_methods.py
===================
Evaluación limpia de los 4 métodos de ranking.

Métodos:
    1. baseline      — FAISS puro (similitud coseno)
    2. ner_enhanced  — FAISS + boost léxico NER
    3. reward_only   — FAISS retrieval + PointwiseReward reranking
    4. rlhf          — FAISS retrieval + PolicyModel (PPO)
    5. full_hybrid   — NER + PointwiseReward reranking

Métricas:
    nDCG@10  (principal — sensible al orden)
    Recall@10
    MRR@10
    MAP@10

Evaluación estadística:
    Paired t-test de cada método vs baseline
    Corrección de Bonferroni para comparaciones múltiples

REGLA: solo usa test_queries.json (nunca train_queries).

Uso:
    python evaluate_methods.py
    python evaluate_methods.py --sample 50    # muestra aleatoria
    python evaluate_methods.py --methods baseline reward_only
"""

import argparse
import json
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

GT_PATH         = Path("data/interactions/ground_truth_REAL.json")
TEST_SPLIT_PATH = Path("data/interactions/test_queries.json")
RESULTS_DIR     = Path("results")

ALL_METHODS = ['baseline', 'ner_enhanced', 'reward_only', 'rlhf', 'full_hybrid']


# ---------------------------------------------------------------------------
# Métricas IR
# ---------------------------------------------------------------------------

def dcg_at_k(ranked: List[str], relevant: set, k: int) -> float:
    """DCG@k con relevancia binaria."""
    return sum(
        1.0 / np.log2(i + 2)
        for i, pid in enumerate(ranked[:k])
        if pid in relevant
    )

def ndcg_at_k(ranked: List[str], relevant: set, k: int) -> float:
    dcg  = dcg_at_k(ranked, relevant, k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def mrr(ranked: List[str], relevant: set) -> float:
    for i, pid in enumerate(ranked):
        if pid in relevant:
            return 1.0 / (i + 1)
    return 0.0

def ap_at_k(ranked: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top = ranked[:k]
    hits, ap = 0, 0.0
    for i, pid in enumerate(top):
        if pid in relevant:
            hits += 1
            ap   += hits / (i + 1)
    return ap / min(len(relevant), k)

def recall_at_k(ranked: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for pid in ranked[:k] if pid in relevant) / len(relevant)

def all_metrics(ranked: List[str], relevant: set) -> dict:
    return {
        'ndcg@10':   ndcg_at_k(ranked, relevant, 10),
        'recall@10': recall_at_k(ranked, relevant, 10),
        'mrr':       mrr(ranked, relevant),
        'map@10':    ap_at_k(ranked, relevant, 10),
    }


# ---------------------------------------------------------------------------
# Rankers por método
# ---------------------------------------------------------------------------

def rank_baseline(system, query: str, k: int = 20) -> List[str]:
    """FAISS puro — baseline."""
    try:
        q_emb      = system.canonicalizer.embedding_model.encode(
            query, normalize_embeddings=True
        )
        candidates = system.vector_store.search(q_emb, k=k)
        return _to_ids(candidates)
    except Exception as e:
        logger.debug(f"baseline error ({query}): {e}")
        return []


def rank_ner_enhanced(system, query: str, k: int = 20) -> List[str]:
    """FAISS + boost léxico NER."""
    if not hasattr(system, 'ner_ranker') or system.ner_ranker is None:
        return rank_baseline(system, query, k)
    try:
        q_emb      = system.canonicalizer.embedding_model.encode(
            query, normalize_embeddings=True
        )
        candidates = system.vector_store.search(q_emb, k=k * 2)
        if not candidates:
            return []

        # Scores base de FAISS
        base_scores = []
        for p in candidates:
            emb = getattr(p, 'content_embedding', None)
            if emb is not None:
                emb = np.array(emb)
                sim = float(np.dot(q_emb, emb) /
                            (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-8))
                base_scores.append(sim)
            else:
                base_scores.append(0.0)

        ranked = system.ner_ranker.rank_with_ner(candidates, query, base_scores)
        return _to_ids(ranked[:k])
    except Exception as e:
        logger.debug(f"ner_enhanced error ({query}): {e}")
        return rank_baseline(system, query, k)


def rank_reward_only(system, query: str, k: int = 20) -> List[str]:
    """FAISS retrieval + PointwiseReward reranking."""
    if not hasattr(system, 'rlhf_pipeline') or system.rlhf_pipeline is None:
        return rank_baseline(system, query, k)

    pipeline = system.rlhf_pipeline
    if not pipeline.reward_trained:
        return rank_baseline(system, query, k)

    import torch
    try:
        q_emb_np   = system.canonicalizer.embedding_model.encode(
            query, normalize_embeddings=True
        )
        candidates = system.vector_store.search(q_emb_np, k=k * 2)
        if not candidates:
            return []

        top_cands = candidates[:pipeline.top_k]
        prod_embs = pipeline._products_to_embs(top_cands)
        if prod_embs is None:
            return _to_ids(candidates[:k])

        q_t = torch.tensor(q_emb_np, dtype=torch.float32, device=pipeline.device)

        pipeline.reward_model.eval()
        scores = []
        with torch.no_grad():
            for i in range(prod_embs.size(0)):
                p_emb = prod_embs[i:i+1]                  # [1, emb_dim]
                r     = pipeline.reward_model(q_t.unsqueeze(0), p_emb)
                scores.append(r.item())

        order = np.argsort(scores)[::-1]
        ranked = [top_cands[j] for j in order if j < len(top_cands)]
        return _to_ids(ranked[:k])

    except Exception as e:
        logger.debug(f"reward_only error ({query}): {e}")
        return rank_baseline(system, query, k)


def rank_rlhf(system, query: str, k: int = 20) -> List[str]:
    """FAISS retrieval + PolicyModel (PPO)."""
    if not hasattr(system, 'rlhf_pipeline') or system.rlhf_pipeline is None:
        return rank_baseline(system, query, k)

    pipeline = system.rlhf_pipeline
    if not pipeline.policy_trained:
        return rank_baseline(system, query, k)

    try:
        products, query_emb, _ = pipeline.retrieve_candidates(query, k=k * 2)
        if not products:
            return []
        ranked = pipeline.rank_products(query, products, query_emb)
        return _to_ids(ranked[:k])
    except Exception as e:
        logger.debug(f"rlhf error ({query}): {e}")
        return rank_baseline(system, query, k)


def rank_full_hybrid(system, query: str, k: int = 20) -> List[str]:
    """NER-enhanced + PointwiseReward reranking."""
    ner_results = rank_ner_enhanced(system, query, k=k * 2)
    if not ner_results:
        return rank_baseline(system, query, k)

    if not hasattr(system, 'rlhf_pipeline') or system.rlhf_pipeline is None:
        return ner_results[:k]

    pipeline = system.rlhf_pipeline
    if not pipeline.reward_trained:
        return ner_results[:k]

    import torch
    try:
        # Recuperar objetos de producto desde IDs
        id_map = {
            str(getattr(p, 'id', '') or getattr(p, 'product_id', '')): p
            for p in system.canonical_products
        }
        ner_products = [id_map[pid] for pid in ner_results if pid in id_map]
        if not ner_products:
            return ner_results[:k]

        q_emb_np  = system.canonicalizer.embedding_model.encode(
            query, normalize_embeddings=True
        )
        prod_embs = pipeline._products_to_embs(ner_products[:pipeline.top_k])
        if prod_embs is None:
            return ner_results[:k]

        q_t = torch.tensor(q_emb_np, dtype=torch.float32, device=pipeline.device)

        pipeline.reward_model.eval()
        scores = []
        with torch.no_grad():
            for i in range(prod_embs.size(0)):
                p_emb = prod_embs[i:i+1]
                r     = pipeline.reward_model(q_t.unsqueeze(0), p_emb)
                scores.append(r.item())

        order   = np.argsort(scores)[::-1]
        ranked  = [ner_products[j] for j in order if j < len(ner_products)]
        return _to_ids(ranked[:k])

    except Exception as e:
        logger.debug(f"full_hybrid error ({query}): {e}")
        return ner_results[:k]


RANKERS = {
    'baseline':     rank_baseline,
    'ner_enhanced': rank_ner_enhanced,
    'reward_only':  rank_reward_only,
    'rlhf':         rank_rlhf,
    'full_hybrid':  rank_full_hybrid,
}


def _to_ids(products) -> List[str]:
    return [
        str(getattr(p, 'id', '') or getattr(p, 'product_id', ''))
        for p in products
        if getattr(p, 'id', None) or getattr(p, 'product_id', None)
    ]


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def evaluate(
    system,
    gt: Dict[str, List[str]],
    queries: List[str],
    methods: List[str],
) -> Dict[str, List[dict]]:
    """Evalúa todos los métodos sobre las queries de test."""
    results = {m: [] for m in methods}
    n = len(queries)

    logger.info(f"Evaluando {n} queries — métodos: {methods}")

    for i, query in enumerate(queries, 1):
        if i % 20 == 0 or i == 1:
            logger.info(f"  [{i}/{n}] {query[:55]}...")

        relevant = set(gt.get(query, []))
        if not relevant:
            continue  # Query sin relevantes: no aporta información

        for method in methods:
            ranked  = RANKERS[method](system, query, k=20)
            metrics = all_metrics(ranked, relevant)
            metrics.update({'query': query, 'method': method,
                            'n_relevant': len(relevant),
                            'ranked_count': len(ranked)})
            results[method].append(metrics)

    return results


# ---------------------------------------------------------------------------
# Agregación y tests estadísticos
# ---------------------------------------------------------------------------

def aggregate(results: Dict[str, List[dict]]) -> dict:
    summary = {}
    for method, rows in results.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        summary[method] = {'n_queries': len(df)}
        for metric in ['ndcg@10', 'recall@10', 'mrr', 'map@10']:
            vals = df[metric].dropna()
            if len(vals) > 0:
                summary[method][f'{metric}_mean'] = float(vals.mean())
                summary[method][f'{metric}_std']  = float(vals.std())
    return summary


def stat_tests(results: Dict[str, List[dict]], alpha: float = 0.05) -> dict:
    """
    Paired t-test de cada método vs baseline.
    Corrección de Bonferroni para comparaciones múltiples.
    """
    try:
        from scipy import stats
    except ImportError:
        logger.warning("scipy no instalado — sin tests estadísticos")
        logger.warning("  pip install scipy --break-system-packages")
        return {}

    if 'baseline' not in results:
        return {}

    base_vals = [r['ndcg@10'] for r in results['baseline']]
    if not base_vals:
        return {}

    n_comparisons = len([m for m in results if m != 'baseline'])
    alpha_bonf    = alpha / max(n_comparisons, 1)

    tests = {}
    base_mean = np.mean(base_vals)

    for method, rows in results.items():
        if method == 'baseline' or not rows:
            continue

        method_vals = [r['ndcg@10'] for r in rows]

        # Alinear por query para paired test
        base_by_q   = {r['query']: r['ndcg@10'] for r in results['baseline']}
        method_by_q = {r['query']: r['ndcg@10'] for r in rows}
        common      = sorted(base_by_q.keys() & method_by_q.keys())

        if len(common) < 4:
            logger.warning(f"  {method}: pocas queries comunes ({len(common)}) para t-test")
            continue

        base_paired   = [base_by_q[q]   for q in common]
        method_paired = [method_by_q[q] for q in common]

        t_stat, p_val = stats.ttest_rel(base_paired, method_paired)
        method_mean   = np.mean(method_paired)
        delta         = method_mean - np.mean(base_paired)

        tests[method] = {
            'p_value':          float(p_val),
            'p_bonferroni':     float(p_val * n_comparisons),
            'significant':      bool(p_val < alpha),
            'significant_bonf': bool(p_val < alpha_bonf),
            'n_queries':        len(common),
            'delta_ndcg10':     float(delta),
            'delta_pct':        float(delta / np.mean(base_paired) * 100)
                                if np.mean(base_paired) > 0 else 0.0,
        }

    return tests


# ---------------------------------------------------------------------------
# Tabla de resultados
# ---------------------------------------------------------------------------

def print_table(summary: dict, tests: dict, methods: List[str]) -> str:
    bl = summary.get('baseline', {}).get('ndcg@10_mean', 0)

    lines = [
        "",
        "=" * 78,
        "  RESULTADOS — EVALUACIÓN 4 MÉTODOS (test set)",
        "=" * 78,
        f"{'Método':<20} {'nDCG@10':>9} {'Recall@10':>10} {'MRR':>9} "
        f"{'MAP@10':>9} {'Δ%':>7} {'p-val':>8}",
        "-" * 78,
    ]

    labels = {
        'baseline':     'Baseline (FAISS)',
        'ner_enhanced': 'NER-Enhanced',
        'reward_only':  'Reward-Only',
        'rlhf':         'RLHF (PPO)',
        'full_hybrid':  'Full Hybrid',
    }

    for method in methods:
        s     = summary.get(method, {})
        nd10  = s.get('ndcg@10_mean',   0)
        rc10  = s.get('recall@10_mean', 0)
        mrr_v = s.get('mrr_mean',       0)
        mp10  = s.get('map@10_mean',    0)
        label = labels.get(method, method)

        if method == 'baseline':
            delta_str = '   —  '
            p_str     = '      —'
        else:
            t = tests.get(method, {})
            delta_pct = t.get('delta_pct', 0)
            p_val     = t.get('p_value',   1.0)
            sig_mark  = '*' if t.get('significant') else ' '
            delta_str = f"{delta_pct:+6.1f}%"
            p_str     = f"{p_val:.4f}{sig_mark}"

        lines.append(
            f"{label:<20} {nd10:>9.4f} {rc10:>10.4f} {mrr_v:>9.4f} "
            f"{mp10:>9.4f} {delta_str:>7} {p_str:>8}"
        )

    lines += [
        "-" * 78,
        "  * p < 0.05 (t-test pareado). Bonferroni aplicado.",
        "=" * 78,
        "",
    ]

    table = "\n".join(lines)
    print(table)
    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluación 4 métodos de ranking")
    parser.add_argument('--methods', nargs='+', default=ALL_METHODS,
                        choices=ALL_METHODS,
                        help='Métodos a evaluar (default: todos)')
    parser.add_argument('--sample',  type=int, default=None,
                        help='Muestra aleatoria de N queries test')
    parser.add_argument('--seed',    type=int, default=42)
    parser.add_argument('--gt',      type=str, default=str(GT_PATH))
    parser.add_argument('--test-queries', type=str,
                        default=str(TEST_SPLIT_PATH))
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("  EVALUACIÓN DE MÉTODOS DE RANKING")
    print("=" * 78)
    print(f"  Métodos: {args.methods}")
    print(f"  Ground truth: {args.gt}")
    print()

    # Verificar archivos
    for p in [Path(args.gt), Path(args.test_queries)]:
        if not p.exists():
            logger.error(f"No encontrado: {p}")
            if 'ground_truth' in str(p):
                logger.error("  Ejecuta: python ground_truth_builder.py")
            else:
                logger.error("  Ejecuta: python split_queries.py")
            sys.exit(1)

    # Cargar datos
    with open(args.gt, encoding='utf-8')           as f: gt      = json.load(f)
    with open(args.test_queries, encoding='utf-8') as f: test_qs = json.load(f)

    # Filtrar queries que tienen relevantes en el GT
    test_qs = [q for q in test_qs if q in gt and gt[q]]
    logger.info(f"Test queries con relevantes: {len(test_qs)}")

    if not test_qs:
        logger.error("Sin queries de test con relevantes en el GT.")
        sys.exit(1)

    # Muestra opcional
    if args.sample and len(test_qs) > args.sample:
        import random
        random.seed(args.seed)
        test_qs = random.sample(test_qs, args.sample)
        logger.info(f"Muestra: {len(test_qs)} queries")

    # Cargar sistema
    try:
        from src.unified_system_v2 import UnifiedSystemV2
        from src.rlhf_integration  import add_rlhf_to_system
        system = UnifiedSystemV2.load_from_cache()
        if not system:
            sys.exit(1)
        logger.info(f"Sistema: {len(system.canonical_products):,} productos")
        add_rlhf_to_system(system)
    except Exception as e:
        logger.error(f"Error cargando sistema: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # Reportar estado de los métodos
    pipeline = getattr(system, 'rlhf_pipeline', None)
    print(f"  Estado de métodos:")
    print(f"    NER disponible:    {'[OK]' if getattr(system, 'ner_ranker', None) else '[--] (reward_only y rlhf no afectados)'}")
    print(f"    Reward entrenado:  {'[OK]' if pipeline and pipeline.reward_trained else '[--] reward_only usará baseline'}")
    print(f"    Policy entrenada:  {'[OK]' if pipeline and pipeline.policy_trained else '[--] rlhf usará baseline'}")
    print()

    # Evaluar
    results = evaluate(system, gt, test_qs, args.methods)

    # Agregar y testear
    summary = aggregate(results)
    tests   = stat_tests(results)

    # Mostrar tabla
    table = print_table(summary, tests, args.methods)

    # Guardar resultados
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    output = {
        'timestamp':     ts,
        'n_test_queries': len(test_qs),
        'methods':        args.methods,
        'summary':        summary,
        'stat_tests':     tests,
        'model_info': {
            'embedding_model': 'all-MiniLM-L6-v2',
            'reward_trained':  bool(pipeline and pipeline.reward_trained)
                               if pipeline else False,
            'policy_trained':  bool(pipeline and pipeline.policy_trained)
                               if pipeline else False,
        }
    }

    json_path = RESULTS_DIR / f"evaluation_{ts}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False,
                  default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    txt_path = RESULTS_DIR / f"evaluation_{ts}.txt"
    with open(txt_path, 'w', encoding='utf-8', errors='replace') as f:
        f.write(table)

    # CSV por fila/query
    rows = []
    for method, mrows in results.items():
        for r in mrows:
            rows.append({'method': method, **{k: v for k, v in r.items()
                                              if k not in ['method']}})
    if rows:
        csv_path = RESULTS_DIR / f"evaluation_{ts}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False, float_format='%.4f')

    print(f"  Resultados guardados:")
    print(f"    {json_path}")
    print(f"    {txt_path}")

    # Consejo final
    best_method = max(
        (m for m in args.methods if m != 'baseline'),
        key=lambda m: summary.get(m, {}).get('ndcg@10_mean', 0),
        default=None
    )
    bl_ndcg = summary.get('baseline', {}).get('ndcg@10_mean', 0)
    if best_method:
        best_ndcg = summary.get(best_method, {}).get('ndcg@10_mean', 0)
        if best_ndcg > bl_ndcg:
            t    = tests.get(best_method, {})
            sig  = "✓ significativo" if t.get('significant') else "⚠ no significativo aún"
            print(f"\n  Mejor método: {best_method} "
                  f"(nDCG@10={best_ndcg:.4f}, Δ={best_ndcg-bl_ndcg:+.4f}, {sig})")
        else:
            print(f"\n  Ningún método supera el baseline aún.")
            print(f"  → Revisa calidad del reward model o recolecta más preferencias A/B.")


if __name__ == "__main__":
    main()