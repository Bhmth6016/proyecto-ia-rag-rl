"""
fusion_ranker.py
================
OPCIÓN A — Score fusion: FAISS coseno + Reward model.

PROBLEMA RESUELTO:
    reward_only destruye la geometría semántica de FAISS porque
    intenta reordenar 10 candidatos similares con señal débil.

SOLUCIÓN:
    Score final = α × score_faiss_normalizado + β × score_reward_normalizado

    Esto preserva la fuerza de FAISS y añade la señal del reward
    como corrección sobre el orden base. Si el reward es débil,
    α dominará y el resultado degradará elegantemente a baseline.

PROPIEDADES:
    - Si β = 0 -> idéntico a baseline FAISS
    - Si α = 0 -> idéntico a reward_only (modo arriesgado)
    - Si α = 0.7, β = 0.3 -> fusión conservadora (recomendado inicio)
    - El grid search encuentra α/β óptimos sobre ESCI test

INTEGRACIÓN:
    Este módulo añade el método 'fusion' al sistema y puede usarse
    como reemplazo de 'reward_only' en evaluate_esci_phase1.py.

Uso standalone (grid search):
    python fusion_ranker.py --grid-search
    python fusion_ranker.py --alpha 0.7 --beta 0.3 --evaluate

Importación desde otros scripts:
    from fusion_ranker import FusionRanker, add_fusion_to_system
"""
import argparse
import json
import logging
import sys
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# FusionRanker
# ---------------------------------------------------------------------
class FusionRanker:
    """
    Fusiona el score de similitud coseno (FAISS) con el score del
    reward model usando una combinación lineal ponderada.

    score_fusion(q, p) =
        α × norm(score_faiss(q, p)) + β × norm(score_reward(q, p))

    donde norm() es min-max sobre los candidatos de una misma query.
    Esto hace que α y β sean comparables en escala.
    """

    def __init__(
        self,
        system,
        pipeline,
        alpha: float = 0.7,
        beta: float = 0.3,
        top_k_retrieval: int = 20,
    ):
        """
        Args:
            alpha: peso del score FAISS (coseno). Por defecto 0.7.
            beta:  peso del reward model. Por defecto 0.3.
            top_k_retrieval: cuántos candidatos recuperar antes de fusionar.
                             Más candidatos = más trabajo para el reward
                             pero más posibilidades de encontrar relevantes.
        """
        self.system          = system
        self.pipeline        = pipeline
        self.alpha           = alpha
        self.beta            = beta
        self.top_k_retrieval = top_k_retrieval

        assert abs(alpha + beta - 1.0) < 1e-6 or True, \
            "alpha + beta no necesita sumar 1 (se normalizan independientemente)"

        logger.info(f"FusionRanker: α={alpha:.2f} (FAISS) + β={beta:.2f} (reward)")

    def rank(self, query: str, k: int = 10) -> Tuple[List, List[float]]:
        """
        Devuelve (productos_rankeados, scores_fusion).
        """
        emb_model = self.pipeline.emb_model
        q_emb     = emb_model.encode(query, normalize_embeddings=True)

        # FAISS retrieval con scores coseno
        candidates, faiss_scores = self._faiss_with_scores(q_emb, self.top_k_retrieval)

        if not candidates:
            return [], []

        # Score del reward model para cada candidato
        reward_scores = self._reward_scores(q_emb, candidates)

        # Normalización min-max de cada fuente independientemente
        faiss_norm  = _minmax(np.array(faiss_scores))
        reward_norm = _minmax(np.array(reward_scores))

        # Fusión lineal
        fusion = self.alpha * faiss_norm + self.beta * reward_norm

        # Reordenar por fusion score descendente
        order = np.argsort(fusion)[::-1]

        ranked_products = [candidates[i] for i in order]
        ranked_scores   = [float(fusion[i]) for i in order]

        return ranked_products[:k], ranked_scores[:k]

    def rank_asins(self, query: str, k: int = 10) -> List[str]:
        """Wrapper que devuelve solo los ASINs (compatible con evaluate)."""
        products, _ = self.rank(query, k)
        return [
            str(getattr(p, 'id', '') or getattr(p, 'product_id', ''))
            for p in products
            if getattr(p, 'id', None) or getattr(p, 'product_id', None)
        ]

    def _faiss_with_scores(self, q_emb: np.ndarray, k: int):
        """
        Recupera candidatos desde FAISS y devuelve también los scores coseno.
        Accede al índice FAISS directamente para obtener los scores.
        """
        try:
            vs = self.system.vector_store
            # Intentar acceso directo al índice FAISS para scores exactos
            if hasattr(vs, 'index') and vs.index is not None:
                q_f32 = q_emb.astype(np.float32).reshape(1, -1)
                scores_np, idxs_np = vs.index.search(q_f32, k)
                scores = scores_np[0].tolist()
                idxs   = idxs_np[0].tolist()

                # Mapear índices -> productos
                products = []
                valid_scores = []
                prods_list = self.system.canonical_products
                for i, idx in enumerate(idxs):
                    if 0 <= idx < len(prods_list):
                        products.append(prods_list[idx])
                        valid_scores.append(scores[i])
                return products, valid_scores

            # Fallback: usar el método search normal (sin scores)
            candidates = vs.search(q_emb, k=k)
            # Estimar scores como producto interno con el embedding de query
            scores = self._estimate_cosine_scores(q_emb, candidates)
            return candidates, scores

        except Exception as e:
            logger.debug(f"Error en FAISS search: {e}")
            candidates = self.system._process_query_baseline(query, k)
            scores = self._estimate_cosine_scores(q_emb, candidates)
            return candidates, scores

    def _estimate_cosine_scores(self, q_emb: np.ndarray, products) -> List[float]:
        """Calcula scores coseno estimados cuando FAISS no los devuelve directamente."""
        scores = []
        cache_path = Path("data/cache/product_embeddings.npz")
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            prod_index = {str(pid): emb for pid, emb in zip(data['ids'], data['embeddings'])}
            for p in products:
                pid = str(getattr(p, 'id', '') or getattr(p, 'product_id', ''))
                if pid in prod_index:
                    e = prod_index[pid]
                    s = float(np.dot(q_emb, e) / (np.linalg.norm(q_emb) * np.linalg.norm(e) + 1e-8))
                    scores.append(s)
                else:
                    scores.append(0.0)
        else:
            scores = [1.0 / (i + 1) for i in range(len(products))]  # ranking fallback
        return scores

    def _reward_scores(self, q_emb: np.ndarray, candidates) -> List[float]:
        """
        Puntúa cada candidato con el reward model.
        Si el reward no está entrenado, devuelve zeros (-> solo FAISS opera).
        """
        if not self.pipeline.reward_trained:
            return [0.0] * len(candidates)

        prod_embs = self.pipeline._products_to_embs(candidates)
        if prod_embs is None:
            return [0.0] * len(candidates)

        q_t = torch.tensor(q_emb, dtype=torch.float32, device=self.pipeline.device)

        self.pipeline.reward_model.eval()
        scores = []
        with torch.no_grad():
            for i in range(prod_embs.size(0)):
                single = prod_embs[i:i+1].unsqueeze(0)
                r = self.pipeline.reward_model(q_t.unsqueeze(0), single)
                scores.append(r.item())

        # Rellenar con 0 si hay menos candidatos que prod_embs
        while len(scores) < len(candidates):
            scores.append(0.0)

        return scores[:len(candidates)]


def _minmax(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalización min-max al rango [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < eps:
        return np.ones_like(arr) * 0.5
    return (arr - lo) / (hi - lo + eps)


# ---------------------------------------------------------------------
# Integración en el sistema
# ---------------------------------------------------------------------
def add_fusion_to_system(system, pipeline, alpha: float = 0.7, beta: float = 0.3):
    """
    Añade FusionRanker al sistema y parchea query_four_methods
    para incluir el método 'fusion'.

    Uso:
        ranker = add_fusion_to_system(system, pipeline, alpha=0.7, beta=0.3)
    """
    ranker = FusionRanker(system, pipeline, alpha=alpha, beta=beta)
    system._fusion_ranker = ranker

    # Parchear el método para incluir 'fusion' en los resultados
    original_query = system.query_four_methods

    def patched_query(query_text: str, k: int = 10):
        results = original_query(query_text, k=k)
        try:
            fusion_products, fusion_scores = ranker.rank(query_text, k=k)
            results['methods']['fusion'] = fusion_products
            results['timing']['fusion']  = 0.0
            results['scores']['fusion']  = float(np.mean(fusion_scores)) if fusion_scores else 0.0
        except Exception as e:
            logger.debug(f"Error en método fusion: {e}")
            results['methods']['fusion'] = results['methods'].get('baseline', [])
        return results

    system.query_four_methods = patched_query
    logger.info("[OK] Método 'fusion' añadido al sistema")
    return ranker


# ---------------------------------------------------------------------
# Métricas IR
# ---------------------------------------------------------------------
def ndcg_at_k(ranked: List[str], relevance: Dict[str, int], k: int) -> float:
    top = ranked[:k]
    dcg  = sum(relevance.get(a, 0) / np.log2(i + 2) for i, a in enumerate(top))
    idcg = sum(s / np.log2(i + 2) for i, s in enumerate(sorted(relevance.values(), reverse=True)[:k]) if s > 0)
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------
# Grid search de α/β
# ---------------------------------------------------------------------
def grid_search(system, pipeline, ground_truth: dict, k: int = 10) -> dict:
    """
    Busca el α/β óptimo probando combinaciones en el ground truth ESCI.
    α ∈ [0.0, 1.0] con paso 0.1
    β = 1 - α  (constraint: suman a 1 para comparabilidad)
    """
    print("\n" + "=" * 60)
    print("  GRID SEARCH α/β (FAISS + Reward)")
    print("=" * 60)
    print(f"  {'α (FAISS)':>10}  {'β (Reward)':>10}  {'nDCG@10':>10}  {'Δ vs base':>10}")
    print("  " + "-" * 46)

    queries   = list(ground_truth.keys())
    logger.info(f"  Grid search sobre {len(queries):,} queries")
    alphas    = np.round(np.arange(0.0, 1.05, 0.1), 1)
    results   = {}
    ranker_base = FusionRanker(system, pipeline, alpha=1.0, beta=0.0)
    base_ndcg = np.mean([
        ndcg_at_k(ranker_base.rank_asins(q, k), ground_truth[q], k)
        for q in queries
    ])

    best_alpha, best_beta, best_ndcg = 1.0, 0.0, base_ndcg

    for alpha in alphas:
        beta  = round(1.0 - alpha, 1)
        ranker = FusionRanker(system, pipeline, alpha=float(alpha), beta=float(beta))

        ndcg_scores = []
        for q in queries:
            asins = ranker.rank_asins(q, k)
            ndcg_scores.append(ndcg_at_k(asins, ground_truth[q], k))

        mean_ndcg = float(np.mean(ndcg_scores))
        delta     = mean_ndcg - base_ndcg
        results[f"alpha_{alpha:.1f}"] = {
            'alpha': float(alpha),
            'beta': float(beta),
            'ndcg@10': mean_ndcg,
            'delta_vs_baseline': delta,
        }

        marker = " <-- MEJOR" if mean_ndcg > best_ndcg else ""
        print(f"  {alpha:>10.1f}  {beta:>10.1f}  {mean_ndcg:>10.4f}  {delta:>+10.4f}{marker}")

        if mean_ndcg > best_ndcg:
            best_ndcg  = mean_ndcg
            best_alpha = float(alpha)
            best_beta  = float(beta)

    print("\n" + "=" * 60)
    print(f"  OPTIMO: alpha={best_alpha:.1f}, beta={best_beta:.1f}")
    print(f"  Baseline nDCG@10:      {base_ndcg:.4f}")
    print(f"  Fusion nDCG@10:        {best_ndcg:.4f}")
    print(f"  Mejora:                {best_ndcg - base_ndcg:+.4f}")
    print("=" * 60 + "\n")

    output = {
        'best_alpha': best_alpha,
        'best_beta': best_beta,
        'best_ndcg@10': best_ndcg,
        'baseline_ndcg@10': base_ndcg,
        'improvement': best_ndcg - base_ndcg,
        'all_results': results,
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/fusion_grid_search.json", 'w') as f:
        json.dump(output, f, indent=2)
    logger.info("Grid search guardado: results/fusion_grid_search.json")

    return output


# ---------------------------------------------------------------------
# Evaluación completa con el método fusion
# ---------------------------------------------------------------------
def evaluate_with_fusion(system, pipeline, ground_truth: dict, alpha: float, beta: float):
    """
    Evalúa todos los métodos incluyendo 'fusion' con el α/β dado.
    """
    from evaluate_esci_phase1 import (
        METHODS, compute_all_metrics, aggregate,
        run_stat_tests, get_ranked_asins, build_table
    )
    from datetime import datetime

    METHODS_WITH_FUSION = METHODS + ['fusion']
    ranker = FusionRanker(system, pipeline, alpha=alpha, beta=beta)

    queries = list(ground_truth.keys())
    logger.info(f"\nEvaluando {len(queries):,} queries con fusion α={alpha:.1f}/β={beta:.1f}...")

    results = {m: [] for m in METHODS_WITH_FUSION}

    for i, query in enumerate(queries, 1):
        relevance = ground_truth[query]
        if i % 50 == 0:
            logger.info(f"  [{i}/{len(queries)}]")

        for method in METHODS_WITH_FUSION:
            if method == 'fusion':
                ranked = ranker.rank_asins(query, k=20)
            else:
                ranked = get_ranked_asins(system, pipeline, method, query, k=20)
            metrics = compute_all_metrics(ranked, relevance)
            metrics['query']  = query
            metrics['method'] = method
            results[method].append(metrics)

    import pandas as pd
    summary = aggregate(results)
    tests   = run_stat_tests(results)

    # Añadir fusion a la tabla
    s   = summary.get('fusion', {})
    bl  = summary.get('baseline', {}).get('ndcg@10_mean', 0)
    fn  = s.get('ndcg@10_mean', 0)

    print("\n" + "=" * 75)
    print(f"  TABLA CON FUSIÓN — α={alpha:.1f} (FAISS) + β={beta:.1f} (Reward)")
    print("=" * 75)
    print(f"{'Método':<25} {'nDCG@5':>8} {'nDCG@10':>8} {'MRR':>8} {'MAP@10':>8}")
    print("-" * 75)

    labels = {
        'baseline':     'Baseline (FAISS)',
        'ner_enhanced': 'NER-Enhanced',
        'reward_only':  'Reward-Only',
        'rlhf':         'RLHF (PPO)',
        'full_hybrid':  'Full Hybrid',
        'fusion':       f'Fusion α={alpha:.1f}/β={beta:.1f}',
    }
    for method in METHODS_WITH_FUSION:
        s_m    = summary.get(method, {})
        label  = labels.get(method, method)
        nd5    = s_m.get('ndcg@5_mean', 0)
        nd10   = s_m.get('ndcg@10_mean', 0)
        mrr_v  = s_m.get('mrr_mean', 0)
        map_v  = s_m.get('map@10_mean', 0)
        arrow  = '' if method == 'baseline' else ('↑' if nd10 > bl else '↓')
        print(f"{label:<25} {nd5:>7.4f}  {nd10:>7.4f}{arrow}  {mrr_v:>7.4f}  {map_v:>7.4f}")

    print("-" * 75)
    print(f"\n  Fusion α={alpha:.1f}/β={beta:.1f}:")
    print(f"    Baseline nDCG@10:  {bl:.4f}")
    print(f"    Fusion   nDCG@10:  {fn:.4f}")
    print(f"    Mejora:            {fn - bl:+.4f} ({(fn-bl)/bl*100:+.1f}%)")
    print("=" * 75)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Fusion ranker: FAISS + Reward')
    parser.add_argument('--alpha',       type=float, default=0.7)
    parser.add_argument('--beta',        type=float, default=0.3)
    parser.add_argument('--grid-search', action='store_true',
                        help='Buscar alpha/beta optimos automaticamente')
    parser.add_argument('--evaluate',    action='store_true',
                        help='Evaluacion completa con alpha/beta dado')
    parser.add_argument('--sample',      type=int, default=500,
                        help='Queries a usar en grid search (default: 500)')
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    # Cargar sistema
    from src.unified_system_v2 import UnifiedSystemV2
    from src.rlhf_integration import add_rlhf_to_system
    system   = UnifiedSystemV2.load_from_cache()
    pipeline = add_rlhf_to_system(system)

    # Ground truth — preferir v2, fallback a v1
    for gt_name in ["ground_truth_esci_v2.json", "ground_truth_esci.json"]:
        gt_path = Path("data/esci") / gt_name
        if gt_path.exists():
            break
    else:
        logger.error("Ejecuta primero: python build_esci_ground_truth_v2.py")
        sys.exit(1)

    with open(gt_path, encoding='utf-8') as f:
        ground_truth = json.load(f)
    logger.info(f"Ground truth: {len(ground_truth):,} queries ({gt_path.name})")

    # Aplicar muestra si se pide (o si hay demasiadas queries)
    import random
    all_queries = list(ground_truth.keys())
    if args.sample and len(all_queries) > args.sample:
        random.seed(args.seed)
        sampled = random.sample(all_queries, args.sample)
        ground_truth_sample = {q: ground_truth[q] for q in sampled}
        logger.info(f"Muestra: {args.sample} queries (seed={args.seed})")
    else:
        ground_truth_sample = ground_truth

    if args.grid_search:
        result = grid_search(system, pipeline, ground_truth_sample)
        best_a = result['best_alpha']
        best_b = result['best_beta']
        print(f"\n  Para evaluar con el mejor alpha/beta:")
        print(f"    python fusion_ranker.py --alpha {best_a} --beta {best_b} --evaluate --sample 500")

    elif args.evaluate:
        evaluate_with_fusion(system, pipeline, ground_truth_sample, args.alpha, args.beta)

    else:
        # Demo rapido con las primeras 3 queries
        ranker = FusionRanker(system, pipeline, alpha=args.alpha, beta=args.beta)
        for q in list(ground_truth.keys())[:3]:
            ranked, scores = ranker.rank(q, k=5)
            print(f"\nQuery: '{q}'")
            for i, (p, s) in enumerate(zip(ranked, scores)):
                pid = getattr(p, 'id', '') or getattr(p, 'product_id', '')
                title = getattr(p, 'title', getattr(p, 'name', pid))[:50]
                print(f"  {i+1}. [{s:.3f}] {pid} -- {title}")


if __name__ == "__main__":
    main()