"""
evaluate_three_methods.py
=========================
Evaluación académica limpia con 3 métodos:
    1. bm25          — baseline léxico (BM25Okapi)
    2. bm25_rerank   — BM25 pool → reward model reranking [PROPUESTO]
    3. faiss         — dense retrieval upper bound

RQ: ¿Puede el reward model (RLHF) cerrar el gap BM25 → FAISS?
"""
import argparse, json, logging, pickle, random, sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional
from scipy import stats

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Métricas IR
# ------------------------------------------------------------------
def ndcg_at_k(ranked, relevance, k):
    top = ranked[:k]
    dcg = sum(relevance.get(a, 0) / np.log2(i+2) for i, a in enumerate(top))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(s / np.log2(i+2) for i, s in enumerate(ideal) if s > 0)
    return dcg / idcg if idcg > 0 else 0.0

def mrr(ranked, relevance):
    for i, a in enumerate(ranked):
        if relevance.get(a, 0) > 0:
            return 1.0 / (i+1)
    return 0.0

def recall_at_k(ranked, relevance, k):
    n_rel = sum(1 for s in relevance.values() if s > 0)
    if not n_rel:
        return 0.0
    return sum(1 for a in ranked[:k] if relevance.get(a, 0) > 0) / n_rel

def all_metrics(ranked, relevance):
    return {
        'ndcg@5':    ndcg_at_k(ranked, relevance, 5),
        'ndcg@10':   ndcg_at_k(ranked, relevance, 10),
        'mrr':       mrr(ranked, relevance),
        'recall@10': recall_at_k(ranked, relevance, 10),
    }

# ------------------------------------------------------------------
# Rankers
# ------------------------------------------------------------------
def tokenize(text):
    return text.lower().split() if text else []

def bm25_rank(query, bm25, pids, k=20):
    scores = bm25.get_scores(tokenize(query))
    top_idx = np.argsort(scores)[::-1][:k]
    return [(pids[i], float(scores[i])) for i in top_idx]

def faiss_rank(query, system, emb_model, k=20):
    q_emb = emb_model.encode(query, normalize_embeddings=True)
    candidates = system.vector_store.search(q_emb, k=k)
    result = []
    for p in candidates:
        pid = str(getattr(p, 'id', '') or getattr(p, 'product_id', ''))
        score = float(getattr(p, 'score', 0) or getattr(p, 'similarity', 0))
        if pid:
            result.append((pid, score))
    return result

def bm25_reward_rerank(query, bm25, pids, reward_model,
                       emb_model, prod_emb_cache,
                       pool_size=100, k=20, device='cpu'):
    # 1. BM25 pool
    pool = bm25_rank(query, bm25, pids, k=pool_size)
    if not pool:
        return []
    
    pool_pids = [pid for pid, _ in pool]
    
    # 2. Query embedding
    q_np = emb_model.encode(query, normalize_embeddings=True)
    q_np = np.array(q_np).flatten()
    
    # 3. Producto embeddings desde cache
    valid_pids, prod_embs = [], []
    for pid in pool_pids:
        emb = prod_emb_cache.get(pid)
        if emb is not None:
            valid_pids.append(pid)
            prod_embs.append(np.array(emb).flatten())
    
    if not valid_pids:
        return pool_pids[:k]
    
    # 4. Reward scores
    q_t = torch.tensor(q_np, dtype=torch.float32, device=device)
    q_t = q_t.unsqueeze(0).expand(len(valid_pids), -1)
    p_t = torch.tensor(np.stack(prod_embs), dtype=torch.float32, device=device)
    
    reward_model.eval()
    with torch.no_grad():
        scores = reward_model(q_t, p_t).squeeze(-1).cpu().numpy()
    
    # 5. Ordenar por reward score puro (sin interpolación con BM25)
    ranked = sorted(zip(valid_pids, scores), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in ranked[:k]]

# ------------------------------------------------------------------
# Evaluación principal
# ------------------------------------------------------------------
def evaluate(queries, ground_truth, bm25, pids, system,
             emb_model, reward_model, prod_emb_cache,
             pool_size, device):
    
    results = {'bm25': [], 'faiss': [], 'bm25_rerank': []}
    n = len(queries)
    
    for i, query in enumerate(queries, 1):
        if i % 500 == 0 or i == 1:
            logger.info(f"  [{i:,}/{n:,}] {query[:50]}...")
        
        relevance = ground_truth.get(query, {})
        
        # BM25
        bm25_result = [pid for pid, _ in bm25_rank(query, bm25, pids, k=20)]
        results['bm25'].append({
            'query': query, **all_metrics(bm25_result, relevance)
        })
        
        # FAISS
        faiss_result = [pid for pid, _ in faiss_rank(query, system, emb_model, k=20)]
        results['faiss'].append({
            'query': query, **all_metrics(faiss_result, relevance)
        })
        
        # BM25 → Reward rerank
        rerank_result = bm25_reward_rerank(
            query, bm25, pids, reward_model, emb_model,
            prod_emb_cache, pool_size=pool_size, k=20, device=device
        )
        results['bm25_rerank'].append({
            'query': query, **all_metrics(rerank_result, relevance)
        })
    
    return results

def print_results(results, pool_size):
    from datetime import datetime
    
    methods = {
        'bm25':       'BM25 (baseline léxico)',
        'bm25_rerank': f'BM25→Reward (pool={pool_size})',
        'faiss':      'FAISS (upper bound)',
    }
    
    summary = {}
    for method, rows in results.items():
        df = pd.DataFrame(rows)
        summary[method] = {
            m: float(df[m].mean())
            for m in ['ndcg@5', 'ndcg@10', 'mrr', 'recall@10']
        }
    
    print("\n" + "=" * 72)
    print("  RESULTADOS — BM25 vs BM25+Reward vs FAISS")
    print(f"  {len(results['bm25']):,} queries | pool_size={pool_size}")
    print("=" * 72)
    print(f"{'Método':<28} {'nDCG@5':>8} {'nDCG@10':>8} "
          f"{'MRR':>8} {'R@10':>8}")
    print("-" * 72)
    
    for method, label in methods.items():
        s = summary[method]
        print(f"{label:<28} {s['ndcg@5']:>8.4f} {s['ndcg@10']:>8.4f} "
              f"{s['mrr']:>8.4f} {s['recall@10']:>8.4f}")
    
    print("-" * 72)
    
    # Deltas vs BM25
    bm25_n10 = summary['bm25']['ndcg@10']
    for method in ['bm25_rerank', 'faiss']:
        delta = summary[method]['ndcg@10'] - bm25_n10
        pct = delta / bm25_n10 * 100 if bm25_n10 > 0 else 0
        
        # Test estadístico
        base_vals = [r['ndcg@10'] for r in results['bm25']]
        method_vals = [r['ndcg@10'] for r in results[method]]
        _, p_val = stats.ttest_rel(base_vals, method_vals)
        
        sig = "✅ sig." if p_val < 0.05 else "⚠️  no sig."
        label = methods[method]
        print(f"  Δ {label} vs BM25: {delta:+.4f} ({pct:+.1f}%) "
              f"p={p_val:.4f} {sig}")
    
    # Gap closure
    faiss_n10 = summary['faiss']['ndcg@10']
    rerank_n10 = summary['bm25_rerank']['ndcg@10']
    gap_total = faiss_n10 - bm25_n10
    gap_closed = rerank_n10 - bm25_n10
    pct_closed = gap_closed / gap_total * 100 if gap_total > 0 else 0
    
    print(f"\n  Gap BM25→FAISS:    {gap_total:+.4f}")
    print(f"  Gap cerrado:       {gap_closed:+.4f} ({pct_closed:.1f}%)")
    print("=" * 72)
    
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool-size', type=int, default=100)
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--quick', action='store_true',
                        help='Primeras 500 queries')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("\n" + "=" * 72)
    print("  EXPERIMENTO: BM25 vs BM25+RLHF Reward vs FAISS")
    print("  RQ: ¿Puede RLHF cerrar el gap léxico→semántico?")
    print("=" * 72)
    
    # Cargar BM25
    bm25_path = Path("data/cache/bm25_index.pkl")
    if not bm25_path.exists():
        logger.error("BM25 no construido. Ejecuta: python build_bm25_index.py")
        sys.exit(1)
    
    logger.info("Cargando BM25...")
    with open(bm25_path, 'rb') as f:
        bm25_data = pickle.load(f)
    bm25, pids = bm25_data['bm25'], bm25_data['pids']
    logger.info(f"BM25: {len(pids):,} documentos")
    
    # Cargar ground truth
    with open("data/esci/ground_truth_esci_v2.json", encoding='utf-8') as f:
        ground_truth = json.load(f)
    logger.info(f"Ground truth: {len(ground_truth):,} queries")
    
    # Cargar sistema FAISS + reward
    from src.unified_system_v2 import UnifiedSystemV2
    from src.rlhf_integration import add_rlhf_to_system
    
    system = UnifiedSystemV2.load_from_cache()
    pipeline = add_rlhf_to_system(system)
    emb_model = pipeline.emb_model
    reward_model = pipeline.reward_model
    device = pipeline.device
    
    # Cargar prod embeddings
    logger.info("Cargando embeddings de productos...")
    data = np.load("data/cache/product_embeddings.npz", allow_pickle=True)
    prod_emb_cache = {str(pid): emb
                      for pid, emb in zip(data['ids'], data['embeddings'])}
    logger.info(f"Embeddings: {len(prod_emb_cache):,} productos")
    
    # Seleccionar queries
    all_queries = list(ground_truth.keys())
    if args.quick:
        queries = all_queries[:500]
        logger.info("Modo quick: 500 queries")
    elif args.sample:
        random.seed(args.seed)
        queries = random.sample(all_queries, min(args.sample, len(all_queries)))
        logger.info(f"Muestra: {len(queries):,} queries")
    else:
        queries = all_queries
        logger.info(f"Evaluación completa: {len(queries):,} queries")
    
    # Evaluar
    logger.info(f"\nIniciando evaluación (pool_size={args.pool_size})...")
    results = evaluate(
        queries, ground_truth, bm25, pids, system,
        emb_model, reward_model, prod_emb_cache,
        pool_size=args.pool_size, device=device
    )
    
    # Resultados
    summary = print_results(results, args.pool_size)
    
    # Guardar
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("results") / f"experiment_{ts}.json"
    out.parent.mkdir(exist_ok=True)
    
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': ts,
            'n_queries': len(queries),
            'pool_size': args.pool_size,
            'summary': summary,
            'config': vars(args)
        }, f, indent=2)
    
    logger.info(f"\nGuardado: {out}")

if __name__ == "__main__":
    main()