# experimento_completo_4_metodos.py
"""
LIMPIADO: Se eliminó train_rlhf_on_system() que entrenaba el RLHFRankerFixed
(ranker heurístico, no RLHF real).

El método RLHF en el experimento ahora:
- Usa RLHFPipeline (PolicyModel + PPO) si está entrenado
- Si no está entrenado, reporta baseline — honestamente, sin fingir RLHF
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import random
import logging
from datetime import datetime
import sys
import traceback

def setup_directories():
    for d in ['data/cache', 'results', 'logs', 'data/interactions', 'data/backups']:
        Path(d).mkdir(parents=True, exist_ok=True)

def setup_logging():
    Path("logs").mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path("logs") / f"experimento_{timestamp}.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logging.getLogger(__name__)

setup_directories()
logger = setup_logging()

import numpy as np
import pandas as pd


def load_ground_truth() -> Dict[str, List[str]]:
    gt_file = Path("data/interactions/ground_truth_REAL.json")
    if not gt_file.exists():
        logger.error(f" Ground truth no encontrado: {gt_file}")
        logger.info("   Ejecuta: python main.py interactivo")
        sys.exit(1)
    with open(gt_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    logger.info(f" Ground truth: {len(ground_truth)} queries")
    return ground_truth


def split_train_test_stratified(ground_truth: Dict, test_size: float = 0.25,
                                seed: int = 42) -> Tuple[List[str], List[str]]:
    random.seed(seed)
    queries_by_count = {}
    for q, ids in ground_truth.items():
        c = len(ids)
        queries_by_count.setdefault(c, []).append(q)
    train_queries, test_queries = [], []
    for _, queries in queries_by_count.items():
        random.shuffle(queries)
        idx = int(len(queries) * (1 - test_size))
        train_queries.extend(queries[:idx])
        test_queries.extend(queries[idx:])
    random.shuffle(train_queries)
    random.shuffle(test_queries)
    logger.info(f" Split: {len(train_queries)} train, {len(test_queries)} test")
    return train_queries, test_queries


def load_or_create_system_v2() -> Any:
    try:
        from src.unified_system_v2 import UnifiedSystemV2
    except ImportError as e:
        logger.error(f" No se pudo importar UnifiedSystemV2: {e}")
        sys.exit(1)

    system_cache = Path("data/cache/unified_system_v2.pkl")

    if system_cache.exists():
        system = UnifiedSystemV2.load_from_cache()
        if system:
            logger.info(f" Sistema: {len(system.canonical_products):,} productos")

            # Inyectar RLHFPipeline si existe checkpoint
            rlhf_checkpoint = Path("data/cache/rlhf/ppo_trainer.pt")
            if rlhf_checkpoint.exists():
                try:
                    from src.rlhf_integration import add_rlhf_to_system
                    pipeline = add_rlhf_to_system(system)
                    if pipeline.policy_trained:
                        logger.info(" RLHFPipeline cargado — método rlhf usará PolicyModel+PPO")
                    else:
                        logger.info(" RLHFPipeline sin entrenar — método rlhf usará baseline")
                except Exception as e:
                    logger.warning(f" No se pudo cargar RLHFPipeline: {e}")
            else:
                logger.info(" Sin checkpoint RLHF — método rlhf reportará baseline")

            return system

    logger.info(" Creando sistema V2 nuevo...")
    system = UnifiedSystemV2()
    success = system.initialize_with_ner(limit=100000, use_cache=True, use_zero_shot=True)
    if not success:
        sys.exit(1)

    rlhf_checkpoint = Path("data/cache/rlhf/ppo_trainer.pt")
    if rlhf_checkpoint.exists():
        try:
            from src.rlhf_integration import add_rlhf_to_system
            add_rlhf_to_system(system)
        except Exception as e:
            logger.warning(f" No se pudo cargar RLHFPipeline: {e}")

    return system


def calculate_ranking_metrics(ranked_ids: List[str], relevant_ids: List[str],
                               k: int = 5) -> Dict[str, float]:
    if not relevant_ids or k == 0:
        return {'mrr': 0.0, 'precision@k': 0.0, 'recall@k': 0.0, 'ndcg@k': 0.0}
    mrr = next(
        (1.0 / (i + 1) for i, pid in enumerate(ranked_ids) if pid in relevant_ids),
        0.0
    )
    top_k = ranked_ids[:k]
    rel_in_top = [p for p in top_k if p in relevant_ids]
    precision = len(rel_in_top) / k
    recall = len(rel_in_top) / len(relevant_ids)
    dcg = sum(1.0 / np.log2(i + 2) for i, p in enumerate(top_k) if p in relevant_ids)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return {'mrr': mrr, 'precision@k': precision, 'recall@k': recall, 'ndcg@k': ndcg,
            'found': len(rel_in_top)}


_evaluation_cache = {}


def evaluate_method_on_query(system, method: str, query: str,
                              relevant_ids: List[str], k: int = 5) -> Dict[str, Any]:
    try:
        cache_key = f"{query}_{method}_{k}"
        if cache_key not in _evaluation_cache:
            results = system.query_four_methods(query, k=k * 2)
            _evaluation_cache[cache_key] = results['methods'].get(method, [])
        method_results = _evaluation_cache[cache_key]

        if not method_results:
            return {'mrr': 0.0, 'precision@k': 0.0, 'recall@k': 0.0,
                    'ndcg@k': 0.0, 'found': 0, 'success': False}

        ranked_ids = [getattr(p, 'id', None) for p in method_results[:k] if getattr(p, 'id', None)]
        metrics = calculate_ranking_metrics(ranked_ids, relevant_ids, k)
        metrics['success'] = True
        return metrics

    except Exception as e:
        logger.error(f" Error evaluando {method} en '{query}': {e}")
        return {'mrr': 0.0, 'precision@k': 0.0, 'recall@k': 0.0,
                'ndcg@k': 0.0, 'found': 0, 'success': False}


def run_statistical_analysis(results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    try:
        from scipy import stats
        baseline_key = 'baseline'
        if baseline_key not in results or len(results[baseline_key]) < 3:
            return {}
        baseline_mrr = [r['mrr'] for r in results[baseline_key] if 'mrr' in r]
        tests = {}
        for method in ['ner_enhanced', 'rlhf', 'full_hybrid']:
            if method not in results or len(results[method]) < 3:
                continue
            method_mrr = [r['mrr'] for r in results[method] if 'mrr' in r]
            if len(baseline_mrr) != len(method_mrr):
                continue
            t_stat, p_value = stats.ttest_rel(baseline_mrr, method_mrr)
            bm = np.mean(baseline_mrr)
            mm = np.mean(method_mrr)
            diff = mm - bm
            pstd = np.sqrt((np.std(baseline_mrr)**2 + np.std(method_mrr)**2) / 2)
            tests[method] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'cohens_d': float(diff / pstd) if pstd else 0.0,
                'mean_improvement': float(diff),
                'percent_improvement': float(diff / bm * 100) if bm > 0 else 0.0,
                'baseline_mean': float(bm),
                'method_mean': float(mm),
            }
        return tests
    except ImportError:
        logger.warning(" SciPy no instalado.")
        return {}


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (bool, np.bool_)): return bool(o)
        if pd.isna(o): return None
        return super().default(o)


def save_results(results, summary, tests, train_queries, test_queries):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    methods = ['baseline', 'ner_enhanced', 'rlhf', 'full_hybrid']

    def nv(v):
        if isinstance(v, np.floating): return float(v)
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, (bool, np.bool_)): return bool(v)
        try:
            if pd.isna(v): return None
        except Exception:
            pass
        if isinstance(v, np.ndarray): return v.tolist()
        return v

    save_data = {
        'metadata': {'timestamp': timestamp,
                     'train_queries_count': len(train_queries),
                     'test_queries_count': len(test_queries)},
        'summary': {m: {k: nv(v) for k, v in s.items()} for m, s in summary.items()},
        'statistical_tests': {m: {k: nv(v) for k, v in t.items()} for m, t in tests.items()},
    }
    json_file = Path(f"results/experimento_4_metodos_{timestamp}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, cls=EnhancedJSONEncoder)

    rows = []
    for method in methods:
        for i, m in enumerate(results.get(method, [])):
            rows.append({'method': method, 'query_idx': i,
                         'mrr': nv(m.get('mrr', 0)),
                         'precision@5': nv(m.get('precision@k', 0)),
                         'recall@5': nv(m.get('recall@k', 0)),
                         'ndcg@5': nv(m.get('ndcg@k', 0))})
    csv_file = Path(f"results/experimento_4_metodos_{timestamp}.csv")
    if rows:
        pd.DataFrame(rows).to_csv(csv_file, index=False)

    logger.info(f" Resultados: {json_file}")
    return json_file, csv_file, None


def main():
    print("\n" + "="*80)
    print(" EXPERIMENTO: 4 MÉTODOS DE RANKING")
    print("="*80)

    ground_truth = load_ground_truth()
    train_queries, test_queries = split_train_test_stratified(ground_truth, test_size=0.25)

    # NOTA: train_queries se usaban antes para entrenar RLHFRankerFixed.
    # Ahora el entrenamiento RLHF es manual vía: python main.py rlhf
    # Aquí solo evaluamos.

    system = load_or_create_system_v2()

    rlhf_trained = getattr(getattr(system, 'rlhf_pipeline', None), 'policy_trained', False)
    if not rlhf_trained:
        print("\n AVISO: Policy RLHF no entrenada.")
        print("   El método 'rlhf' mostrará los mismos resultados que baseline.")
        print("   Para entrenarlo: python main.py rlhf --preferences -> --train-reward -> --ppo")

    methods = ['baseline', 'ner_enhanced', 'rlhf', 'full_hybrid']
    results = {m: [] for m in methods}

    print(f"\n Evaluando {len(test_queries)} queries...")
    for i, query in enumerate(test_queries, 1):
        relevant_ids = ground_truth.get(query, [])
        if not relevant_ids:
            continue
        if i % 5 == 0 or i == 1:
            print(f"   [{i}/{len(test_queries)}] '{query[:40]}'")
        for method in methods:
            metrics = evaluate_method_on_query(system, method, query, relevant_ids, k=5)
            results[method].append(metrics)

    summary = {}
    for method in methods:
        mr = [r for r in results[method] if r.get('success')]
        if mr:
            df = pd.DataFrame(mr)
            summary[method] = {
                'mrr_mean': float(df['mrr'].mean()),
                'mrr_std': float(df['mrr'].std()),
                'precision_mean': float(df['precision@k'].mean()),
                'recall_mean': float(df['recall@k'].mean()),
                'ndcg_mean': float(df['ndcg@k'].mean()),
                'total_found': int(df['found'].sum()),
                'n_queries': len(mr),
            }
        else:
            summary[method] = {'mrr_mean': 0.0, 'mrr_std': 0.0,
                               'precision_mean': 0.0, 'recall_mean': 0.0,
                               'ndcg_mean': 0.0, 'total_found': 0, 'n_queries': 0}

    tests = run_statistical_analysis(results)

    print("\n" + "="*80)
    print(f"{'Método':<20} {'MRR':<8} {'P@5':<8} {'R@5':<8} {'NDCG@5':<8}")
    print("-"*60)
    for m in methods:
        s = summary.get(m, {})
        print(f"{m.replace('_',' ').title():<20} "
              f"{s.get('mrr_mean',0):.4f}  "
              f"{s.get('precision_mean',0):.4f}  "
              f"{s.get('recall_mean',0):.4f}  "
              f"{s.get('ndcg_mean',0):.4f}")

    save_results(results, summary, tests, train_queries, test_queries)
    print("\n EXPERIMENTO COMPLETADO — resultados en results/")


if __name__ == "__main__":
    main()