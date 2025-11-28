#!/usr/bin/env python3
"""
deepeval.py
Script de evaluación para las 4 modalidades de tu sistema RAG/RLHF/híbrido.

Usage:
    python deepeval.py --mode basic
    python deepeval.py --mode all
"""
from pathlib import Path
import os
import sys
import json
import glob
import time
import argparse
import logging
from collections import defaultdict, Counter
from math import log2# configuración de logging del proyect
# --- Configuración logger ---
logging.basicConfig(
    filename="logs/amazon_recommendations.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)
# --- Helpers de métricas (implementaciones simples, sin librerías externas) ---
def mean(iterable):
    s = 0
    n = 0
    for v in iterable:
        s += v; n += 1
    return s / n if n else 0.0

def mrr_at_k(retrieved_lists, ground_truth_sets, k=10):
    """Mean Reciprocal Rank @k.
    retrieved_lists: list of lists (ids) per query
    ground_truth_sets: list of sets (relevant ids)
    """
    rr = []
    for res, gt in zip(retrieved_lists, ground_truth_sets):
        found = 0.0
        for i, doc in enumerate(res[:k], start=1):
            if doc in gt:
                found = 1.0 / i
                break
        rr.append(found)
    return mean(rr)

def dcg_at_k(ranked_list, gt_set, k=10):
    dcg = 0.0
    for i, doc in enumerate(ranked_list[:k], start=1):
        rel = 1.0 if doc in gt_set else 0.0
        if i == 1:
            dcg += rel
        else:
            dcg += rel / log2(i)
    return dcg

def ndcg_at_k(retrieved_lists, ground_truth_sets, k=10):
    ndcgs = []
    for res, gt in zip(retrieved_lists, ground_truth_sets):
        idcg = dcg_at_k(list(gt), gt, k)  # ideal: put all relevant first (approx)
        dcg = dcg_at_k(res, gt, k)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return mean(ndcgs)

# BLEU-1 (unigram precision) simplified
def bleu1(candidate, references):
    c_tokens = candidate.split()
    if not c_tokens:
        return 0.0
    ref_tokens = [r.split() for r in references]
    ref_counts = Counter()
    for r in ref_tokens:
        ref_counts |= Counter(r)
    cand_counts = Counter(c_tokens)
    clipped = sum(min(cand_counts[w], ref_counts[w]) for w in cand_counts)
    precision = clipped / len(c_tokens)
    # brevity penalty
    ref_lens = [len(r) for r in ref_tokens]
    closest_ref_len = min(ref_lens, key=lambda x: (abs(x - len(c_tokens)), x))
    bp = 1.0 if len(c_tokens) > closest_ref_len else \
         (len(c_tokens)/closest_ref_len) if closest_ref_len>0 else 1.0
    return precision * bp

# ROUGE-L (LCS-based) simplified
def lcs(a, b):
    # dynamic programming LCS length
    la, lb = len(a), len(b)
    dp = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la-1,-1,-1):
        for j in range(lb-1,-1,-1):
            if a[i]==b[j]:
                dp[i][j] = 1 + dp[i+1][j+1]
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j+1])
    return dp[0][0]

def rouge_l(candidate, references):
    c_tokens = candidate.split()
    if not c_tokens:
        return 0.0
    scores = []
    for r in references:
        r_tokens = r.split()
        l = lcs(c_tokens, r_tokens)
        prec = l / max(1, len(c_tokens))
        rec  = l / max(1, len(r_tokens))
        if prec + rec == 0:
            scores.append(0.0)
        else:
            scores.append((2 * prec * rec) / (prec + rec))
    return max(scores)

# --- I/O utilities ---
def load_jsonl(path):
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data.append(json.loads(line))
            except Exception:
                logging.exception("Error leyendo jsonl linea: %s", line[:200])
    return data

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_jsonl(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# --- Intenta cargar tus módulos del proyecto; si fallan registra y usa stubs ---
MISSING_IMPLEMENTATIONS = []
def try_import(path_module, attr=None):
    """
    path_module: e.g. 'src.core.rag.basic.retriever'
    attr: optional attribute name to get from the module
    """
    try:
        # ensure repo root is on path
        repo_root = os.getcwd()
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        mod = __import__(path_module, fromlist=['*'])
        return getattr(mod, attr) if attr else mod
    except Exception as e:
        MISSING_IMPLEMENTATIONS.append((path_module, attr, str(e)))
        logging.warning("No se pudo importar %s (attr=%s): %s", path_module, attr, e)
        return None

# --- Carga de implementaciones reales (intentos) ---
Retriever = try_import("src.core.rag.basic.retriever", "Retriever")
WorkingAgent = try_import("src.core.rag.advanced.WorkingRAGAgent", "WorkingAdvancedRAGAgent")
UserManager = try_import("src.core.data.user_manager", "UserManager")
CollaborativeFilter = try_import("src.core.rag.advanced.collaborative_filter", "CollaborativeFilter")
UserProfile = try_import("src.core.data.user_models", "UserProfile")
RLHFTrainer = try_import("src.core.rag.advanced.trainer", "RLHFTrainer")
RLHFMonitor = try_import("src.core.rag.advanced.RLHFMonitor", "RLHFMonitor")
FeedbackProcessor = try_import("src.core.rag.advanced.feedback_processor", "FeedbackProcessor")
ScoreNormalizer = try_import("src.core.scoring.score_normalizer", "ScoreNormalizer")

# --- Stubs (si faltan implementaciones) ---
class StubRetriever:
    def __init__(self, *a, **k): pass
    def retrieve(self, query, top_k=10):
        # returns list of (product_id, score)
        return [("prod_stub_%d" % i, 1.0/(i+1)) for i in range(top_k)]
    def _expand_query(self, q): return q
    def _score(self, docs): return docs
    def update_feedback_weights_immediately(self, *a, **k): pass
    def _apply_temporal_decay(self, *a, **k): pass
    def _save_feedback_weights(self, *a, **k): pass

class StubAgent:
    def __init__(self, config=None): pass
    def process_query(self, text, user_id=None):
        # returns response_text and list of product ids recommended
        return "respuesta_stub sobre " + text, ["prod_stub_0", "prod_stub_1"]
    def _filter_gaming_products(self, products): return products
    def _generate_gaming_response(self, products): return "desc_stub"
    def _rerank_with_rlhf(self, *a, **k): return []
    def _retrain_with_feedback(self, *a, **k): pass
    def _count_recent_feedback(self, *a, **k): return 0
    def _check_and_retrain(self, *a, **k): pass
    def _infer_selected_product(self, *a, **k): return None
    def log_feedback(self, *a, **k): pass
    def _calculate_dynamic_weights(self, *a, **k): return (0.6, 0.4)

class StubUserManager:
    """Stub mejorado que simila usuarios similares básicos."""
    def __init__(self, users=None):
        self._users = users or []
        # Crear algunos perfiles de usuario simulados si no hay datos
        if not self._users:
            self._users = [
                {"id": f"user_sim_{i}", "categories": ["gaming", "electronics"]} 
                for i in range(10)
            ]
    
    def find_similar_users(self, target_user, min_similarity=0.6):
        """
        Devuelve usuarios simulados con similitud básica por categorías.
        """
        similar = []
        target_categories = set(target_user.get("categories", [])) if isinstance(target_user, dict) else set()
        
        for user in self._users:
            if user.get("id") == target_user.get("id") if isinstance(target_user, dict) else False:
                continue  # saltar el mismo usuario
            
            user_categories = set(user.get("categories", []))
            # Similitud simple: Jaccard similarity
            if target_categories:
                intersection = len(target_categories & user_categories)
                union = len(target_categories | user_categories)
                similarity = intersection / union if union > 0 else 0
            else:
                similarity = 0.3  # similitud por defecto si no hay categorías
            
            if similarity >= min_similarity:
                similar.append(user)
        
        return similar[:5]  # máximo 5 usuarios similares
class StubCollaborative:
    def __init__(self, user_manager=None): 
        self.user_manager = user_manager
        self.fallback_scores = {}
        
    def get_collaborative_scores(self, user_id, candidates):
        """
        Stub mejorado que simula scores colaborativos realistas
        """
        if not candidates:
            return {}
            
        scores = {}
        
        # Simular preferencias de usuario basadas en el ID
        user_preferences = {
            "user_0": ["games", "consoles"],
            "user_1": ["peripherals", "audio"], 
            "user_2": ["monitors", "gaming"],
            "default": ["games", "gaming"]
        }
        
        # Determinar preferencias del usuario
        prefs = user_preferences.get(user_id, user_preferences["default"])
        
        # Asignar scores basados en simulación de preferencias
        for i, candidate in enumerate(candidates):
            base_score = 0.3  # Score base
            
            # Boost por posición (los primeros resultados son más relevantes)
            position_boost = 0.4 / (i + 1)
            
            # Boost simulado por "preferencias de usuario"
            preference_boost = 0.0
            if any(pref in str(candidate).lower() for pref in prefs):
                preference_boost = 0.3
                
            # Score final
            final_score = base_score + position_boost + preference_boost
            scores[candidate] = min(final_score, 0.9)  # Cap en 0.9
            
        return scores
class StubUserProfile:
    def __init__(self, *a, **k): pass
    def calculate_similarity(self, other):
        return 0.5

class StubTrainer:
    def __init__(self, *a, **k): pass
    def prepare_rlhf_dataset_from_logs(self, logs):
        return {"n_examples": len(logs)}
    def train(self, dataset):
        return {"trained_examples": dataset.get("n_examples", 0), "loss": 0.1}

class StubMonitor:
    def log_training_session(self, session): pass
    def get_training_stats(self): return {}

# asigna stubs según sea necesario
if Retriever is None:
    Retriever = StubRetriever
if WorkingAgent is None:
    WorkingAgent = StubAgent
if CollaborativeFilter is None:
    CollaborativeFilter = StubCollaborative
if UserProfile is None:
    UserProfile = StubUserProfile
if RLHFTrainer is None:
    RLHFTrainer = StubTrainer
if RLHFMonitor is None:
    RLHFMonitor = StubMonitor
if ScoreNormalizer is None:
    ScoreNormalizer = None  # opcional

# --- Data paths (según tu descripción) ---
PATH_SUCCESS_LOG = "data/feedback/success_queries.log"
PATH_FAILED_LOG = "data/feedback/failed_queries.log"
PATH_USERS_GLOB = "data/users/*.json"
PATH_PRODUCTS = "data/processed/products.json"
PATH_FEEDBACK_WEIGHTS = "data/feedback/feedback_weights.json"
PATH_RLHF_METRICS = "data/feedback/rlhf_metrics/training_metrics.jsonl"

# --- Load datasets helpers ---
def load_users():
    users = []
    for p in glob.glob(PATH_USERS_GLOB):
        try:
            with open(p, "r", encoding="utf-8") as f:
                users.append(json.load(f))
        except Exception:
            logging.exception("Error leyendo user %s", p)
    return users

def load_products():
    if os.path.exists(PATH_PRODUCTS):
        with open(PATH_PRODUCTS, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # fallback: scan data/raw
        prods = []
        for p in glob.glob("data/raw/*.jsonl"):
            prods.extend(load_jsonl(p))
        return prods

# --- Evaluation procedures por modalidad ---
def eval_basic(agent, retriever, queries, gt_map, top_k=10):
    """
    queries: list of query strings
    gt_map: list of sets with relevant product ids per query (aligned)
    """
    t0 = time.perf_counter()
    retrieved_lists = []
    responses = []
    for q in queries:
        # retrieval
        docs = retriever.retrieve(q, top_k=top_k)
        # docs may be list of tuples (id,score)
        if docs and isinstance(docs[0], (list, tuple)):
            doc_ids = [d[0] for d in docs]
        else:
            doc_ids = list(docs)
        retrieved_lists.append(doc_ids)
        # agent response (no reranking)
        resp = agent.process_query(q)
        resp_text = resp.text
        recommended = resp.recommended
        responses.append((resp_text, recommended))
    elapsed = time.perf_counter() - t0

    # metrics
    mrr = mrr_at_k(retrieved_lists, gt_map, k=top_k)
    ndcg = ndcg_at_k(retrieved_lists, gt_map, k=top_k)
    # quality metrics: compare textual responses with ground truth texts if provided
    # For simplicity assume gt_map_text is not provided -> compute BLEU/ROUGE on recommended product titles if available
    bleu_scores = []
    rouge_scores = []
    for resp, gt in zip(responses, gt_map):
        resp_text = resp[0]
        # create dummy references from groundtruth ids names if products known
        refs = []
        for pid in list(gt)[:3]:
            refs.append(str(pid))  # fallback: use id string
        if refs:
            bleu_scores.append(bleu1(resp_text, refs))
            rouge_scores.append(rouge_l(resp_text, refs))
    return {
        "time_seconds": elapsed,
        "mrr": mrr,
        "ndcg": ndcg,
        "bleu1": mean(bleu_scores) if bleu_scores else 0.0,
        "rouge_l": mean(rouge_scores) if rouge_scores else 0.0,
        "retrieved_count_avg": mean([len(r) for r in retrieved_lists])
    }

def eval_collaborative(agent, retriever, collab, users, queries, gt_map, top_k=20):
    t0 = time.perf_counter()
    retrieved_lists = []
    collab_scores_times = []
    hits = []
    
    for i, (q, user) in enumerate(zip(queries, users)):
        try:
            # 🔥 CORRECCIÓN: Asegurar que user tenga ID
            if not user or not user.get("id"):
                user_id = f"default_user_{i}"
            else:
                user_id = user.get("id")
                
            docs = retriever.retrieve(q, top_k=top_k)
            doc_ids = [d[0] if isinstance(d, (list,tuple)) else d for d in docs]
            
            # measure collaborative ranking time
            t1 = time.perf_counter()
            
            # 🔥 CORRECCIÓN: Pasar user_id en lugar del objeto user completo
            collab_scores = collab.get_collaborative_scores(user_id, doc_ids)
            
            t2 = time.perf_counter()
            collab_scores_times.append(t2-t1)
            
            # merge/weighted rerank: 60% collab / 40% rag (simple)
            reranked = sorted(doc_ids, 
                            key=lambda pid: (collab_scores.get(pid,0)*0.6 + 0.4*(1/(1+doc_ids.index(pid)))), 
                            reverse=True)
            retrieved_lists.append(reranked)
            
            # hit rate: at least 1 relevant in top-10
            gt = gt_map[i]  # Usar índice en lugar de queries.index(q)
            hit = any(pid in gt for pid in reranked[:10])
            hits.append(1.0 if hit else 0.0)
            
        except Exception as e:
            logging.error(f"Error en evaluación colaborativa para query {i}: {e}")
            hits.append(0.0)
            retrieved_lists.append([])
    
    elapsed = time.perf_counter() - t0
    return {
        "time_seconds": elapsed,
        "avg_collab_time": mean(collab_scores_times) if collab_scores_times else 0,
        "hit_rate_top10": mean(hits) if hits else 0,
        "mrr": mrr_at_k(retrieved_lists, gt_map, k=10) if retrieved_lists else 0,
        "ndcg": ndcg_at_k(retrieved_lists, gt_map, k=10) if retrieved_lists else 0,
    }

def eval_rlhf(trainer, monitor, feedback_logs_path, retrain_trigger_count=50):
    # load logs (success + failed + feedback files)
    success = load_jsonl(PATH_SUCCESS_LOG)
    failed = load_jsonl(PATH_FAILED_LOG)
    feedback_all = []
    for p in glob.glob("data/feedback/feedback_*.jsonl"):
        feedback_all.extend(load_jsonl(p))
    all_logs = success + failed + feedback_all
    t0 = time.perf_counter()
    failed_log = Path("data/feedback/failed.jsonl")
    success_log = Path("data/feedback/success.jsonl")

    dataset = trainer.prepare_rlhf_dataset_from_logs(failed_log, success_log)
    prep_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    train_res = trainer.train(dataset)
    train_time = time.perf_counter() - t1
    # monitor statistics
    try:
        stats = monitor.get_training_stats()
    except Exception:
        stats = {}
    result = {
        "n_logs": len(all_logs),
        "prepare_time": prep_time,
        "train_time": train_time,
        "train_result": train_res,
        "monitor_stats": stats
    }
    # append to metrics file
    append_jsonl({
        "timestamp": time.time(),
        "summary": result
    }, PATH_RLHF_METRICS)
    return result

def eval_hybrid(agent, retriever, collab, trainer, monitor, queries, users, gt_map):
    """
    Eval integration: rerank_with_rlhf(), retriever updates feedback weights, decay, cache times, etc.
    We'll measure end-to-end pipeline time and integration metrics.
    """
    t0 = time.perf_counter()
    overall_hits = []
    pipeline_times = []
    for q, user in zip(queries, users):
        t1 = time.perf_counter()
        docs = retriever.retrieve(q, top_k=20)
        doc_ids = [d[0] if isinstance(d, (list,tuple)) else d for d in docs]
        # simulate update feedback weights immediate (if exists)
        if hasattr(retriever, "update_feedback_weights_immediately"):
            t_up0 = time.perf_counter()
            retriever.update_feedback_weights_immediately(user.get("id", None), q, doc_ids)
            t_up1 = time.perf_counter()
            update_time = t_up1 - t_up0
        else:
            update_time = 0.0
        # collaborative scores
        t_coll0 = time.perf_counter()
        collab_scores = collab.get_collaborative_scores(user.get("id", None), doc_ids)
        t_coll1 = time.perf_counter()
        collab_time = t_coll1 - t_coll0
        # rerank with rl hf integration if agent has method
        if hasattr(agent, "_rerank_with_rlhf"):
            reranked = agent._rerank_with_rlhf(doc_ids, collab_scores)
            if not reranked:
                # fallback weighted merge
                reranked = sorted(doc_ids, key=lambda pid: (collab_scores.get(pid,0)*0.6 + 0.4*(1/(1+doc_ids.index(pid)))), reverse=True)
        else:
            reranked = sorted(doc_ids, key=lambda pid: collab_scores.get(pid,0), reverse=True)
        t2 = time.perf_counter()
        pipeline_times.append(t2 - t1)
        gt = gt_map[queries.index(q)]
        hit = any(pid in gt for pid in reranked[:10])
        overall_hits.append(1.0 if hit else 0.0)
    elapsed = time.perf_counter() - t0
    return {
        "total_time": elapsed,
        "avg_pipeline_time": mean(pipeline_times),
        "hit_rate_top10": mean(overall_hits),
    }

# --- Utility: build queries + ground truth from logs simple heuristics ---
def build_queries_and_gts(n=50):  # Reducir a 50 para pruebas
    """
    Build queries and ground truth from logs - VERSIÓN CORREGIDA
    """
    queries = []
    gt_sets = []
    
    # Primero: usar datos de prueba recién creados
    test_queries = [
        "juegos de acción para ps5",
        "auriculares gaming inalámbricos", 
        "teclado mecánico gamer",
        "monitor gaming 144hz",
        "silla gamer ergonómica",
        "nintendo switch oled",
        "xbox series x",
        "juegos multiplayer pc",
        "ratón gaming inalámbrico",
        "ssd nvme 1tb"
    ]
    
    # Agregar queries de prueba con ground truth simple
    for q in test_queries[:n]:
        queries.append(q)
        # Ground truth simulado basado en términos de la query
        gt_products = set()
        if "ps5" in q.lower():
            gt_products.add("B09V3HN1KC")  # God of War
        if "auriculares" in q.lower():
            gt_products.add("B0BDJHN2GS")  # HyperX
        if "teclado" in q.lower():
            gt_products.add("B0C5N4VYF2")  # Logitech
        if not gt_products:
            # Fallback: agregar algunos productos gaming comunes
            gt_products.add("B09V3HN1KC")
            gt_products.add("B0BDJHN2GS")
        gt_sets.append(gt_products)
    
    # Segundo: intentar cargar de logs existentes
    success_data = load_jsonl("data/feedback/success_queries.log")
    for item in success_data:
        if len(queries) >= n:
            break
        if isinstance(item, dict) and 'query' in item:
            queries.append(item['query'])
            # Ground truth del item si existe
            gt = set()
            if 'selected_product_id' in item and item['selected_product_id']:
                gt.add(item['selected_product_id'])
            gt_sets.append(gt)
    
    logger.info(f"📊 Construidas {len(queries)} consultas con ground truth")
    return queries, gt_sets
# --- Main entrypoint ---
def main(args):
    # instantiate components
    retriever_inst = Retriever() if callable(Retriever) else Retriever
    agent_inst = WorkingAgent() if callable(WorkingAgent) else WorkingAgent
    trainer_inst = RLHFTrainer() if callable(RLHFTrainer) else RLHFTrainer
    monitor_inst = RLHFMonitor() if callable(RLHFMonitor) else RLHFMonitor

    # load resources
    users = load_users()
    if not users:
        # create synthetic users if none present
        users = [{"id": f"user_{i}"} for i in range(50)]
    products = load_products()
    if not products:
        products = [{"id": f"prod_stub_{i}", "title": f"Producto {i}"} for i in range(100)]
    # Crear user_manager real si existe, sino usar stub
    user_manager_inst = None
    if UserManager is not None and callable(UserManager):
        try:
            user_manager_inst = UserManager()
        except Exception as e:
            logging.warning("No se pudo instanciar UserManager real: %s — usar StubUserManager", e)
            user_manager_inst = StubUserManager()
    else:
        user_manager_inst = StubUserManager()

    # ahora instanciamos CollaborativeFilter pasándole user_manager_inst
    try:
        collab_inst = CollaborativeFilter(user_manager_inst) if callable(CollaborativeFilter) else CollaborativeFilter
    except TypeError as e:
        # En caso de que CollaborativeFilter tenga otra firma, caemos a un stub seguro
        logging.warning("CollaborativeFilter.__init__ requires diferentes args (%s). Usando StubCollaborative.", e)
        collab_inst = StubCollaborative()
    queries, gt_sets = build_queries_and_gts(n=len(users))
    # align sizes
    n = min(len(queries), len(users), 50)
    queries = queries[:n]
    gt_sets = gt_sets[:n]
    users = users[:n]

    results = {}
    mode = args.mode.lower()
    if mode in ("basic", "all"):
        logging.info("Evaluando: RAG BÁSICO")
        res_basic = eval_basic(agent_inst, retriever_inst, queries, gt_sets, top_k=10)
        results["basic"] = res_basic
        logging.info("Basic result: %s", res_basic)

    if mode in ("collab", "all"):
        logging.info("Evaluando: RAG + COLABORATIVO")
        res_collab = eval_collaborative(agent_inst, retriever_inst, collab_inst, users, queries, gt_sets, top_k=20)
        results["collaborative"] = res_collab
        logging.info("Collab result: %s", res_collab)

    if mode in ("rlhf", "all"):
        logging.info("Evaluando: RAG + RLHF")
        res_rlhf = eval_rlhf(trainer_inst, monitor_inst, "data/feedback")
        results["rlhf"] = res_rlhf
        logging.info("RLHF result: %s", res_rlhf)

    if mode in ("hybrid", "all"):
        logging.info("Evaluando: SISTEMA HÍBRIDO COMPLETO")
        res_hybrid = eval_hybrid(agent_inst, retriever_inst, collab_inst, trainer_inst, monitor_inst, queries, users, gt_sets)
        results["hybrid"] = res_hybrid
        logging.info("Hybrid result: %s", res_hybrid)

    # save summary
    out_path = f"data/feedback/eval_summary_{mode}.json"
    save_json({"timestamp": time.time(), "mode": mode, "results": results, "missing": MISSING_IMPLEMENTATIONS}, out_path)
    logging.info("Resumen guardado en %s", out_path)
    # append to rlhf metrics if RLHF evaluated
    if "rlhf" in results:
        append_jsonl({"timestamp": time.time(), "rlhf_summary": results["rlhf"]}, PATH_RLHF_METRICS)
    # print missing implementations to console
    if MISSING_IMPLEMENTATIONS:
        logging.warning("FALTAN implementaciones / imports. Para evaluación completa proporciona el código de las siguientes rutas y funciones (ver lista):")
        for mod, attr, err in MISSING_IMPLEMENTATIONS:
            logging.warning(" - %s (attr=%s) error=%s", mod, attr, err)
    print("FIN. Resultados clave:", json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepEval: evaluación de RAG/RLHF/híbrido")
    parser.add_argument("--mode", type=str, default="all", help="Modo: basic | collab | rlhf | hybrid | all")
    parsed = parser.parse_args()
    main(parsed)
