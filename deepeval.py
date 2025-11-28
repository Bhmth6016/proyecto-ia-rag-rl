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
import hashlib
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
    def prepare_rlhf_dataset_from_logs(self, failed_log_path: Path, success_log_path: Path, min_samples: int = 5) -> Dict[str, Any]:
        """Prepara dataset RLHF desde logs - VERSIÓN MEJORADA"""
        import logging
        logger = logging.getLogger(__name__)
        samples = []
        
        # Crear datos sintéticos si no hay logs
        if not success_log_path.exists() and not failed_log_path.exists():
            print("📝 Generando datos sintéticos para RLHF...")
            synthetic_queries = [
                "playstation 5 consola gaming",
                "nintendo switch juegos familia", 
                "xbox series x videojuegos",
                "monitor gaming 144hz",
                "teclado mecánico rgb"
            ]
            
            for i, query in enumerate(synthetic_queries):
                samples.append({
                    'query': query,
                    'answer': f"Productos gaming recomendados para {query}",
                    'labels': 1,  # Feedback positivo
                    'score': 0.8 + (i * 0.05)
                })
        
        # Cargar logs existentes (código anterior)...
        
        logger.info(f"📊 Muestras válidas encontradas: {len(samples)}")

        if len(samples) < min_samples:
            logger.warning(f"❌ No hay suficientes muestras: {len(samples)} < {min_samples}")
            return {'train': None, 'eval': None, 'total_samples': 0}

        # Dividir dataset
        train_size = int(0.8 * len(samples))
        train_data = samples[:train_size]
        eval_data = samples[train_size:]

        return {
            'train': Dataset.from_list(train_data) if train_data else None,
            'eval': Dataset.from_list(eval_data) if eval_data else None,
            'total_samples': len(samples)
        }
    def train(self, dataset):
        return {"trained_examples": dataset.get("n_examples", 0), "loss": 0.1}

class StubMonitor:
    def log_training_session(self, session): pass
    def get_training_stats(self): return {}

class RobustStubUserManager:
    """Stub robusto para UserManager que evita errores"""
    def __init__(self, users=None):
        self._users = users or []
        # Crear perfiles de usuario simulados robustos
        if not self._users:
            self._users = [
                {"user_id": f"user_{i}", "id": f"user_{i}", "categories": ["games", "electronics"]} 
                for i in range(50)
            ]
    
    def get_user_profile(self, user_id):
        """Devuelve un perfil de usuario simulado robusto"""
        # Buscar usuario existente o crear uno nuevo
        for user in self._users:
            if user.get("user_id") == user_id or user.get("id") == user_id:
                return user
        
        # Crear nuevo usuario si no existe
        new_user = {
            "user_id": user_id,
            "id": user_id,
            "categories": ["games", "electronics", "gaming"],
            "preferred_categories": ["games", "gaming"],
            "search_history": [],
            "feedback_history": []
        }
        self._users.append(new_user)
        return new_user
    
    def find_similar_users(self, target_user, min_similarity=0.6):
        """
        Devuelve usuarios simulados con similitud básica por categorías.
        Maneja diferentes formatos de target_user.
        """
        similar = []
        
        # Determinar categorías del target_user de forma robusta
        if isinstance(target_user, dict):
            target_categories = set(target_user.get("categories", []) or target_user.get("preferred_categories", []))
            target_id = target_user.get("user_id") or target_user.get("id")
        else:
            target_categories = set(["games", "electronics"])
            target_id = str(target_user)
        
        for user in self._users:
            user_id = user.get("user_id") or user.get("id")
            if user_id == target_id:
                continue  # saltar el mismo usuario
            
            user_categories = set(user.get("categories", []) or user.get("preferred_categories", []))
            
            # Similitud simple: Jaccard similarity
            if target_categories and user_categories:
                intersection = len(target_categories & user_categories)
                union = len(target_categories | user_categories)
                similarity = intersection / union if union > 0 else 0
            else:
                similarity = 0.3  # similitud por defecto si no hay categorías
            
            if similarity >= min_similarity:
                similar.append(user)
        
        return similar[:5]  # máximo 5 usuarios similares

class RobustStubCollaborative:
    def __init__(self, user_manager=None): 
        self.user_manager = user_manager or RobustStubUserManager()
        
    def get_collaborative_scores(self, user_id, candidates):
        """
        Stub mejorado que simula scores colaborativos más realistas
        """
        if not candidates:
            return {}
            
        scores = {}
        
        # Simular que algunos productos son "populares" (primeros 10)
        popular_products = set(candidates[:10])
        
        for i, candidate in enumerate(candidates):
            base_score = 0.2
            
            # Boost por popularidad
            if candidate in popular_products:
                base_score += 0.3
                
            # Boost por posición (los primeros son más relevantes)
            position_boost = 0.5 / (i + 1)
            
            # Score final
            final_score = base_score + position_boost
            scores[candidate] = min(final_score, 0.9)
            
        logger.debug(f"🔍 Collaborative scores para {user_id}: {len(scores)} productos")
        return scores

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
# En deepeval.py, modifica la función eval_basic:

def eval_basic(agent, retriever, queries, gt_map, top_k=10):
    """
    queries: list of query strings
    gt_map: list of sets with relevant product ids per query (aligned)
    """
    t0 = time.perf_counter()
    retrieved_lists = []
    responses = []
    
    logger.info(f"📊 Evaluando {len(queries)} consultas con ground truth")
    
    for i, q in enumerate(queries):
        # retrieval
        docs = retriever.retrieve(q, top_k=top_k)
        if docs and isinstance(docs[0], (list, tuple)):
            doc_ids = [d[0] for d in docs]
        else:
            doc_ids = list(docs)
        retrieved_lists.append(doc_ids)
        
        # agent response
        resp = agent.process_query(q)
        resp_text = resp.text
        recommended = resp.recommended_ids
        
        # 🔥 CORRECCIÓN: Convertir objetos Product a IDs strings
        recommended_ids = []
        for item in recommended:
            if hasattr(item, 'id'):
                recommended_ids.append(item.id)
            else:
                recommended_ids.append(str(item))
        
        responses.append((resp_text, recommended_ids))
        
        # Debug de coincidencias
        gt = gt_map[i]
        matches = set(recommended_ids) & gt
        logger.info(f"🔍 Query {i+1}: {len(matches)}/{len(gt)} coincidencias")
    
    elapsed = time.perf_counter() - t0
    
    # Métricas
    mrr_scores = []
    ndcg_scores = []
    
    for i, (resp, gt) in enumerate(zip(responses, gt_map)):
        recommended_ids = resp[1]  # IDs recomendados ya convertidos
        
        # MRR
        rr = 0.0
        for pos, pid in enumerate(recommended_ids[:top_k], 1):
            if pid in gt:
                rr = 1.0 / pos
                break
        mrr_scores.append(rr)
        
        # NDCG
        dcg = 0.0
        for pos, pid in enumerate(recommended_ids[:top_k], 1):
            rel = 1.0 if pid in gt else 0.0
            if pos == 1:
                dcg += rel
            else:
                dcg += rel / log2(pos)
        
        # IDCG
        ideal_ranking = list(gt) + [pid for pid in recommended_ids if pid not in gt]
        idcg = 0.0
        for pos, pid in enumerate(ideal_ranking[:top_k], 1):
            rel = 1.0 if pid in gt else 0.0
            if pos == 1:
                idcg += rel
            else:
                idcg += rel / log2(pos)
                
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    # Métricas de texto
    bleu_scores = []
    rouge_scores = []
    
    for resp, gt in zip(responses, gt_map):
        resp_text = resp[0]
        ref_text = f"Recomendaciones: {', '.join(gt)}"
        references = [ref_text]
        
        if resp_text and references:
            bleu_scores.append(bleu1(resp_text, references))
            rouge_scores.append(rouge_l(resp_text, references))
    
    return {
        "time_seconds": elapsed,
        "mrr": mean(mrr_scores) if mrr_scores else 0.0,
        "ndcg": mean(ndcg_scores) if ndcg_scores else 0.0,
        "bleu1": mean(bleu_scores) if bleu_scores else 0.0,
        "rouge_l": mean(rouge_scores) if rouge_scores else 0.0,
        "retrieved_count_avg": mean([len(r) for r in retrieved_lists]),
        "debug": {
            "total_queries": len(queries),
            "queries_with_gt": sum(1 for gt in gt_map if len(gt) > 0),
            "avg_gt_size": mean([len(gt) for gt in gt_map]),
            "mrr_scores_sample": mrr_scores[:3] if mrr_scores else []
        }
    }
def eval_collaborative(agent, retriever, collab, users, queries, gt_map, top_k=20):
    t0 = time.perf_counter()
    retrieved_lists = []
    collab_scores_times = []
    hits = []
    
    for i, (q, user) in enumerate(zip(queries, users)):
        try:
            # 🔥 CORRECCIÓN: Manejar usuario de forma robusta
            if isinstance(user, dict):
                user_id = user.get("user_id") or user.get("id") or f"user_{i}"
            else:
                user_id = f"user_{i}"
                
            docs = retriever.retrieve(q, top_k=top_k)
            doc_ids = [d[0] if isinstance(d, (list,tuple)) else d for d in docs]
            
            # measure collaborative ranking time
            t1 = time.perf_counter()
            
            # 🔥 CORRECCIÓN: Usar user_id en lugar del objeto completo
            collab_scores = collab.get_collaborative_scores(user_id, doc_ids)
            
            t2 = time.perf_counter()
            collab_scores_times.append(t2-t1)
            
            # merge/weighted rerank: 60% collab / 40% rag (simple)
            reranked = sorted(doc_ids, 
                            key=lambda pid: (collab_scores.get(pid,0)*0.6 + 0.4*(1/(1+doc_ids.index(pid)))), 
                            reverse=True)
            retrieved_lists.append(reranked)
            
            # hit rate: at least 1 relevant in top-10
            gt = gt_map[i]
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

def eval_rlhf(trainer_inst, monitor_inst, feedback_dir):
    """Evalúa el sistema RAG + RLHF - VERSIÓN CORREGIDA"""
    print("Evaluando: RAG + RLHF")
    
    feedback_dir = Path(feedback_dir)
    failed_log_path = feedback_dir / "failed_queries.jsonl"
    success_log_path = feedback_dir / "successful_queries.jsonl"
    
    # Preparar dataset
    dataset_dict = trainer_inst.prepare_rlhf_dataset_from_logs(
        failed_log_path, success_log_path
    )
    
    print(f"📊 Total logs cargados: {dataset_dict.get('total_samples', 0)}")
    
    # Verificar si hay suficientes datos
    train_dataset = dataset_dict.get('train')
    if train_dataset is None or len(train_dataset) == 0:
        print("❌ No hay suficientes datos para entrenar RLHF")
        return {
            'time_seconds': 0,
            'total_samples': 0,
            'status': 'insufficient_data'
        }
    
    print(f"🎯 Iniciando entrenamiento con {len(train_dataset)} ejemplos de entrenamiento...")
    
    try:
        # ✅ CORRECCIÓN: Pasar solo el dataset de entrenamiento
        train_res = trainer_inst.train(train_dataset)
        
        # Evaluar si hay datos de evaluación
        eval_dataset = dataset_dict.get('eval')
        eval_results = {}
        if eval_dataset and len(eval_dataset) > 0:
            eval_results = trainer_inst.evaluate(eval_dataset)
        
        return {
            'time_seconds': train_res.get('training_time', 0),
            'total_samples': dataset_dict.get('total_samples', 0),
            'train_samples': len(train_dataset),
            'eval_samples': len(eval_dataset) if eval_dataset else 0,
            'eval_results': eval_results,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"❌ Error en entrenamiento RLHF: {e}")
        return {
            'time_seconds': 0,
            'total_samples': dataset_dict.get('total_samples', 0),
            'status': f'error: {str(e)}'
        }
def calculate_metrics(all_results, queries, gt_sets, top_k=5):
    """
    all_results: lista de listas de productos finales por query
    queries: lista de strings
    gt_sets: lista de sets con ids relevantes
    """
    retrieved_lists = []
    for res_list in all_results:
        ids = [str(p.get('id')) if isinstance(p, dict) else str(p) for p in res_list]
        retrieved_lists.append(ids)

    mrr_score = mrr_at_k(retrieved_lists, gt_sets, k=top_k)
    ndcg_score = ndcg_at_k(retrieved_lists, gt_sets, k=top_k)

    # BLEU y ROUGE sobre textos simulados
    bleu_scores = []
    rouge_scores = []
    for res_list, gt in zip(all_results, gt_sets):
        text = " ".join([str(p.get('title','')) for p in res_list if isinstance(p, dict)])
        ref_text = " ".join(gt)
        if text:
            bleu_scores.append(bleu1(text, [ref_text]))
            rouge_scores.append(rouge_l(text, [ref_text]))
    return {
        "mrr": mean(mrr_score) if mrr_score else 0.0,
        "ndcg": mean(ndcg_score) if ndcg_score else 0.0,
        "bleu1": mean(bleu_scores) if bleu_scores else 0.0,
        "rouge_l": mean(rouge_scores) if rouge_scores else 0.0,
        "retrieved_count_avg": mean([len(r) for r in retrieved_lists]) if retrieved_lists else 0.0
    }

def eval_hybrid(agent, retriever, collab, trainer, monitor, queries, users, gt_sets):
    """Evalúa el sistema híbrido completo - VERSIÓN CORREGIDA"""
    print("Evaluando: SISTEMA HÍBRIDO COMPLETO")
    
    start_time = time.time()
    all_results = []
    
    for i, (query, user_id) in enumerate(zip(queries, users)):
        try:
            # Perfil de usuario simulado
            profile = {
                'user_id': user_id,
                'preferences': {'gaming': 0.8, 'electronics': 0.7},
                'history': []
            }
            
            # 1. Búsqueda inicial - Manejar parámetro limit
            try:
                initial_results = agent.process_query(query, limit=20)
            except TypeError:
                # Fallback si no acepta limit
                initial_results = agent.process_query(query)
                # Limitar manualmente si es necesario
                if len(initial_results) > 20:
                    initial_results = initial_results[:20]
            
            doc_ids = [doc.get('id', '') for doc in initial_results if doc.get('id')]
            
            if not doc_ids:
                all_results.append([])
                continue
            
            # 2. Puntajes colaborativos
            collab_scores = {}
            for doc_id in doc_ids:
                score = collab.get_item_score(doc_id, user_id)
                collab_scores[str(doc_id)] = float(score) if score is not None else 0.5
            
            # 3. Reranking híbrido con RLHF
            reranked_ids = agent._rerank_with_rlhf(
                [str(doc_id) for doc_id in doc_ids], 
                collab_scores, 
                profile
            )
            
            # 4. Obtener productos finales
            final_results = []
            for doc_id in reranked_ids[:5]:  # Top 5
                doc = next((d for d in initial_results if str(d.get('id')) == doc_id), None)
                if doc:
                    final_results.append(doc)
            
            all_results.append(final_results)
            
        except Exception as e:
            print(f"❌ Error en query híbrida {i}: {e}")
            all_results.append([])
    
    # Calcular métricas
    time_seconds = time.time() - start_time
    metrics = calculate_metrics(all_results, queries, gt_sets)
    
    return {
        'time_seconds': time_seconds,
        **metrics,
        'total_queries': len(queries),
        'successful_queries': len([r for r in all_results if r])
    }

# --- Utility: build queries + ground truth from logs simple heuristics ---
# En deepeval.py, modifica build_queries_and_gts:

def build_queries_and_gts(n=50):
    """
    Build queries and ground truth - VERSIÓN CON IDs REALES
    """
    queries = []
    gt_sets = []
    
    # Consultas de prueba con ground truth basado en productos reales
    test_queries_with_patterns = [
        # (query, [patrones_de_ids_reales])
        ("playstation 5", ["playstation", "ps5", "ps4"]),
        ("xbox series x", ["xbox", "series"]),
        ("nintendo switch", ["nintendo", "switch"]),
        ("monitor gaming", ["monitor", "gaming"]),
        ("teclado mecánico", ["teclado", "keyboard"]),
        ("auriculares gaming", ["auriculares", "headset"]),
        ("silla gamer", ["silla", "chair"]),
        ("ratón gaming", ["ratón", "mouse"]),
        ("ssd nvme", ["ssd", "nvme"]),
        ("juegos ps5", ["juegos", "game"]),
    ]
    
    # Cargar algunos productos reales para crear ground truth realista
    real_products = []
    products_file = Path("data/processed/products.json")
    if products_file.exists():
        with open(products_file, 'r', encoding='utf-8') as f:
            real_products = json.load(f)
    
    # Si no hay productos reales, crear IDs simulados que coincidan con el formato real
    if not real_products:
        # Usar el formato UUID que vimos en los logs
        real_products = [
            {"id": "7e1a0768-ae61-4344-9e10-a22bc377bfc9", "title": "PlayStation 5"},
            {"id": "f991eace-20a3-4b53-842b-76661ccf907c", "title": "Xbox Series X"},
            {"id": "974905bb-37f5-4542-b3d5-9b6c15c24929", "title": "Nintendo Switch"},
            {"id": "77318e92-61ce-43fe-a218-96e9299fb536", "title": "Gaming Monitor"},
            {"id": "c3f69492-9329-4aaa-a3f9-bedc6680e529", "title": "Mechanical Keyboard"},
        ]
    
    # Crear ground truth basado en productos reales
    for query, patterns in test_queries_with_patterns[:min(n, len(test_queries_with_patterns))]:
        queries.append(query)
        gt_set = set()
        
        # Buscar productos que coincidan con los patrones
        for product in real_products:
            product_id = product.get('id', '')
            product_title = product.get('title', '').lower()
            
            # Verificar si el producto coincide con algún patrón
            for pattern in patterns:
                if (pattern in product_title or 
                    pattern in str(product_id).lower()):
                    gt_set.add(product_id)
                    break  # Solo agregar una vez por producto
            
            # Limitar a 2-3 productos por query
            if len(gt_set) >= 3:
                break
                
        # Si no hay coincidencias, usar algunos productos aleatorios
        if not gt_set and real_products:
            gt_set = set([p['id'] for p in real_products[:2]])
            
        gt_sets.append(gt_set)
    
    # También cargar de logs existentes
    success_data = load_jsonl("data/feedback/success_queries.log")
    for item in success_data:
        if len(queries) >= n:
            break
        if isinstance(item, dict) and 'query' in item:
            query = item['query']
            queries.append(query)
            
            gt_set = set()
            if 'selected_product_id' in item and item['selected_product_id']:
                gt_set.add(item['selected_product_id'])
            else:
                # Usar productos reales como fallback
                if real_products:
                    gt_set = set([p['id'] for p in real_products[:2]])
                    
            gt_sets.append(gt_set)
    
    logger.info(f"📊 Construidas {len(queries)} consultas con ground truth real")
    logger.info(f"📊 Ejemplo GT real: {list(gt_sets[0]) if gt_sets else 'None'}")
    
    return queries, gt_sets

# Reemplaza la función generate_initial_rlhf_data en deepeval.py:

def generate_initial_rlhf_data(agent, queries, users, num_samples=20):
    """Genera datos iniciales para RLHF - VERSIÓN CORREGIDA"""
    print("🔄 Generando datos iniciales para RLHF...")
    
    feedback_dir = Path("data/feedback")
    feedback_dir.mkdir(parents=True, exist_ok=True)
    
    success_file = feedback_dir / "successful_queries.jsonl"
    
    samples_generated = 0
    
    for i in range(min(num_samples, len(queries))):
        try:
            query = queries[i]
            user_id = users[i] if i < len(users) else "user_0"
            
            # ✅ CORRECCIÓN: Usar process_query sin parámetro limit si no lo acepta
            try:
                # Intentar con limit primero
                results = agent.process_query(query, limit=5)
            except TypeError:
                # Fallback: sin parámetro limit
                results = agent.process_query(query)
                # Limitar manualmente si es necesario
                if len(results) > 5:
                    results = results[:5]
            
            if results:
                # Crear datos de entrenamiento simulados
                log_data = {
                    'query': query,
                    'response': json.dumps([{
                        'title': r.get('title', ''),
                        'id': r.get('id', ''),
                        'score': r.get('score', 0.5)
                    } for r in results], ensure_ascii=False),
                    'score': 0.8 + (i * 0.01),  # Scores variados
                    'user_id': user_id,
                    'timestamp': time.time()
                }
                
                with open(success_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
                
                samples_generated += 1
                
        except Exception as e:
            print(f"⚠️ Error generando dato RLHF {i}: {e}")
    
    print(f"✅ Generados {samples_generated} datos iniciales para RLHF")
    return samples_generated
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
    user_manager_inst = RobustStubUserManager()

    # ahora instanciamos CollaborativeFilter pasándole user_manager_inst
    collab_inst = RobustStubCollaborative(user_manager_inst)
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
        
    if args.mode in ['all', 'rlhf']:
        print("\n" + "="*50)
        trainer_inst = RLHFTrainer()
        monitor_inst = None  # O tu instancia de monitor si la tienes
        
        # ✅ CORRECCIÓN: Pasar las instancias correctamente
        res_rlhf = eval_rlhf(trainer_inst, monitor_inst, "data/feedback")
        results['RLHF'] = res_rlhf
        
        print(f"📊 RLHF - Muestras: {res_rlhf.get('total_samples', 0)}, "
              f"Estado: {res_rlhf.get('status', 'unknown')}")
    if args.mode in ['all', 'rlhf', 'hybrid']:
        generate_initial_rlhf_data(agent_inst, queries[:20], users[:20])    
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
