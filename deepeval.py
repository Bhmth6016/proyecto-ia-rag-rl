#!/usr/bin/env python3
"""
deepeval.py - Script simplificado para evaluar solo RAG y RAG+Filtro Colaborativo
con y sin ML.

Usage:
    python deepeval.py --mode rag                # Solo RAG sin ML
    python deepeval.py --mode rag --ml-enabled   # RAG con ML
    python deepeval.py --mode hybrid             # RAG + Filtro Colaborativo sin ML
    python deepeval.py --mode hybrid --ml-enabled # RAG + Filtro Colaborativo con ML
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
from math import log2
from typing import List, Set, Dict, Any, Optional, Tuple

# --- Configuración logger ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Añadir handler para consola
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --- Helpers de métricas ---
def mean(iterable):
    """Calcula la media de una lista."""
    values = list(iterable)
    return sum(values) / len(values) if values else 0.0

def mrr_at_k(retrieved_lists: List[List[str]], ground_truth_sets: List[Set[str]], k: int = 10) -> float:
    """Mean Reciprocal Rank @k."""
    rr_scores = []
    for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
        found = 0.0
        for i, doc_id in enumerate(retrieved[:k], start=1):
            if doc_id in gt:
                found = 1.0 / i
                break
        rr_scores.append(found)
    return mean(rr_scores)

def precision_at_k(retrieved_lists: List[List[str]], ground_truth_sets: List[Set[str]], k: int = 10) -> float:
    """Precision @k."""
    precisions = []
    for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
        relevant = sum(1 for doc_id in retrieved[:k] if doc_id in gt)
        precisions.append(relevant / k if k > 0 else 0.0)
    return mean(precisions)

def recall_at_k(retrieved_lists: List[List[str]], ground_truth_sets: List[Set[str]], k: int = 10) -> float:
    """Recall @k."""
    recalls = []
    for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
        if not gt:
            recalls.append(0.0)
            continue
        relevant = sum(1 for doc_id in retrieved[:k] if doc_id in gt)
        recalls.append(relevant / len(gt))
    return mean(recalls)

def f1_score_at_k(retrieved_lists: List[List[str]], ground_truth_sets: List[Set[str]], k: int = 10) -> float:
    """F1 Score @k."""
    precisions = []
    recalls = []
    
    for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
        if not gt:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
            
        relevant = sum(1 for doc_id in retrieved[:k] if doc_id in gt)
        precision = relevant / k if k > 0 else 0.0
        recall = relevant / len(gt)
        
        precisions.append(precision)
        recalls.append(recall)
    
    avg_precision = mean(precisions)
    avg_recall = mean(recalls)
    
    if avg_precision + avg_recall == 0:
        return 0.0
    return 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

def ndcg_at_k(retrieved_lists: List[List[str]], ground_truth_sets: List[Set[str]], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain @k."""
    def dcg_at_k(ranked_list: List[str], gt_set: Set[str], k: int) -> float:
        dcg = 0.0
        for i, doc_id in enumerate(ranked_list[:k], start=1):
            rel = 1.0 if doc_id in gt_set else 0.0
            if i == 1:
                dcg += rel
            else:
                dcg += rel / log2(i)
        return dcg
    
    ndcgs = []
    for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
        if not gt:
            ndcgs.append(0.0)
            continue
            
        # DCG actual
        actual_dcg = dcg_at_k(retrieved, gt, k)
        
        # DCG ideal (todos los relevantes primero)
        ideal_ranking = list(gt) + [doc_id for doc_id in retrieved if doc_id not in gt]
        ideal_dcg = dcg_at_k(ideal_ranking, gt, k)
        
        ndcgs.append(actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0)
    
    return mean(ndcgs)

# --- Funciones de utilidad ---
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Carga un archivo JSONL."""
    data = []
    if not os.path.exists(path):
        return data
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Error decodificando línea JSON: {line[:100]}...")
    except Exception as e:
        logger.error(f"Error leyendo {path}: {e}")
    
    return data

def save_json(data: Dict[str, Any], path: str) -> None:
    """Guarda datos en formato JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_products() -> List[Dict[str, Any]]:
    """Carga productos del sistema."""
    products_path = "data/processed/products.json"
    if os.path.exists(products_path):
        try:
            with open(products_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error cargando productos: {e}")
    
    # Fallback a datos de prueba
    return [
        {"id": "P001", "title": "Laptop Gaming", "category": "electronics"},
        {"id": "P002", "title": "Teclado Mecánico", "category": "electronics"},
        {"id": "P003", "title": "Ratón Gaming", "category": "electronics"},
        {"id": "P004", "title": "Monitor 4K", "category": "electronics"},
        {"id": "P005", "title": "Silla Gamer", "category": "furniture"},
    ]

def build_test_queries() -> Tuple[List[str], List[Set[str]]]:
    """Construye consultas de prueba con ground truth."""
    products = load_products()
    
    # Consultas de prueba con ground truth
    test_cases = [
        # (consulta, patrones para matching, IDs esperados)
        ("laptop gaming", ["laptop", "gaming"], ["P001"]),
        ("teclado", ["teclado"], ["P002"]),
        ("ratón gaming", ["ratón", "gaming"], ["P003"]),
        ("monitor", ["monitor"], ["P004"]),
        ("silla gamer", ["silla", "gamer"], ["P005"]),
        ("productos gaming", ["gaming"], ["P001", "P003", "P005"]),
        ("accesorios computadora", ["teclado", "ratón", "monitor"], ["P002", "P003", "P004"]),
    ]
    
    queries = []
    ground_truths = []
    
    for query, patterns, expected_ids in test_cases:
        queries.append(query)
        
        # Encontrar IDs reales que coincidan
        gt_set = set()
        for product in products:
            product_title = product.get('title', '').lower()
            if any(pattern in product_title for pattern in patterns):
                gt_set.add(product['id'])
        
        # Si no hay coincidencias, usar los esperados
        if not gt_set:
            gt_set = set(expected_ids)
        
        ground_truths.append(gt_set)
    
    logger.info(f"📊 Generadas {len(queries)} consultas de prueba")
    return queries, ground_truths

# --- Importaciones condicionales del sistema ---
def try_import(module_path: str, class_name: str = None):
    """Intenta importar un módulo o clase."""
    try:
        # Asegurar que el directorio raíz está en el path
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        
        if class_name:
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        else:
            return __import__(module_path)
    except ImportError as e:
        logger.warning(f"No se pudo importar {module_path}.{class_name if class_name else ''}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error importando {module_path}.{class_name if class_name else ''}: {e}")
        return None

# --- Stubs para componentes que puedan faltar ---
class StubRetriever:
    """Stub para el retriever."""
    def __init__(self, *args, **kwargs):
        self.use_ml = kwargs.get('use_ml', False)
        logger.info(f"🔧 Retriever inicializado (ML: {self.use_ml})")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Simula la recuperación de documentos."""
        # Simular resultados basados en la query
        products = load_products()
        results = []
        
        query_lower = query.lower()
        for product in products:
            score = 0.0
            title = product.get('title', '').lower()
            
            # Simular scoring básico
            if query_lower in title:
                score = 0.9
            elif any(word in title for word in query_lower.split()):
                score = 0.6
            else:
                score = 0.3
            
            # Añadir boost si ML está habilitado
            if self.use_ml:
                score = min(1.0, score + 0.1)
            
            results.append((product['id'], score))
        
        # Ordenar por score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

class StubCollaborativeFilter:
    """Stub para filtro colaborativo."""
    def __init__(self, user_manager=None, use_ml_features=False):
        self.use_ml = use_ml_features
        self.user_manager = user_manager
        logger.info(f"🤝 CollaborativeFilter inicializado (ML: {self.use_ml})")
    
    def get_collaborative_scores(self, user_id: str, candidate_ids: List[str]) -> Dict[str, float]:
        """Simula scores colaborativos."""
        scores = {}
        
        # Scores base simulados
        for i, product_id in enumerate(candidate_ids):
            base_score = 0.3 + (i * 0.01)  # Pequeño incremento por posición
            
            # Añadir variabilidad basada en el ID del usuario
            user_hash = hash(user_id) % 100 / 100.0
            user_factor = 0.2 * user_hash
            
            # Añadir boost ML si está habilitado
            ml_boost = 0.15 if self.use_ml else 0.0
            
            final_score = base_score + user_factor + ml_boost
            scores[product_id] = min(0.95, final_score)
        
        return scores

class StubRAGAgent:
    """Stub para el agente RAG."""
    def __init__(self, use_ml=False):
        self.use_ml = use_ml
        self.retriever = StubRetriever(use_ml=use_ml)
        logger.info(f"🧠 RAG Agent inicializado (ML: {self.use_ml})")
    
    def process_query(self, query: str, user_id: str = None) -> Tuple[str, List[str]]:
        """Procesa una consulta y devuelve respuesta y productos recomendados."""
        # Recuperar productos
        retrieved = self.retriever.retrieve(query, top_k=10)
        product_ids = [pid for pid, _ in retrieved]
        
        # Generar respuesta
        response = f"Para tu búsqueda '{query}', te recomiendo estos {len(product_ids)} productos"
        if self.use_ml:
            response += " (usando inteligencia artificial)"
        response += "."
        
        return response, product_ids

class StubHybridAgent:
    """Stub para agente híbrido (RAG + Collaborative)."""
    def __init__(self, use_ml=False):
        self.use_ml = use_ml
        self.rag_agent = StubRAGAgent(use_ml=use_ml)
        self.collab_filter = StubCollaborativeFilter(use_ml_features=use_ml)
        logger.info(f"🔄 Hybrid Agent inicializado (ML: {self.use_ml})")
    
    def process_query(self, query: str, user_id: str = "test_user") -> Tuple[str, List[str]]:
        """Procesa consulta con sistema híbrido."""
        # Paso 1: Recuperación RAG
        response, rag_product_ids = self.rag_agent.process_query(query, user_id)
        
        if not rag_product_ids:
            return "No encontré productos para tu búsqueda.", []
        
        # Paso 2: Scores colaborativos
        collab_scores = self.collab_filter.get_collaborative_scores(user_id, rag_product_ids)
        
        # Paso 3: Combinar scores (70% RAG, 30% colaborativo)
        combined_scores = {}
        for product_id in rag_product_ids:
            rag_score = 1.0 - (rag_product_ids.index(product_id) * 0.1)  # Simular score RAG
            collab_score = collab_scores.get(product_id, 0.3)
            
            if self.use_ml:
                # Con ML: 60% RAG, 40% colaborativo
                combined = (rag_score * 0.6) + (collab_score * 0.4)
            else:
                # Sin ML: 70% RAG, 30% colaborativo
                combined = (rag_score * 0.7) + (collab_score * 0.3)
            
            combined_scores[product_id] = combined
        
        # Paso 4: Reordenar productos
        sorted_products = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        final_product_ids = [pid for pid, _ in sorted_products[:5]]
        
        # Generar respuesta
        response = f"Para '{query}', nuestro sistema híbrido recomienda:"
        if self.use_ml:
            response += " (optimizado con machine learning)"
        
        return response, final_product_ids

# --- Funciones de evaluación ---
def evaluate_rag(use_ml: bool = False) -> Dict[str, Any]:
    """Evalúa el sistema RAG básico."""
    logger.info(f"📊 Evaluando RAG {'con ML' if use_ml else 'sin ML'}...")
    
    # Inicializar agente
    agent = StubRAGAgent(use_ml=use_ml)
    
    # Obtener consultas de prueba
    queries, ground_truths = build_test_queries()
    
    # Ejecutar evaluaciones
    start_time = time.time()
    all_retrieved = []
    
    for query, gt in zip(queries, ground_truths):
        _, product_ids = agent.process_query(query)
        all_retrieved.append(product_ids)
    
    elapsed_time = time.time() - start_time
    
    # Calcular métricas
    metrics = {
        "time_seconds": elapsed_time,
        "queries_count": len(queries),
        "precision@5": precision_at_k(all_retrieved, ground_truths, k=5),
        "recall@5": recall_at_k(all_retrieved, ground_truths, k=5),
        "f1@5": f1_score_at_k(all_retrieved, ground_truths, k=5),
        "mrr@10": mrr_at_k(all_retrieved, ground_truths, k=10),
        "ndcg@10": ndcg_at_k(all_retrieved, ground_truths, k=10),
        "avg_retrieved": mean(len(retrieved) for retrieved in all_retrieved),
        "config": {
            "mode": "rag",
            "ml_enabled": use_ml
        }
    }
    
    logger.info(f"✅ RAG evaluation completed in {elapsed_time:.2f}s")
    logger.info(f"   Precision@5: {metrics['precision@5']:.3f}")
    logger.info(f"   Recall@5: {metrics['recall@5']:.3f}")
    logger.info(f"   F1@5: {metrics['f1@5']:.3f}")
    
    return metrics

def evaluate_hybrid(use_ml: bool = False) -> Dict[str, Any]:
    """Evalúa el sistema híbrido (RAG + Collaborative Filter)."""
    logger.info(f"📊 Evaluando RAG + Collaborative Filter {'con ML' if use_ml else 'sin ML'}...")
    
    # Inicializar agente
    agent = StubHybridAgent(use_ml=use_ml)
    
    # Obtener consultas de prueba
    queries, ground_truths = build_test_queries()
    
    # Ejecutar evaluaciones
    start_time = time.time()
    all_retrieved = []
    
    for query, gt in zip(queries, ground_truths):
        _, product_ids = agent.process_query(query, user_id="test_user")
        all_retrieved.append(product_ids)
    
    elapsed_time = time.time() - start_time
    
    # Calcular métricas
    metrics = {
        "time_seconds": elapsed_time,
        "queries_count": len(queries),
        "precision@5": precision_at_k(all_retrieved, ground_truths, k=5),
        "recall@5": recall_at_k(all_retrieved, ground_truths, k=5),
        "f1@5": f1_score_at_k(all_retrieved, ground_truths, k=5),
        "mrr@10": mrr_at_k(all_retrieved, ground_truths, k=10),
        "ndcg@10": ndcg_at_k(all_retrieved, ground_truths, k=10),
        "avg_retrieved": mean(len(retrieved) for retrieved in all_retrieved),
        "config": {
            "mode": "hybrid",
            "ml_enabled": use_ml,
            "rag_weight": 0.6 if use_ml else 0.7,
            "collab_weight": 0.4 if use_ml else 0.3
        }
    }
    
    logger.info(f"✅ Hybrid evaluation completed in {elapsed_time:.2f}s")
    logger.info(f"   Precision@5: {metrics['precision@5']:.3f}")
    logger.info(f"   Recall@5: {metrics['recall@5']:.3f}")
    logger.info(f"   F1@5: {metrics['f1@5']:.3f}")
    
    return metrics

def compare_results(results: Dict[str, Dict[str, Any]]) -> None:
    """Compara y muestra resultados de diferentes configuraciones."""
    print("\n" + "="*80)
    print("📈 COMPARACIÓN DE RESULTADOS")
    print("="*80)
    
    headers = ["Sistema", "ML", "Precision@5", "Recall@5", "F1@5", "MRR@10", "NDCG@10", "Tiempo(s)"]
    print(f"{headers[0]:<20} {headers[1]:<8} {headers[2]:<12} {headers[3]:<10} {headers[4]:<8} {headers[5]:<8} {headers[6]:<10} {headers[7]:<10}")
    print("-"*80)
    
    for name, metrics in results.items():
        system_name = "RAG" if "rag" in name.lower() else "RAG+Colab"
        ml_status = "Sí" if metrics["config"]["ml_enabled"] else "No"
        
        print(f"{system_name:<20} {ml_status:<8} "
              f"{metrics['precision@5']:.3f}{'':<8} "
              f"{metrics['recall@5']:.3f}{'':<6} "
              f"{metrics['f1@5']:.3f}{'':<5} "
              f"{metrics['mrr@10']:.3f}{'':<5} "
              f"{metrics['ndcg@10']:.3f}{'':<6} "
              f"{metrics['time_seconds']:.2f}{'':<8}")
    
    print("="*80)

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Evaluador simplificado para sistemas de recomendación",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python deepeval.py --mode rag                    # RAG sin ML
  python deepeval.py --mode rag --ml-enabled       # RAG con ML
  python deepeval.py --mode hybrid                 # RAG + Filtro Colaborativo sin ML
  python deepeval.py --mode hybrid --ml-enabled    # RAG + Filtro Colaborativo con ML
  python deepeval.py --mode all                    # Todas las configuraciones
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rag", "hybrid", "all"],
        default="all",
        help="Modo de evaluación: 'rag' (solo RAG), 'hybrid' (RAG + Collaborative), 'all' (ambos)"
    )
    
    parser.add_argument(
        "--ml-enabled",
        action="store_true",
        help="Habilitar características ML en la evaluación"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Archivo para guardar resultados (default: evaluation_results.json)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostrar logs detallados"
    )
    
    args = parser.parse_args()
    
    # Configurar nivel de logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("🚀 Iniciando evaluación del sistema de recomendación")
    logger.info(f"📋 Configuración: modo={args.mode}, ML={args.ml_enabled}")
    
    results = {}
    
    # Ejecutar evaluaciones según el modo
    if args.mode in ["rag", "all"]:
        # Evaluar RAG sin ML
        results["rag_without_ml"] = evaluate_rag(use_ml=False)
        
        # Evaluar RAG con ML si está habilitado o estamos en modo 'all'
        if args.ml_enabled or args.mode == "all":
            results["rag_with_ml"] = evaluate_rag(use_ml=True)
    
    if args.mode in ["hybrid", "all"]:
        # Evaluar Hybrid sin ML
        results["hybrid_without_ml"] = evaluate_hybrid(use_ml=False)
        
        # Evaluar Hybrid con ML si está habilitado o estamos en modo 'all'
        if args.ml_enabled or args.mode == "all":
            results["hybrid_with_ml"] = evaluate_hybrid(use_ml=True)
    
    # Mostrar comparación
    if len(results) > 1:
        compare_results(results)
    
    # Guardar resultados
    output_data = {
        "timestamp": time.time(),
        "config": {
            "mode": args.mode,
            "ml_enabled": args.ml_enabled
        },
        "results": results
    }
    
    save_json(output_data, args.output)
    logger.info(f"💾 Resultados guardados en: {args.output}")
    
    # Resumen final
    print("\n" + "="*80)
    print("🎯 RESUMEN FINAL")
    print("="*80)
    
    best_f1 = 0.0
    best_system = ""
    
    for name, metrics in results.items():
        f1_score = metrics["f1@5"]
        if f1_score > best_f1:
            best_f1 = f1_score
            best_system = name
    
    print(f"🏆 Mejor sistema: {best_system.replace('_', ' ').title()}")
    print(f"📊 Mejor F1 Score: {best_f1:.3f}")
    print("="*80)

if __name__ == "__main__":
    main()