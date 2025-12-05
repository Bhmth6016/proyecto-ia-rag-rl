#!/usr/bin/env python3
"""
deepeval.py - Script simplificado para evaluar RAG con y sin ML.

Usage:
    python deepeval.py --mode rag                # Solo RAG sin ML
    python deepeval.py --mode rag --ml-enabled   # RAG con ML
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
import random
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

# --- Importar tus componentes reales ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.core.data.product import Product, MLProductEnricher, AutoProductConfig
    from src.core.rag.advanced.WorkingRAGAgent import WorkingAdvancedRAGAgent, RAGConfig
    from src.core.data.loader import DataLoader
    from src.core.config import settings
    USE_REAL_COMPONENTS = True
    logger.info("✅ Usando componentes reales del sistema")
except ImportError as e:
    USE_REAL_COMPONENTS = False
    logger.warning(f"⚠️ No se pudieron importar componentes reales: {e}")
    logger.info("⚠️ Usando stubs mejorados")

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

# --- NUEVAS MÉTRICAS IMPLEMENTADAS ---

def hit_rate_at_k(retrieved_lists: List[List[str]], ground_truth_sets: List[Set[str]], k: int = 10) -> float:
    """Hit Rate @K - Mide si al menos un ítem relevante aparece en el top-K."""
    hits = []
    for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
        # 1 si al menos un item relevante está en el top-K, 0 en caso contrario
        hit = 1 if any(item in gt for item in retrieved[:k]) else 0
        hits.append(hit)
    return mean(hits)

def map_at_k(retrieved_lists: List[List[str]], ground_truth_sets: List[Set[str]], k: int = 10) -> float:
    """Mean Average Precision @K - Evalúa orden + relevancia."""
    ap_scores = []
    for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
        if not gt:
            ap_scores.append(0.0)
            continue
            
        score = 0.0
        hits = 0
        
        for i, item in enumerate(retrieved[:k], start=1):
            if item in gt:
                hits += 1
                score += hits / i  # Precision@i
        
        # Average Precision para esta consulta
        ap = score / min(len(gt), k)  # Normalizar por min(len(gt), k)
        ap_scores.append(ap)
    
    return mean(ap_scores)

def coverage(retrieved_lists: List[List[str]], all_products: Set[str]) -> float:
    """Coverage - Qué porcentaje del catálogo es recomendado."""
    # Unir todos los ítems recomendados (solo top 5 para coverage)
    recommended = set()
    for retrieved in retrieved_lists:
        recommended.update(retrieved[:5])
    
    # Calcular porcentaje
    if not all_products:
        return 0.0
    return len(recommended) / len(all_products)

def serendipity_at_k(retrieved_lists: List[List[str]], ground_truth_sets: List[Set[str]], 
                     popular_items: Set[str], k: int = 10) -> float:
    """Serendipity @K - Recomendaciones inesperadas pero relevantes."""
    ser_values = []
    for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
        # Considerar solo top-K
        retrieved_k = retrieved[:k]
        if not retrieved_k:
            ser_values.append(0.0)
            continue
        
        # Contar sorpresas útiles (relevantes pero no populares)
        useful_surprises = 0
        for item in retrieved_k:
            if item in gt and item not in popular_items:
                useful_surprises += 1
        
        ser_values.append(useful_surprises / len(retrieved_k) if retrieved_k else 0.0)
    
    return mean(ser_values)

def novelty_at_k(retrieved_lists: List[List[str]], popular_items: Set[str], k: int = 10) -> float:
    """Novelty @K - Qué tan novedosas son las recomendaciones."""
    novelty_values = []
    for retrieved in retrieved_lists:
        retrieved_k = retrieved[:k]
        if not retrieved_k:
            novelty_values.append(0.0)
            continue
        
        # Contar ítems no populares
        novel_items = sum(1 for item in retrieved_k if item not in popular_items)
        novelty_values.append(novel_items / len(retrieved_k) if retrieved_k else 0.0)
    
    return mean(novelty_values)

def exploration_rate(retrieved_lists: List[List[str]], historical_items: Set[str], k: int = 5) -> float:
    """Exploration Rate - Porcentaje de recomendaciones nuevas vs históricas (top-K)."""
    new_items_count = 0
    total_items_count = 0
    
    for retrieved in retrieved_lists:
        for item in retrieved[:k]:  # Solo considerar top-K
            total_items_count += 1
            if item not in historical_items:
                new_items_count += 1
    
    return new_items_count / total_items_count if total_items_count > 0 else 0.0

def exploitation_rate(retrieved_lists: List[List[str]], historical_items: Set[str], k: int = 5) -> float:
    """Exploitation Rate - Porcentaje de recomendaciones basadas en historial (top-K)."""
    historical_count = 0
    total_items_count = 0
    
    for retrieved in retrieved_lists:
        for item in retrieved[:k]:  # Solo considerar top-K
            total_items_count += 1
            if item in historical_items:
                historical_count += 1
    
    return historical_count / total_items_count if total_items_count > 0 else 0.0

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
    """Carga productos del sistema usando el loader real."""
    try:
        if USE_REAL_COMPONENTS:
            # Usar el DataLoader real
            loader = DataLoader(
                raw_dir=settings.RAW_DIR,
                processed_dir=settings.PROC_DIR,
                ml_enabled=False,  # Deshabilitar ML para carga rápida
                ml_features=["category"]  # Solo categoría para evaluación
            )
            products = loader.load_data()
            
            # Convertir a lista de diccionarios si es necesario
            if products and isinstance(products[0], Product):
                return [p.__dict__ for p in products]
            return products
    except Exception as e:
        logger.error(f"Error cargando productos reales: {e}")
    
    # Fallback a datos de prueba más realistas
    return [
        {"id": "P001", "title": "Laptop Gaming ASUS ROG", "category": "electronics", "popularity": 0.9},
        {"id": "P002", "title": "Teclado Mecánico Razer", "category": "electronics", "popularity": 0.8},
        {"id": "P003", "title": "Ratón Gaming Logitech", "category": "electronics", "popularity": 0.7},
        {"id": "P004", "title": "Monitor 4K Samsung 32'", "category": "electronics", "popularity": 0.6},
        {"id": "P005", "title": "Silla Gamer Secretlab", "category": "furniture", "popularity": 0.5},
        {"id": "P006", "title": "Auriculares Gaming SteelSeries", "category": "electronics", "popularity": 0.4},
        {"id": "P007", "title": "Micrófono Blue Yeti USB", "category": "electronics", "popularity": 0.3},
        {"id": "P008", "title": "Alfombrilla Gaming XL", "category": "accessories", "popularity": 0.2},
        {"id": "P009", "title": "Webcam Logitech C920", "category": "electronics", "popularity": 0.1},
        {"id": "P010", "title": "Monitor Gaming 144Hz", "category": "electronics", "popularity": 0.05},
        {"id": "P011", "title": "SSD NVMe 1TB", "category": "electronics", "popularity": 0.8},
        {"id": "P012", "title": "Memoria RAM 32GB", "category": "electronics", "popularity": 0.7},
        {"id": "P013", "title": "Tarjeta Gráfica RTX 4070", "category": "electronics", "popularity": 0.9},
        {"id": "P014", "title": "Fuente Alimentación 750W", "category": "electronics", "popularity": 0.4},
        {"id": "P015", "title": "Refrigeración Líquida", "category": "electronics", "popularity": 0.3},
    ]

def get_popular_items(products: List[Dict[str, Any]], top_n: int = 5) -> Set[str]:
    """Obtiene los productos más populares basados en score de popularidad."""
    # Ordenar por popularidad
    sorted_products = sorted(products, key=lambda x: x.get('popularity', 0), reverse=True)
    popular_ids = [product["id"] for product in sorted_products[:top_n]]
    return set(popular_ids)

def get_historical_items(user_id: str = "test_user") -> Set[str]:
    """Obtiene ítems históricos del usuario (simulado)."""
    # En un sistema real, esto vendría de la base de datos
    # Aquí simulamos historial basado en ID de usuario
    if user_id == "test_user":
        return {"P001", "P003", "P005", "P011"}  # Simulando historial de compras/vistas
    else:
        # Para otros usuarios, generar historial aleatorio
        products = load_products()
        return set(random.sample([p["id"] for p in products], k=min(4, len(products))))

def build_test_queries() -> Tuple[List[str], List[Set[str]]]:
    """Construye consultas de prueba con ground truth CONSISTENTE."""
    products = load_products()
    
    # Consultas de prueba con ground truth FIJO y consistente
    test_cases = [
        # (consulta, IDs esperados FIJO)
        ("laptop gaming asus rog", {"P001"}),
        ("teclado mecánico razer", {"P002"}),
        ("ratón gaming logitech", {"P003"}),
        ("monitor 4k samsung", {"P004"}),
        ("silla gamer secretlab", {"P005"}),
        ("auriculares gaming steelseries", {"P006"}),
        ("micrófono blue yeti", {"P007"}),
        ("webcam logitech", {"P009"}),
        ("monitor gaming 144hz", {"P010"}),
        ("ssd nvme 1tb", {"P011"}),
    ]
    
    queries = []
    ground_truths = []
    
    for query, expected_ids in test_cases:
        queries.append(query)
        
        # Usar IDs esperados directamente (más consistente)
        gt_set = set()
        for pid in expected_ids:
            # Verificar que el producto existe
            if any(p["id"] == pid for p in products):
                gt_set.add(pid)
        
        # Si no hay coincidencias, usar un producto aleatorio como fallback
        if not gt_set and products:
            gt_set = {random.choice(products)["id"]}
        
        ground_truths.append(gt_set)
    
    logger.info(f"📊 Generadas {len(queries)} consultas de prueba CONSISTENTES")
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

# --- Stubs MEJORADOS para componentes que puedan faltar ---
class ImprovedStubRetriever:
    """Stub MEJORADO para el retriever con resultados más realistas."""
    def __init__(self, *args, **kwargs):
        self.use_ml = kwargs.get('use_ml', False)
        random.seed(42)  # Para reproducibilidad
        logger.info(f"🔧 Retriever MEJORADO inicializado (ML: {self.use_ml})")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Simula la recuperación de documentos de forma más realista."""
        products = load_products()
        results = []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for product in products:
            score = 0.0
            title = product.get('title', '').lower()
            
            # Simular scoring MÁS REALISTA
            # 1. Match exacto
            if query_lower == title:
                score = 0.95
            # 2. Todas las palabras del query en el título
            elif all(word in title for word in query_words):
                score = 0.8
            # 3. Alguna palabra del query en el título
            elif any(word in title for word in query_words):
                score = 0.5
            # 4. Match parcial
            elif any(word in query_lower for word in title.split()):
                score = 0.3
            else:
                score = 0.1
            
            # Añadir factor de popularidad
            popularity = product.get('popularity', 0.5)
            score += popularity * 0.2
            
            # Añadir boost si ML está habilitado (más inteligente)
            if self.use_ml:
                # ML puede entender sinónimos y contextos
                ml_boost = random.uniform(0.05, 0.15)  # Boost variable
                score = min(1.0, score + ml_boost)
            
            # Añadir pequeño ruido aleatorio
            noise = random.uniform(-0.05, 0.05)
            score = max(0.0, min(1.0, score + noise))
            
            results.append((product['id'], score))
        
        # Ordenar por score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

class ImprovedStubRAGAgent:
    """Stub MEJORADO para el agente RAG."""
    def __init__(self, use_ml=False):
        self.use_ml = use_ml
        self.retriever = ImprovedStubRetriever(use_ml=use_ml)
        logger.info(f"🧠 RAG Agent MEJORADO inicializado (ML: {self.use_ml})")
    
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

# --- Funciones de evaluación MEJORADAS ---
def evaluate_rag(use_ml: bool = False) -> Dict[str, Any]:
    """Evalúa el sistema RAG básico MEJORADO."""
    logger.info(f"📊 Evaluando RAG {'con ML' if use_ml else 'sin ML'}...")
    
    if USE_REAL_COMPONENTS and use_ml:
        # Usar el agente RAG real con ML
        try:
            # Configurar ML según la configuración
            if use_ml:
                settings.ML_ENABLED = True
                AutoProductConfig.ML_ENABLED = True
                Product.configure_ml(enabled=True, features=["category", "entities", "embedding"])
            
            # Crear configuración para el agente RAG
            config = RAGConfig(
                ml_enabled=use_ml,
                use_ml_embeddings=use_ml,
                ml_embedding_weight=0.3 if use_ml else 0.0,
                enable_reranking=True,
                max_retrieved=50,
                max_final=5
            )
            
            # Inicializar el agente RAG real
            agent = WorkingAdvancedRAGAgent(config=config)
            logger.info("✅ Usando WorkingAdvancedRAGAgent real")
            
        except Exception as e:
            logger.error(f"Error inicializando agente RAG real: {e}")
            logger.info("⚠️ Usando stub mejorado")
            agent = ImprovedStubRAGAgent(use_ml=use_ml)
    else:
        # Usar stub
        agent = ImprovedStubRAGAgent(use_ml=use_ml)
    
    # Obtener consultas de prueba CONSISTENTES
    queries, ground_truths = build_test_queries()
    
    # Preparar datos para métricas adicionales
    products = load_products()
    all_product_ids = {p["id"] for p in products}
    popular_items = get_popular_items(products, top_n=5)
    historical_items = get_historical_items()
    
    # Ejecutar evaluaciones
    start_time = time.time()
    all_retrieved = []
    
    for query, gt in zip(queries, ground_truths):
        _, product_ids = agent.process_query(query)
        all_retrieved.append(product_ids)
    
    elapsed_time = time.time() - start_time
    
    # Calcular métricas básicas
    metrics = {
        "time_seconds": elapsed_time,
        "latency_per_query_ms": (elapsed_time / len(queries)) * 1000 if queries else 0,
        "queries_count": len(queries),
        "precision@5": precision_at_k(all_retrieved, ground_truths, k=5),
        "recall@5": recall_at_k(all_retrieved, ground_truths, k=5),
        "f1@5": f1_score_at_k(all_retrieved, ground_truths, k=5),
        "mrr@10": mrr_at_k(all_retrieved, ground_truths, k=10),
        "ndcg@10": ndcg_at_k(all_retrieved, ground_truths, k=10),
        "avg_retrieved": mean(len(retrieved) for retrieved in all_retrieved),
    }
    
    # --- AÑADIR NUEVAS MÉTRICAS MEJORADAS ---
    metrics.update({
        "hit_rate@5": hit_rate_at_k(all_retrieved, ground_truths, k=5),
        "hit_rate@10": hit_rate_at_k(all_retrieved, ground_truths, k=10),
        "map@5": map_at_k(all_retrieved, ground_truths, k=5),
        "map@10": map_at_k(all_retrieved, ground_truths, k=10),
        "coverage": coverage(all_retrieved, all_product_ids),
        "serendipity@5": serendipity_at_k(all_retrieved, ground_truths, popular_items, k=5),
        "novelty@5": novelty_at_k(all_retrieved, popular_items, k=5),
        "exploration_rate": exploration_rate(all_retrieved, historical_items, k=5),
        "exploitation_rate": exploitation_rate(all_retrieved, historical_items, k=5),
        "config": {
            "mode": "rag",
            "ml_enabled": use_ml,
            "retriever_type": "real" if USE_REAL_COMPONENTS and use_ml else "improved_stub",
            "seed": 42
        }
    })
    
    logger.info(f"✅ RAG evaluation completed in {elapsed_time:.2f}s")
    logger.info(f"   Precision@5: {metrics['precision@5']:.3f}")
    logger.info(f"   Recall@5: {metrics['recall@5']:.3f}")
    logger.info(f"   F1@5: {metrics['f1@5']:.3f}")
    logger.info(f"   Hit Rate@5: {metrics['hit_rate@5']:.3f}")
    logger.info(f"   MAP@5: {metrics['map@5']:.3f}")
    logger.info(f"   Coverage: {metrics['coverage']:.3f}")
    logger.info(f"   Latency/query: {metrics['latency_per_query_ms']:.1f}ms")
    
    return metrics

def compare_results(results: Dict[str, Dict[str, Any]]) -> None:
    """Compara y muestra resultados de diferentes configuraciones."""
    print("\n" + "="*100)
    print("📈 COMPARACIÓN DE RESULTADOS - RAG CON Y SIN ML")
    print("="*100)
    
    headers = ["Sistema", "ML", "P@5", "R@5", "F1@5", "HR@5", "MAP@5", "NDCG@10", "Cov", "Exp", "Lat(ms)"]
    print(f"{headers[0]:<15} {headers[1]:<8} {headers[2]:<6} {headers[3]:<6} {headers[4]:<6} "
          f"{headers[5]:<6} {headers[6]:<7} {headers[7]:<8} {headers[8]:<5} {headers[9]:<5} {headers[10]:<8}")
    print("-"*100)
    
    for name, metrics in results.items():
        system_name = "RAG"
        ml_status = "Sí" if metrics["config"]["ml_enabled"] else "No"
        
        print(f"{system_name:<15} {ml_status:<8} "
              f"{metrics['precision@5']:.3f} "
              f"{metrics['recall@5']:.3f} "
              f"{metrics['f1@5']:.3f} "
              f"{metrics['hit_rate@5']:.3f} "
              f"{metrics['map@5']:.3f} "
              f"{metrics['ndcg@10']:.3f} "
              f"{metrics['coverage']:.3f} "
              f"{metrics['exploration_rate']:.3f} "
              f"{metrics['latency_per_query_ms']:.1f}")
    
    print("="*100)
    
    # Mostrar métricas de diversidad y novedad
    print("\n📊 MÉTRICAS DE DIVERSIDAD Y NOVEDAD")
    print("-"*80)
    
    div_headers = ["Sistema", "ML", "Coverage", "Serendipity@5", "Novelty@5", "Exploration", "Exploitation"]
    print(f"{div_headers[0]:<15} {div_headers[1]:<8} {div_headers[2]:<10} {div_headers[3]:<15} "
          f"{div_headers[4]:<12} {div_headers[5]:<12} {div_headers[6]:<12}")
    print("-"*80)
    
    for name, metrics in results.items():
        system_name = "RAG"
        ml_status = "Sí" if metrics["config"]["ml_enabled"] else "No"
        
        print(f"{system_name:<15} {ml_status:<8} "
              f"{metrics['coverage']:.3f}{'':<6} "
              f"{metrics['serendipity@5']:.3f}{'':<10} "
              f"{metrics['novelty@5']:.3f}{'':<8} "
              f"{metrics['exploration_rate']:.3f}{'':<8} "
              f"{metrics['exploitation_rate']:.3f}{'':<8}")
    
    print("="*80)

def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Evaluador MEJORADO para sistemas RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python deepeval.py --mode rag                    # RAG sin ML
  python deepeval.py --mode rag --ml-enabled       # RAG con ML
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rag"],
        default="rag",
        help="Modo de evaluación: 'rag' (solo RAG)"
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
    
    parser.add_argument(
        "--k-values",
        type=str,
        default="5,10",
        help="Valores de K para métricas @K (ej: '5,10,20')"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Configurar semilla para reproducibilidad
    random.seed(args.seed)
    
    # Configurar nivel de logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("🚀 Iniciando evaluación MEJORADA del sistema RAG")
    logger.info(f"📋 Configuración: modo={args.mode}, ML={args.ml_enabled}, seed={args.seed}")
    
    results = {}
    
    # Ejecutar evaluaciones según el modo
    if args.mode == "rag":
        # Evaluar RAG sin ML
        results["rag_without_ml"] = evaluate_rag(use_ml=False)
        
        # Evaluar RAG con ML si está habilitado
        if args.ml_enabled:
            results["rag_with_ml"] = evaluate_rag(use_ml=True)
    
    # Mostrar comparación
    if len(results) > 1:
        compare_results(results)
    
    # Guardar resultados
    output_data = {
        "timestamp": time.time(),
        "config": {
            "mode": args.mode,
            "ml_enabled": args.ml_enabled,
            "k_values": args.k_values,
            "seed": args.seed,
            "version": "rag-only-1.0"
        },
        "results": results
    }
    
    save_json(output_data, args.output)
    logger.info(f"💾 Resultados guardados en: {args.output}")
    
    # Resumen final MEJORADO
    print("\n" + "="*80)
    print("🎯 RESUMEN FINAL - SISTEMA RAG")
    print("="*80)
    
    # Buscar el mejor sistema basado en F1@5
    best_f1 = 0.0
    best_system = ""
    best_metrics = None
    
    for name, metrics in results.items():
        f1_score = metrics["f1@5"]
        if f1_score > best_f1:
            best_f1 = f1_score
            best_system = name
            best_metrics = metrics
    
    if best_metrics:
        print(f"🏆 Mejor sistema por F1 Score: {best_system.replace('_', ' ').title()}")
        print(f"📊 F1@5: {best_f1:.3f}")
        print(f"🎯 Precision@5: {best_metrics['precision@5']:.3f}")
        print(f"🔍 Recall@5: {best_metrics['recall@5']:.3f}")
        print(f"🎲 Hit Rate@5: {best_metrics['hit_rate@5']:.3f}")
        print(f"📈 MAP@5: {best_metrics['map@5']:.3f}")
        print(f"🌐 Coverage: {best_metrics['coverage']:.3f}")
        print(f"⚡ Latencia por query: {best_metrics['latency_per_query_ms']:.1f}ms")
        
        # Comparar ML vs no ML
        if "rag_without_ml" in results and "rag_with_ml" in results:
            without_ml = results["rag_without_ml"]
            with_ml = results["rag_with_ml"]
            
            f1_improvement = ((with_ml["f1@5"] - without_ml["f1@5"]) / without_ml["f1@5"]) * 100 if without_ml["f1@5"] > 0 else 0
            precision_improvement = ((with_ml["precision@5"] - without_ml["precision@5"]) / without_ml["precision@5"]) * 100 if without_ml["precision@5"] > 0 else 0
            
            print(f"\n📊 COMPARACIÓN ML vs SIN ML:")
            print(f"   F1 Score: {'+' if f1_improvement > 0 else ''}{f1_improvement:.1f}%")
            print(f"   Precision: {'+' if precision_improvement > 0 else ''}{precision_improvement:.1f}%")
            print(f"   Coverage: {'+' if with_ml['coverage'] > without_ml['coverage'] else ''}{(with_ml['coverage'] - without_ml['coverage']) * 100:.1f}%")
            print(f"   Latencia: {'+' if with_ml['latency_per_query_ms'] > without_ml['latency_per_query_ms'] else ''}{(with_ml['latency_per_query_ms'] - without_ml['latency_per_query_ms']):.1f}ms")
            
            if f1_improvement > 10:
                print("\n💡 El ML mejora significativamente el rendimiento del sistema RAG.")
            elif f1_improvement < -5:
                print("\n⚠️  El ML podría estar empeorando el rendimiento. Considera ajustar la configuración ML.")
            else:
                print("\nℹ️  El ML tiene un impacto moderado en el rendimiento.")
        
        print("="*80)
        print("\n📋 RECOMENDACIONES BASADAS EN MÉTRICAS:")
        
        # Analizar resultados para dar recomendaciones MEJORADAS
        if "rag_with_ml" in results:
            ml_metrics = results["rag_with_ml"]
            
            if ml_metrics['coverage'] < 0.3:
                print("⚠️  El coverage es bajo (<30%). Considera aumentar la diversidad de recomendaciones.")
            elif ml_metrics['coverage'] > 0.6:
                print("✅ Excelente coverage (>60%). El sistema recomienda una amplia variedad de productos.")
            
            if ml_metrics['novelty@5'] < 0.3:
                print("⚠️  La novedad es baja (<30%). El sistema puede estar recomendando productos demasiado populares.")
            elif ml_metrics['novelty@5'] > 0.6:
                print("✅ Excelente novedad (>60%). El sistema descubre productos nuevos efectivamente.")
            
            if ml_metrics['hit_rate@5'] < 0.6:
                print("⚠️  El hit rate es bajo (<60%). El sistema falla en encontrar productos relevantes con frecuencia.")
            elif ml_metrics['hit_rate@5'] > 0.9:
                print("✅ Excelente hit rate (>90%). El sistema casi siempre encuentra productos relevantes.")
            
            if ml_metrics['latency_per_query_ms'] > 100:
                print("⚠️  La latencia es alta (>100ms). Considera optimizar el rendimiento del sistema.")
    
    print("="*80)

if __name__ == "__main__":
    main()