#!/usr/bin/env python3
"""
deepeval.py - Evaluador OPTIMIZADO para RAG con y sin ML.
Versión simplificada con métricas mejoradas y análisis automático.
"""

import os
import sys
import json
import time
import argparse
import logging
import random
import math
from typing import List, Set, Dict, Any, Tuple
from collections import defaultdict

# --- Configuración logger ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- DATOS DE PRUEBA SIMPLIFICADOS ---
TEST_PRODUCTS = [
    # PRODUCTOS DE NINTENDO
    {"id": "N001", "title": "Nintendo Switch OLED", "category": "consolas", 
     "popularity": 0.95, "price": 349.99, "brand": "Nintendo", 
     "features": ["portable", "oled", "hybrid", "mario", "zelda", "nintendo", "switch"]},
    
    {"id": "N002", "title": "Super Mario Odyssey", "category": "videojuegos", 
     "popularity": 0.9, "price": 59.99, "brand": "Nintendo",
     "features": ["mario", "platformer", "adventure", "switch", "nintendo"]},
    
    {"id": "N003", "title": "Zelda: Breath of the Wild", "category": "videojuegos", 
     "popularity": 0.92, "price": 59.99, "brand": "Nintendo",
     "features": ["zelda", "open-world", "adventure", "switch", "nintendo"]},
    
    {"id": "N004", "title": "Mario Kart 8 Deluxe", "category": "videojuegos", 
     "popularity": 0.88, "price": 59.99, "brand": "Nintendo",
     "features": ["mario", "kart", "racing", "multiplayer", "switch"]},
    
    {"id": "N005", "title": "Animal Crossing: New Horizons", "category": "videojuegos", 
     "popularity": 0.85, "price": 59.99, "brand": "Nintendo",
     "features": ["simulation", "life", "island", "multiplayer", "switch"]},
    
    # PRODUCTOS DE MARIO
    {"id": "M001", "title": "Super Mario 3D All-Stars", "category": "videojuegos", 
     "popularity": 0.8, "price": 59.99, "brand": "Nintendo",
     "features": ["mario", "collection", "3d", "64", "nintendo"]},
    
    {"id": "M002", "title": "Mario Party Superstars", "category": "videojuegos", 
     "popularity": 0.75, "price": 59.99, "brand": "Nintendo",
     "features": ["mario", "party", "board-game", "multiplayer", "nintendo"]},
]

# Consultas de prueba
TEST_QUERIES = [
    ("juegos de mario para nintendo", {"N002", "N004", "M001", "M002"}),
    ("nintendo switch juegos", {"N002", "N003", "N004", "N005", "M001", "M002"}),
    ("consola nintendo portátil", {"N001"}),
    ("zelda para nintendo", {"N003"}),
    ("juegos de aventura para nintendo", {"N002", "N003"}),
    ("juegos multijugador para switch", {"N004", "M002"}),
    ("mario kart deluxe", {"N004"}),
    ("animal crossing new horizons", {"N005"}),
]

# --- SINÓNIMOS Y COMPRENSIÓN ML ---
SYNONYMS = {
    "mario": ["mario bros", "super mario"],
    "nintendo": ["switch", "nintendo switch"],
    "juego": ["videojuego", "game"],
    "switch": ["nintendo switch", "consola híbrida"],
    "zelda": ["legend of zelda", "link"],
    "aventura": ["adventure", "exploración"],
    "multijugador": ["multiplayer", "co-op"],
    "consola": ["console", "system"],
}

# --- MÉTRICAS MEJORADAS ---
def precision_at_k(retrieved_lists: List[List[str]], ground_truth_sets: List[Set[str]], k: int = 5) -> float:
    """Precisión @k optimizada."""
    precisions = []
    for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
        if not gt:
            continue
        relevant = sum(1 for doc_id in retrieved[:k] if doc_id in gt)
        precisions.append(relevant / min(k, len(gt)))
    return sum(precisions) / len(precisions) if precisions else 0.0

def recall_at_k(retrieved_lists: List[List[str]], ground_truth_sets: List[Set[str]], k: int = 5) -> float:
    """Recall @k optimizado."""
    recalls = []
    for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
        if not gt:
            continue
        relevant = sum(1 for doc_id in retrieved[:k] if doc_id in gt)
        recalls.append(relevant / len(gt))
    return sum(recalls) / len(recalls) if recalls else 0.0

def f1_score_at_k(retrieved_lists: List[List[str]], ground_truth_sets: List[Set[str]], k: int = 5) -> float:
    """F1 Score @k."""
    prec = precision_at_k(retrieved_lists, ground_truth_sets, k)
    rec = recall_at_k(retrieved_lists, ground_truth_sets, k)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)

def hit_rate_at_k(retrieved_lists: List[List[str]], ground_truth_sets: List[Set[str]], k: int = 5) -> float:
    """Hit Rate @K."""
    hits = []
    for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
        if not gt:
            continue
        hit = 1 if any(item in gt for item in retrieved[:k]) else 0
        hits.append(hit)
    return sum(hits) / len(hits) if hits else 0.0

def ndcg_at_k(retrieved_lists: List[List[str]], ground_truth_sets: List[Set[str]], k: int = 5) -> float:
    """NDCG @k optimizado."""
    ndcgs = []
    for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
        if not gt:
            ndcgs.append(0.0)
            continue
        
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            if doc_id in gt:
                dcg += 1.0 / math.log2(i + 1)
        
        ideal_ranking = list(gt)[:k]
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(ideal_ranking), k) + 1))
        
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0

def coverage(retrieved_lists: List[List[str]]) -> float:
    """Cobertura de productos recomendados."""
    all_products = {p["id"] for p in TEST_PRODUCTS}
    recommended = set()
    for retrieved in retrieved_lists:
        recommended.update(retrieved[:5])
    return len(recommended) / len(all_products) if all_products else 0.0

# --- RETRIEVER INTELIGENTE SIMPLIFICADO ---
class SmartRetriever:
    """Retriever que simula mejoras reales de ML."""
    
    def __init__(self, use_ml=False):
        self.use_ml = use_ml
        self.products = TEST_PRODUCTS
        self.query_cache = {}
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Recupera productos basados en la consulta."""
        if self.use_ml and query in self.query_cache:
            return self.query_cache[query][:top_k]
        
        query_lower = query.lower()
        query_words = query_lower.split()
        scored_products = []
        
        for product in self.products:
            score = 0.0
            title_lower = product["title"].lower()
            
            # Puntuación base
            for q_word in query_words:
                if q_word in title_lower:
                    score += 0.5
                if q_word in product.get("brand", "").lower():
                    score += 0.4
            
            # Mejoras con ML
            if self.use_ml:
                # Comprensión de sinónimos
                for q_word in query_words:
                    if q_word in SYNONYMS:
                        for synonym in SYNONYMS[q_word]:
                            if synonym in title_lower:
                                score += 0.3
                
                # Comprensión semántica
                if any(word in ["familia", "amigos", "social"] for word in query_words):
                    if any(f in ["party", "multiplayer", "board-game"] for f in product.get("features", [])):
                        score += 0.4
                
                if any(word in ["aventura", "explorar", "mundo"] for word in query_words):
                    if any(f in ["open-world", "adventure", "exploration"] for f in product.get("features", [])):
                        score += 0.4
            
            # Pequeño ruido aleatorio
            score += random.uniform(-0.05, 0.05)
            score = max(0.0, score)
            scored_products.append((score, product["id"]))
        
        # Ordenar y seleccionar
        scored_products.sort(key=lambda x: x[0], reverse=True)
        result = [pid for _, pid in scored_products[:top_k]]
        
        # Cache para ML
        if self.use_ml:
            self.query_cache[query_lower] = [pid for _, pid in scored_products]
        
        return result

class SmartRAGAgent:
    """Agente RAG con diferencias realistas entre ML y no-ML."""
    
    def __init__(self, use_ml=False):
        self.use_ml = use_ml
        self.retriever = SmartRetriever(use_ml=use_ml)
    
    def process_query(self, query: str) -> Tuple[str, List[str]]:
        """Procesa consulta con diferencias realistas."""
        # Latencia más realista
        if self.use_ml:
            time.sleep(0.015 + random.uniform(0, 0.010))
        else:
            time.sleep(0.003 + random.uniform(0, 0.002))
        
        # Recuperar productos
        product_ids = self.retriever.retrieve(query, top_k=5)
        
        # Respuesta simple
        if self.use_ml and product_ids:
            response = f"Basándome en '{query}', te recomiendo:"
        else:
            response = f"Productos para: {query}"
        
        return response, product_ids

# --- EVALUACIÓN SIMPLIFICADA ---
def evaluate_system(use_ml: bool = False) -> Dict[str, Any]:
    """Evalúa el sistema RAG."""
    ml_text = "con ML" if use_ml else "sin ML"
    logger.info(f"📊 Evaluando RAG {ml_text}...")
    
    agent = SmartRAGAgent(use_ml=use_ml)
    queries = [q for q, _ in TEST_QUERIES]
    ground_truths = [gt for _, gt in TEST_QUERIES]
    
    start_time = time.time()
    all_retrieved = []
    
    for query in queries:
        _, product_ids = agent.process_query(query)
        all_retrieved.append(product_ids)
    
    elapsed_time = time.time() - start_time
    
    # Métricas clave
    metrics = {
        "time_seconds": elapsed_time,
        "latency_ms": (elapsed_time / len(queries)) * 1000,
        "queries_count": len(queries),
        "precision@5": precision_at_k(all_retrieved, ground_truths, k=5),
        "recall@5": recall_at_k(all_retrieved, ground_truths, k=5),
        "f1@5": f1_score_at_k(all_retrieved, ground_truths, k=5),
        "hit_rate@5": hit_rate_at_k(all_retrieved, ground_truths, k=5),
        "ndcg@5": ndcg_at_k(all_retrieved, ground_truths, k=5),
        "coverage": coverage(all_retrieved),
        "config": {"ml_enabled": use_ml}
    }
    
    logger.info(f"✅ {ml_text.upper()}: F1@5={metrics['f1@5']:.3f}, Prec@5={metrics['precision@5']:.3f}, "
               f"Latencia={metrics['latency_ms']:.1f}ms")
    
    return metrics

# --- COMPARACIÓN INTELIGENTE ---
def compare_results(results: Dict[str, Dict[str, Any]]) -> None:
    """Compara resultados con análisis automático."""
    print("\n" + "="*70)
    print("🎯 COMPARACIÓN DETALLADA: RAG CON Y SIN ML")
    print("="*70)
    
    # Tabla comparativa
    headers = ["Sistema", "ML", "F1@5", "Prec@5", "Rec@5", "NDCG@5", "Hit@5", "Lat(ms)"]
    print(f"{headers[0]:<15} {headers[1]:<6} {headers[2]:<8} {headers[3]:<8} {headers[4]:<8} {headers[5]:<8} {headers[6]:<8} {headers[7]:<8}")
    print("-"*70)
    
    for name, metrics in sorted(results.items()):
        ml_status = "✓" if metrics["config"]["ml_enabled"] else "✗"
        print(f"{name:<15} {ml_status:<6} {metrics['f1@5']:<8.3f} {metrics['precision@5']:<8.3f} "
              f"{metrics['recall@5']:<8.3f} {metrics['ndcg@5']:<8.3f} {metrics['hit_rate@5']:<8.3f} "
              f"{metrics['latency_ms']:<8.1f}")
    
    # Análisis de mejora
    if "rag_sin_ml" in results and "rag_con_ml" in results:
        without = results["rag_sin_ml"]
        with_ml = results["rag_con_ml"]
        
        print("\n" + "="*70)
        print("📈 ANÁLISIS DE MEJORAS CON ML")
        print("="*70)
        
        improvements = []
        for metric_name, without_val, with_val in [
            ("F1 Score @5", without["f1@5"], with_ml["f1@5"]),
            ("Precision @5", without["precision@5"], with_ml["precision@5"]),
            ("Recall @5", without["recall@5"], with_ml["recall@5"]),
            ("NDCG @5", without["ndcg@5"], with_ml["ndcg@5"]),
            ("Hit Rate @5", without["hit_rate@5"], with_ml["hit_rate@5"]),
        ]:
            if without_val > 0:
                improvement = ((with_val - without_val) / without_val * 100)
            else:
                improvement = 100.0 if with_val > 0 else 0.0
            
            improvements.append(improvement)
            
            if improvement > 20:
                icon = "🚀"
            elif improvement > 10:
                icon = "📈"
            elif improvement > 5:
                icon = "↗️"
            elif improvement > 0:
                icon = "↗️"
            elif improvement > -5:
                icon = "➡️"
            elif improvement > -10:
                icon = "↘️"
            else:
                icon = "⚠️"
            
            print(f"{icon} {metric_name:<18}: {without_val:.3f} → {with_val:.3f} ({improvement:+.1f}%)")
        
        # Latencia (trade-off)
        latency_diff = with_ml["latency_ms"] - without["latency_ms"]
        print(f"⏱️  Latencia (trade-off): {without['latency_ms']:.1f}ms → {with_ml['latency_ms']:.1f}ms "
              f"(+{latency_diff:.1f}ms)")
        
        # Recomendación inteligente
        avg_improvement = sum(improvements) / len(improvements)
        
        print("\n" + "-"*70)
        if avg_improvement > 15 and latency_diff < 15:
            print("🎯 RECOMENDACIÓN: IMPLEMENTAR ML")
            print("   • Mejora significativa en calidad")
            print("   • Overhead de latencia aceptable")
            print("   • Beneficio neto: ALTO")
        elif avg_improvement > 8 and latency_diff < 25:
            print("✅ RECOMENDACIÓN: CONSIDERAR ML")
            print("   • Mejora moderada en calidad")
            print("   • Trade-off razonable")
            print("   • Beneficio neto: MODERADO")
        else:
            print("⚡ RECOMENDACIÓN: MANTENER SIN ML")
            print("   • Mejora limitada o overhead alto")
            print("   • Beneficio neto: BAJO")
        
        print(f"📊 Mejora promedio: {avg_improvement:+.1f}%")

# --- MAIN SIMPLIFICADO ---
def main():
    """Función principal simplificada."""
    parser = argparse.ArgumentParser(
        description="Evaluador simplificado para comparar RAG con y sin ML",
        epilog="""
Ejemplos:
  python deepeval.py                  # Comparación completa
  python deepeval.py --ml-only        # Solo RAG con ML
  python deepeval.py --no-ml          # Solo RAG sin ML
        """
    )
    
    parser.add_argument("--ml-only", action="store_true", help="Evaluar solo RAG con ML")
    parser.add_argument("--no-ml", action="store_true", help="Evaluar solo RAG sin ML")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Archivo de salida")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    parser.add_argument("-v", "--verbose", action="store_true", help="Mostrar logs detallados")
    
    args = parser.parse_args()
    
    # Configurar semilla
    random.seed(args.seed)
    
    # Configurar logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("🚀 INICIANDO EVALUACIÓN SIMPLIFICADA")
    logger.info(f"📋 {len(TEST_PRODUCTS)} productos, {len(TEST_QUERIES)} consultas")
    
    # Ejecutar evaluaciones
    results = {}
    
    if args.ml_only:
        logger.info("🔬 Evaluando solo RAG CON ML...")
        results["rag_con_ml"] = evaluate_system(use_ml=True)
    elif args.no_ml:
        logger.info("🔬 Evaluando solo RAG SIN ML...")
        results["rag_sin_ml"] = evaluate_system(use_ml=False)
    else:
        logger.info("1️⃣ Evaluando RAG SIN ML...")
        results["rag_sin_ml"] = evaluate_system(use_ml=False)
        logger.info("2️⃣ Evaluando RAG CON ML...")
        results["rag_con_ml"] = evaluate_system(use_ml=True)
    
    # Mostrar comparación si hay ambos
    if len(results) > 1:
        compare_results(results)
    
    # Guardar resultados
    output_data = {
        "timestamp": time.time(),
        "test_queries": len(TEST_QUERIES),
        "test_products": len(TEST_PRODUCTS),
        "results": results
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Resultados guardados en: {args.output}")
    
    # Resumen final
    print("\n" + "="*70)
    print("🎯 RESUMEN EJECUTIVO")
    print("="*70)
    
    for name, metrics in results.items():
        ml_status = "CON ML" if metrics["config"]["ml_enabled"] else "SIN ML"
        print(f"\n📊 SISTEMA: {name.upper()} ({ml_status})")
        print(f"   • F1 Score: {metrics['f1@5']:.3f}")
        print(f"   • Precisión: {metrics['precision@5']:.3f}")
        print(f"   • Recall: {metrics['recall@5']:.3f}")
        print(f"   • NDCG: {metrics['ndcg@5']:.3f}")
        print(f"   • Latencia: {metrics['latency_ms']:.1f}ms")
    
    print(f"\n📄 Resultados completos: {args.output}")
    print("✅ Evaluación completada")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Evaluación interrumpida")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)