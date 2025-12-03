#!/usr/bin/env python3
"""
deepeval_fixed.py - Script CORREGIDO para evaluación consistente
"""
import json
import time
import logging
from typing import List, Set, Dict, Any, Tuple
import os
import sys

# --- Configuración logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def mean(iterable):
    values = list(iterable)
    return sum(values) / len(values) if values else 0.0

# --- MÉTRICAS BÁSICAS (CONSISTENTES) ---
def precision_at_k(retrieved_lists, ground_truth_sets, k=5):
    precisions = []
    for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
        relevant = sum(1 for doc_id in retrieved[:k] if doc_id in gt)
        precisions.append(relevant / k if k > 0 else 0.0)
    return mean(precisions)

def recall_at_k(retrieved_lists, ground_truth_sets, k=5):
    recalls = []
    for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
        if not gt:
            recalls.append(0.0)
            continue
        relevant = sum(1 for doc_id in retrieved[:k] if doc_id in gt)
        recalls.append(relevant / len(gt))
    return mean(recalls)

def f1_score_at_k(retrieved_lists, ground_truth_sets, k=5):
    p = precision_at_k(retrieved_lists, ground_truth_sets, k)
    r = recall_at_k(retrieved_lists, ground_truth_sets, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

# --- MÉTRICAS NUEVAS (CORREGIDAS) ---
def hit_rate_at_k(retrieved_lists, ground_truth_sets, k=5):
    """Hit Rate @K - Mide si al menos un ítem relevante está en top-K."""
    hits = []
    for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
        # Verificar si AL MENOS UN item relevante está en top-K
        hit_found = False
        for item in retrieved[:k]:
            if item in gt:
                hit_found = True
                break
        hits.append(1.0 if hit_found else 0.0)
    return mean(hits)

def map_at_k(retrieved_lists, ground_truth_sets, k=5):
    """Mean Average Precision @K."""
    ap_scores = []
    for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
        if not gt:  # Si no hay ground truth
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

def coverage(retrieved_lists, all_product_ids):
    """Coverage - Porcentaje del catálogo recomendado."""
    if not all_product_ids:
        return 0.0
    
    recommended = set()
    for retrieved in retrieved_lists:
        recommended.update(retrieved)
    
    return len(recommended) / len(all_product_ids)

# --- SISTEMA DE PRUEBA CONSISTENTE ---
class ConsistentTestSystem:
    """Sistema de prueba que genera resultados consistentes."""
    
    def __init__(self, mode="rag", use_ml=False):
        self.mode = mode
        self.use_ml = use_ml
        self.products = self._load_products()
        self.all_product_ids = {p["id"] for p in self.products}
        
    def _load_products(self):
        """Carga productos consistentes."""
        return [
            {"id": "P001", "title": "Laptop Gaming", "category": "electronics", "popularity": 0.9},
            {"id": "P002", "title": "Teclado Mecánico", "category": "electronics", "popularity": 0.8},
            {"id": "P003", "title": "Ratón Gaming", "category": "electronics", "popularity": 0.7},
            {"id": "P004", "title": "Monitor 4K", "category": "electronics", "popularity": 0.6},
            {"id": "P005", "title": "Silla Gamer", "category": "furniture", "popularity": 0.5},
            {"id": "P006", "title": "Auriculares Gaming", "category": "electronics", "popularity": 0.4},
            {"id": "P007", "title": "Micrófono USB", "category": "electronics", "popularity": 0.3},
            {"id": "P008", "title": "Alfombrilla Gaming", "category": "accessories", "popularity": 0.2},
            {"id": "P009", "title": "Webcam 4K", "category": "electronics", "popularity": 0.1},
            {"id": "P010", "title": "Monitor 27 pulgadas", "category": "electronics", "popularity": 0.05},
        ]
    
    def get_popular_items(self, top_n=3):
        """Obtiene productos populares basados en popularidad."""
        sorted_products = sorted(self.products, key=lambda x: x["popularity"], reverse=True)
        return {p["id"] for p in sorted_products[:top_n]}
    
    def build_test_queries(self):
        """Construye consultas de prueba CONSISTENTES."""
        test_cases = [
            ("laptop gaming", ["P001"]),  # Solo laptop gaming
            ("teclado", ["P002"]),        # Solo teclado
            ("ratón gaming", ["P003"]),   # Solo ratón gaming
            ("monitor", ["P004", "P010"]), # Dos monitores
            ("silla gamer", ["P005"]),    # Solo silla
            ("productos gaming", ["P001", "P003", "P005", "P006", "P008"]),  # 5 productos gaming
            ("accesorios computadora", ["P002", "P003", "P006", "P007", "P009"]),  # 5 accesorios
        ]
        
        queries = []
        ground_truths = []
        
        for query, expected_ids in test_cases:
            queries.append(query)
            ground_truths.append(set(expected_ids))
        
        logger.info(f"Generadas {len(queries)} consultas consistentes")
        return queries, ground_truths
    
    def simulate_rag_retrieval(self, query, top_k=10):
        """Simula recuperación RAG de forma consistente."""
        query_lower = query.lower()
        results = []
        
        for product in self.products:
            score = 0.0
            title = product["title"].lower()
            
            # Scoring básico pero consistente
            if query_lower == title:
                score = 1.0
            elif all(word in title for word in query_lower.split()):
                score = 0.8
            elif any(word in title for word in query_lower.split()):
                score = 0.5
            else:
                score = 0.1
            
            # Boost ML si está habilitado
            if self.use_ml:
                score = min(1.0, score + 0.15)
            
            results.append((product["id"], score))
        
        # Ordenar y devolver top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in results[:top_k]]
    
    def simulate_hybrid_retrieval(self, query, user_id="test_user", top_k=5):
        """Simula recuperación híbrida."""
        # Primero obtener resultados RAG
        rag_results = self.simulate_rag_retrieval(query, top_k=10)
        
        if not rag_results:
            return []
        
        # Simular scores colaborativos
        collab_scores = {}
        for i, pid in enumerate(rag_results):
            # Score base basado en posición
            base_score = 0.7 - (i * 0.05)
            
            # Factor usuario
            user_hash = hash(user_id) % 100 / 100.0
            user_factor = 0.2 * user_hash
            
            # Factor ML
            ml_factor = 0.15 if self.use_ml else 0.0
            
            collab_scores[pid] = min(0.95, base_score + user_factor + ml_factor)
        
        # Combinar scores
        combined_scores = {}
        for i, pid in enumerate(rag_results):
            rag_score = 1.0 - (i * 0.1)  # Score RAG basado en posición
            collab_score = collab_scores.get(pid, 0.3)
            
            if self.use_ml:
                # 60% RAG, 40% colaborativo con ML
                combined = (rag_score * 0.6) + (collab_score * 0.4)
            else:
                # 70% RAG, 30% colaborativo sin ML
                combined = (rag_score * 0.7) + (collab_score * 0.3)
            
            combined_scores[pid] = combined
        
        # Ordenar y devolver top_k
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in sorted_items[:top_k]]
    
    def evaluate(self):
        """Ejecuta evaluación completa."""
        # Obtener consultas
        queries, ground_truths = self.build_test_queries()
        
        # Preparar para métricas
        all_retrieved = []
        start_time = time.time()
        
        # Ejecutar consultas
        for query in queries:
            if self.mode == "rag":
                retrieved = self.simulate_rag_retrieval(query, top_k=10)
            else:  # hybrid
                retrieved = self.simulate_hybrid_retrieval(query, top_k=5)
            all_retrieved.append(retrieved)
        
        elapsed_time = time.time() - start_time
        
        # Calcular métricas
        metrics = {
            "time_seconds": elapsed_time,
            "latency_per_query_ms": (elapsed_time / len(queries)) * 1000,
            "queries_count": len(queries),
            "precision@5": precision_at_k(all_retrieved, ground_truths, k=5),
            "recall@5": recall_at_k(all_retrieved, ground_truths, k=5),
            "f1@5": f1_score_at_k(all_retrieved, ground_truths, k=5),
            "hit_rate@5": hit_rate_at_k(all_retrieved, ground_truths, k=5),
            "map@5": map_at_k(all_retrieved, ground_truths, k=5),
            "coverage": coverage(all_retrieved, self.all_product_ids),
        }
        
        # Añadir métricas específicas
        if self.mode == "rag":
            metrics["avg_retrieved"] = 10.0
        else:
            metrics["avg_retrieved"] = 5.0
        
        metrics["config"] = {
            "mode": self.mode,
            "ml_enabled": self.use_ml
        }
        
        return metrics

def run_comparative_analysis():
    """Ejecuta análisis comparativo completo."""
    print("\n" + "="*80)
    print("📊 ANÁLISIS COMPARATIVO - PRE vs POST ENTRENAMIENTO")
    print("="*80)
    
    # Configuraciones a evaluar
    configs = [
        ("rag", False, "RAG sin ML"),
        ("rag", True, "RAG con ML"),
        ("hybrid", False, "Híbrido sin ML"),
        ("hybrid", True, "Híbrido con ML"),
    ]
    
    results = {}
    
    for mode, use_ml, name in configs:
        system = ConsistentTestSystem(mode=mode, use_ml=use_ml)
        metrics = system.evaluate()
        results[name] = metrics
        
        print(f"\n🔧 {name}:")
        print(f"   Precision@5: {metrics['precision@5']:.3f}")
        print(f"   Recall@5: {metrics['recall@5']:.3f}")
        print(f"   F1@5: {metrics['f1@5']:.3f}")
        print(f"   Hit Rate@5: {metrics['hit_rate@5']:.3f}")
        print(f"   MAP@5: {metrics['map@5']:.3f}")
        print(f"   Coverage: {metrics['coverage']:.3f}")
        print(f"   Latencia: {metrics['latency_per_query_ms']:.1f}ms")
    
    # Análisis de mejora
    print("\n" + "="*80)
    print("📈 RESUMEN DE MEJORAS")
    print("="*80)
    
    # Calcular promedios
    avg_precision = mean([m['precision@5'] for m in results.values()])
    avg_recall = mean([m['recall@5'] for m in results.values()])
    avg_f1 = mean([m['f1@5'] for m in results.values()])
    
    print(f"\n📊 Promedio de todas las configuraciones:")
    print(f"   Precision@5 promedio: {avg_precision:.3f}")
    print(f"   Recall@5 promedio: {avg_recall:.3f}")
    print(f"   F1@5 promedio: {avg_f1:.3f}")
    
    # Comparar con tus resultados anteriores
    print(f"\n🔍 Comparación con tus resultados:")
    print(f"   PRE-entrenamiento (old): Precision=0.714, Recall=0.091, F1=0.161")
    print(f"   POST-entrenamiento (old): Precision=0.486, Recall=0.411, F1=0.445")
    print(f"   NUEVO sistema: Precision={avg_precision:.3f}, Recall={avg_recall:.3f}, F1={avg_f1:.3f}")
    
    # Guardar resultados
    output_data = {
        "timestamp": time.time(),
        "analysis": "Evaluación consistente corregida",
        "results": results,
        "averages": {
            "precision@5": avg_precision,
            "recall@5": avg_recall,
            "f1@5": avg_f1,
        }
    }
    
    with open("evaluation_consistent.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: evaluation_consistent.json")
    print("="*80)

def analyze_your_data():
    """Analiza tus datos existentes para identificar problemas."""
    print("\n" + "="*80)
    print("🔍 DIAGNÓSTICO DE TUS RESULTADOS EXISTENTES")
    print("="*80)
    
    # Datos de pre-entrenamiento
    pre_data = {
        "precision@5": 0.714,
        "recall@5": 0.091,
        "f1@5": 0.161
    }
    
    # Datos de post-entrenamiento (primer set)
    post_data1 = {
        "precision@5": 0.486,
        "recall@5": 0.411,
        "f1@5": 0.445
    }
    
    # Datos de post-entrenamiento (segundo set - con métricas nuevas)
    post_data2 = {
        "precision@5": 0.425,
        "recall@5": 0.359,
        "f1@5": 0.389
    }
    
    print(f"\n📉 PROBLEMAS IDENTIFICADOS:")
    print(f"1. Recall PRE-entrenamiento: {pre_data['recall@5']:.3f} (MUY BAJO)")
    print(f"2. Recall POST-entrenamiento: {post_data1['recall@5']:.3f} (4.5x mejor)")
    print(f"3. ¿Ground Truth diferente? Posiblemente SÍ")
    
    print(f"\n🎯 RECOMENDACIONES:")
    print(f"1. Revisar la función build_test_queries()")
    print(f"2. Asegurar consistencia en los productos de prueba")
    print(f"3. Verificar que los stubs generen resultados variables")
    print(f"4. Usar el sistema ConsistentTestSystem proporcionado")
    
    # Calcular mejoras
    recall_improvement = (post_data1['recall@5'] - pre_data['recall@5']) / pre_data['recall@5'] * 100
    f1_improvement = (post_data1['f1@5'] - pre_data['f1@5']) / pre_data['f1@5'] * 100
    
    print(f"\n📈 MEJORAS OBSERVADAS:")
    print(f"   Recall: +{recall_improvement:.0f}%")
    print(f"   F1 Score: +{f1_improvement:.0f}%")
    print("="*80)

if __name__ == "__main__":
    print("🚀 SISTEMA DE EVALUACIÓN CONSISTENTE")
    print("="*80)
    
    # 1. Primero analizar tus datos existentes
    analyze_your_data()
    
    # 2. Ejecutar evaluación corregida
    run_comparative_analysis()
    
    # 3. Instrucciones para uso
    print("\n📋 INSTRUCCIONES PARA USAR EL SISTEMA CORREGIDO:")
    print("1. Reemplaza tu función build_test_queries() por la versión consistente")
    print("2. Asegúrate de usar los mismos productos en todas las evaluaciones")
    print("3. Ejecuta: python deepeval_fixed.py")
    print("4. Compara los nuevos resultados con los anteriores")
    print("="*80)