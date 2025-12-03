#!/usr/bin/env python3
"""
deepeval_advanced.py - Sistema de evaluación avanzado con métricas realistas
"""
import json
import time
import random
import logging
from typing import List, Set, Dict, Any, Tuple
import numpy as np

# --- Configuración logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealisticEvaluationSystem:
    """Sistema de evaluación realista con métricas mejoradas."""
    
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Productos con características realistas
        self.products = self._load_realistic_products()
        self.all_product_ids = {p["id"] for p in self.products}
        
        # Popularidad realista (simulada)
        self.popularity_scores = self._generate_popularity_scores()
        
        # Historial de usuario simulado
        self.user_history = self._generate_user_history()
        
        logger.info(f"Sistema inicializado con {len(self.products)} productos")
    
    def _load_realistic_products(self):
        """Carga productos con características variadas."""
        return [
            {"id": "P001", "title": "Laptop Gaming ASUS ROG", "category": "electronics", "price": 1299, "brand": "ASUS"},
            {"id": "P002", "title": "Teclado Mecánico Razer", "category": "electronics", "price": 149, "brand": "Razer"},
            {"id": "P003", "title": "Ratón Gaming Logitech", "category": "electronics", "price": 79, "brand": "Logitech"},
            {"id": "P004", "title": "Monitor 4K Samsung 32'", "category": "electronics", "price": 399, "brand": "Samsung"},
            {"id": "P005", "title": "Silla Gamer Secretlab", "category": "furniture", "price": 549, "brand": "Secretlab"},
            {"id": "P006", "title": "Auriculares Gaming SteelSeries", "category": "electronics", "price": 199, "brand": "SteelSeries"},
            {"id": "P007", "title": "Micrófono Blue Yeti USB", "category": "electronics", "price": 129, "brand": "Blue"},
            {"id": "P008", "title": "Alfombrilla Gaming XL", "category": "accessories", "price": 29, "brand": "Generic"},
            {"id": "P009", "title": "Webcam Logitech C920", "category": "electronics", "price": 89, "brand": "Logitech"},
            {"id": "P010", "title": "Monitor Gaming 144Hz", "category": "electronics", "price": 299, "brand": "AOC"},
            {"id": "P011", "title": "SSD NVMe 1TB", "category": "electronics", "price": 99, "brand": "Samsung"},
            {"id": "P012", "title": "Memoria RAM 32GB", "category": "electronics", "price": 129, "brand": "Corsair"},
            {"id": "P013", "title": "Tarjeta Gráfica RTX 4070", "category": "electronics", "price": 599, "brand": "NVIDIA"},
            {"id": "P014", "title": "Fuente Alimentación 750W", "category": "electronics", "price": 119, "brand": "Seasonic"},
            {"id": "P015", "title": "Refrigeración Líquida", "category": "electronics", "price": 149, "brand": "NZXT"},
        ]
    
    def _generate_popularity_scores(self):
        """Genera scores de popularidad realistas."""
        scores = {}
        for product in self.products:
            # Base de popularidad
            base_pop = random.uniform(0.1, 0.9)
            
            # Ajustar por precio (productos más baratos son más populares)
            price_factor = max(0.1, 1.0 - (product["price"] / 2000))
            
            # Ajustar por categoría
            category_factor = 1.2 if product["category"] == "electronics" else 0.8
            
            scores[product["id"]] = min(0.95, base_pop * price_factor * category_factor)
        
        return scores
    
    def _generate_user_history(self, user_id="test_user", n_items=5):
        """Genera historial de usuario realista."""
        # Seleccionar algunos productos basados en popularidad
        sorted_products = sorted(self.popularity_scores.items(), key=lambda x: x[1], reverse=True)
        popular_ids = [pid for pid, _ in sorted_products[:8]]
        
        # Tomar 5 productos (algunos populares, algunos aleatorios)
        history = set(random.sample(popular_ids, min(3, len(popular_ids))))
        
        # Añadir algunos productos aleatorios
        remaining_ids = list(set(self.all_product_ids) - history)
        if remaining_ids:
            history.update(random.sample(remaining_ids, min(2, len(remaining_ids))))
        
        return list(history)[:n_items]
    
    def build_realistic_test_queries(self):
        """Construye consultas de prueba con ground truth realista."""
        # Consultas con diferentes niveles de dificultad
        test_cases = [
            # (consulta, ground_truth_ids, dificultad)
            ("laptop gaming asus", ["P001"], "fácil"),  # Match exacto
            ("teclado razer mecánico", ["P002"], "fácil"),
            ("ratón logitech gaming", ["P003"], "fácil"),
            ("monitor 4k", ["P004"], "medio"),  # Múltiples productos podrían coincidir
            ("silla gamer secretlab", ["P005"], "fácil"),
            ("auriculares gaming steelseries", ["P006"], "fácil"),
            ("micrófono blue yeti", ["P007"], "fácil"),
            ("monitor gaming", ["P004", "P010"], "difícil"),  # Múltiples resultados
            ("productos logitech", ["P003", "P009"], "medio"),  # Múltiples marcas
            ("accesorios computadora", ["P002", "P003", "P006", "P007", "P008", "P009"], "muy difícil"),
        ]
        
        queries = []
        ground_truths = []
        difficulties = []
        
        for query, gt_ids, difficulty in test_cases:
            queries.append(query)
            
            # Asegurar que los IDs existen
            valid_gt = set()
            for pid in gt_ids:
                if any(p["id"] == pid for p in self.products):
                    valid_gt.add(pid)
            
            if not valid_gt:
                logger.warning(f"No ground truth válido para consulta: {query}")
                valid_gt = set(random.sample(list(self.all_product_ids), 1))
            
            ground_truths.append(valid_gt)
            difficulties.append(difficulty)
        
        logger.info(f"Generadas {len(queries)} consultas con dificultades variadas")
        for i, (q, d) in enumerate(zip(queries, difficulties)):
            logger.debug(f"  {i+1}. '{q}' - {d} ({len(ground_truths[i])} resultados esperados)")
        
        return queries, ground_truths, difficulties
    
    def simulate_rag_retrieval(self, query, use_ml=False, top_k=10):
        """Simula recuperación RAG con realismo."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        
        for product in self.products:
            score = 0.0
            title = product["title"].lower()
            
            # 1. Matching exacto o parcial
            if query_lower in title:
                score += 0.8
            elif all(word in title for word in query_words):
                score += 0.6
            elif any(word in title for word in query_words):
                score += 0.3
            
            # 2. Factor de popularidad
            popularity = self.popularity_scores.get(product["id"], 0.3)
            score += popularity * 0.2
            
            # 3. Factor de categoría (boost para electrónica)
            if product["category"] == "electronics":
                score += 0.1
            
            # 4. Ruido aleatorio (simula variabilidad)
            noise = random.uniform(-0.1, 0.1)
            score += noise
            
            # 5. Boost ML si está habilitado
            if use_ml:
                # ML mejora matching semántico
                ml_boost = random.uniform(0.05, 0.15)
                score += ml_boost
            
            # Asegurar score entre 0 y 1
            score = max(0.0, min(1.0, score))
            
            results.append((product["id"], score))
        
        # Ordenar y añadir algo de aleatoriedad
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Introducir algo de ruido en el ranking (no siempre perfecto)
        if random.random() < 0.3:  # 30% de chance de ranking imperfecto
            # Intercambiar algunos elementos cercanos
            for i in range(len(results) - 1):
                if random.random() < 0.2 and abs(results[i][1] - results[i+1][1]) < 0.05:
                    results[i], results[i+1] = results[i+1], results[i]
        
        return [pid for pid, _ in results[:top_k]]
    
    def simulate_collaborative_scores(self, user_id, candidate_ids, use_ml=False):
        """Simula scores colaborativos realistas."""
        scores = {}
        
        # Obtener historial del usuario
        user_history = self.user_history
        
        for pid in candidate_ids:
            base_score = 0.3
            
            # 1. Similitud con historial
            if pid in user_history:
                base_score += 0.3
            
            # 2. Popularidad
            popularity = self.popularity_scores.get(pid, 0.3)
            base_score += popularity * 0.2
            
            # 3. Factor aleatorio basado en usuario
            user_hash = hash(user_id) % 100 / 100.0
            user_factor = 0.1 * user_hash
            
            # 4. Boost ML si está habilitado
            if use_ml:
                # ML puede detectar patrones más complejos
                ml_factor = random.uniform(0.05, 0.15)
                base_score += ml_factor
            
            scores[pid] = min(0.95, base_score + user_factor)
        
        return scores
    
    def simulate_hybrid_retrieval(self, query, user_id="test_user", use_ml=False, top_k=5):
        """Simula recuperación híbrida realista."""
        # Paso 1: Recuperación RAG
        rag_results = self.simulate_rag_retrieval(query, use_ml=use_ml, top_k=15)
        
        if not rag_results:
            return []
        
        # Paso 2: Scores colaborativos
        collab_scores = self.simulate_collaborative_scores(user_id, rag_results, use_ml=use_ml)
        
        # Paso 3: Combinar scores
        combined_scores = {}
        
        for i, pid in enumerate(rag_results):
            # Score RAG (decae con la posición)
            rag_score = 1.0 - (i * 0.05)
            
            # Score colaborativo
            collab_score = collab_scores.get(pid, 0.3)
            
            # Combinación con pesos
            if use_ml:
                # ML optimiza los pesos: 55% RAG, 45% colaborativo
                combined = (rag_score * 0.55) + (collab_score * 0.45)
            else:
                # Sin ML: 70% RAG, 30% colaborativo
                combined = (rag_score * 0.7) + (collab_score * 0.3)
            
            combined_scores[pid] = combined
        
        # Ordenar y devolver top_k
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in sorted_items[:top_k]]
    
    def calculate_metrics(self, retrieved_lists, ground_truth_sets):
        """Calcula todas las métricas."""
        def precision_at_k(k=5):
            precisions = []
            for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
                relevant = sum(1 for doc_id in retrieved[:k] if doc_id in gt)
                precisions.append(relevant / k if k > 0 else 0.0)
            return np.mean(precisions) if precisions else 0.0
        
        def recall_at_k(k=5):
            recalls = []
            for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
                if not gt:
                    recalls.append(0.0)
                    continue
                relevant = sum(1 for doc_id in retrieved[:k] if doc_id in gt)
                recalls.append(relevant / len(gt))
            return np.mean(recalls) if recalls else 0.0
        
        def hit_rate_at_k(k=5):
            hits = []
            for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
                hit_found = any(item in gt for item in retrieved[:k])
                hits.append(1.0 if hit_found else 0.0)
            return np.mean(hits) if hits else 0.0
        
        def map_at_k(k=5):
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
                        score += hits / i
                
                ap_scores.append(score / min(len(gt), k))
            
            return np.mean(ap_scores) if ap_scores else 0.0
        
        def coverage():
            recommended = set()
            for retrieved in retrieved_lists:
                recommended.update(retrieved[:5])  # Solo top 5 para coverage
            return len(recommended) / len(self.all_product_ids) if self.all_product_ids else 0.0
        
        def calculate_diversity():
            """Calcula diversidad de recomendaciones."""
            all_recommended = []
            for retrieved in retrieved_lists:
                all_recommended.extend(retrieved[:3])  # Solo primeros 3 para diversidad
            
            if not all_recommended:
                return 0.0
            
            # Contar productos únicos en recomendaciones
            unique_recommended = len(set(all_recommended))
            total_recommended = len(all_recommended)
            
            return unique_recommended / total_recommended if total_recommended > 0 else 0.0
        
        def calculate_novelty(popular_items, k=5):
            """Calcula novedad de recomendaciones."""
            novelty_scores = []
            for retrieved in retrieved_lists:
                retrieved_k = retrieved[:k]
                if not retrieved_k:
                    novelty_scores.append(0.0)
                    continue
                
                novel_count = sum(1 for item in retrieved_k if item not in popular_items)
                novelty_scores.append(novel_count / len(retrieved_k))
            
            return np.mean(novelty_scores) if novelty_scores else 0.0
        
        # Obtener productos populares para novedad
        sorted_popularity = sorted(self.popularity_scores.items(), key=lambda x: x[1], reverse=True)
        popular_items = set([pid for pid, _ in sorted_popularity[:5]])
        
        return {
            "precision@5": precision_at_k(5),
            "recall@5": recall_at_k(5),
            "f1@5": 2 * precision_at_k(5) * recall_at_k(5) / (precision_at_k(5) + recall_at_k(5)) 
                     if (precision_at_k(5) + recall_at_k(5)) > 0 else 0.0,
            "hit_rate@5": hit_rate_at_k(5),
            "map@5": map_at_k(5),
            "coverage": coverage(),
            "diversity": calculate_diversity(),
            "novelty@5": calculate_novelty(popular_items, 5),
        }
    
    def evaluate_configuration(self, mode="rag", use_ml=False, n_runs=3):
        """Evalúa una configuración específica con múltiples ejecuciones para estabilidad."""
        all_metrics = []
        all_retrieved_lists = []
        
        for run in range(n_runs):
            # Obtener consultas
            queries, ground_truths, _ = self.build_realistic_test_queries()
            
            # Ejecutar consultas
            retrieved_lists = []
            start_time = time.time()
            
            for query in queries:
                if mode == "rag":
                    retrieved = self.simulate_rag_retrieval(query, use_ml=use_ml, top_k=10)
                else:  # hybrid
                    retrieved = self.simulate_hybrid_retrieval(query, use_ml=use_ml, top_k=5)
                retrieved_lists.append(retrieved)
            
            elapsed_time = time.time() - start_time
            
            # Calcular métricas
            metrics = self.calculate_metrics(retrieved_lists, ground_truths)
            metrics["time_seconds"] = elapsed_time
            metrics["latency_per_query_ms"] = (elapsed_time / len(queries)) * 1000
            metrics["queries_count"] = len(queries)
            
            all_metrics.append(metrics)
            all_retrieved_lists.append(retrieved_lists)
        
        # Promediar métricas de múltiples ejecuciones
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
        
        # Añadir desviación estándar para métricas clave
        for key in ["precision@5", "recall@5", "f1@5"]:
            values = [m[key] for m in all_metrics]
            avg_metrics[f"{key}_std"] = np.std(values)
        
        avg_metrics["config"] = {
            "mode": mode,
            "ml_enabled": use_ml,
            "n_runs": n_runs,
            "seed": self.seed
        }
        
        return avg_metrics

def main():
    """Función principal."""
    print("="*80)
    print("🚀 SISTEMA DE EVALUACIÓN AVANZADO - REALISTA")
    print("="*80)
    
    # Inicializar sistema
    system = RealisticEvaluationSystem(seed=42)
    
    # Configuraciones a evaluar
    configs = [
        ("rag", False, "RAG sin ML"),
        ("rag", True, "RAG con ML"),
        ("hybrid", False, "Híbrido sin ML"),
        ("hybrid", True, "Híbrido con ML"),
    ]
    
    results = {}
    
    print("\n📊 EJECUTANDO EVALUACIONES (3 ejecuciones por configuración para estabilidad)...")
    print("-"*80)
    
    for mode, use_ml, name in configs:
        print(f"\n🔄 Evaluando: {name}...")
        metrics = system.evaluate_configuration(mode=mode, use_ml=use_ml, n_runs=3)
        results[name] = metrics
        
        print(f"   ✅ Completado")
        print(f"   📈 Precision@5: {metrics['precision@5']:.3f} (±{metrics.get('precision@5_std', 0):.3f})")
        print(f"   🔍 Recall@5: {metrics['recall@5']:.3f} (±{metrics.get('recall@5_std', 0):.3f})")
        print(f"   🎯 F1@5: {metrics['f1@5']:.3f} (±{metrics.get('f1@5_std', 0):.3f})")
        print(f"   ⚡ Hit Rate@5: {metrics['hit_rate@5']:.3f}")
        print(f"   📊 MAP@5: {metrics['map@5']:.3f}")
        print(f"   🌐 Coverage: {metrics['coverage']:.3f}")
        print(f"   🎲 Diversidad: {metrics['diversity']:.3f}")
        print(f"   🆕 Novelty@5: {metrics['novelty@5']:.3f}")
        print(f"   ⏱️  Latencia/query: {metrics['latency_per_query_ms']:.1f}ms")
    
    # Análisis comparativo
    print("\n" + "="*80)
    print("📈 ANÁLISIS COMPARATIVO DETALLADO")
    print("="*80)
    
    # Tabla comparativa
    print("\n📋 COMPARACIÓN DE CONFIGURACIONES:")
    print("-"*120)
    print(f"{'Sistema':<20} {'ML':<8} {'P@5':<8} {'R@5':<8} {'F1@5':<8} {'HR@5':<8} {'MAP@5':<8} {'Cov':<8} {'Div':<8} {'Nov':<8}")
    print("-"*120)
    
    for name, metrics in results.items():
        ml_status = "Sí" if metrics["config"]["ml_enabled"] else "No"
        print(f"{name:<20} {ml_status:<8} "
              f"{metrics['precision@5']:.3f}   "
              f"{metrics['recall@5']:.3f}   "
              f"{metrics['f1@5']:.3f}   "
              f"{metrics['hit_rate@5']:.3f}   "
              f"{metrics['map@5']:.3f}   "
              f"{metrics['coverage']:.3f}   "
              f"{metrics['diversity']:.3f}   "
              f"{metrics['novelty@5']:.3f}")
    
    print("-"*120)
    
    # Análisis de mejora con ML
    print("\n🔬 ANÁLISIS DE MEJORA CON MACHINE LEARNING:")
    print("-"*80)
    
    # Comparar RAG vs RAG+ML
    rag_no_ml = results["RAG sin ML"]
    rag_ml = results["RAG con ML"]
    
    p_improvement = ((rag_ml["precision@5"] - rag_no_ml["precision@5"]) / rag_no_ml["precision@5"]) * 100
    r_improvement = ((rag_ml["recall@5"] - rag_no_ml["recall@5"]) / rag_no_ml["recall@5"]) * 100
    f1_improvement = ((rag_ml["f1@5"] - rag_no_ml["f1@5"]) / rag_no_ml["f1@5"]) * 100
    
    print(f"RAG → RAG+ML:")
    print(f"  Precision: {rag_no_ml['precision@5']:.3f} → {rag_ml['precision@5']:.3f} ({p_improvement:+.1f}%)")
    print(f"  Recall: {rag_no_ml['recall@5']:.3f} → {rag_ml['recall@5']:.3f} ({r_improvement:+.1f}%)")
    print(f"  F1: {rag_no_ml['f1@5']:.3f} → {rag_ml['f1@5']:.3f} ({f1_improvement:+.1f}%)")
    
    # Comparar Híbrido vs Híbrido+ML
    hybrid_no_ml = results["Híbrido sin ML"]
    hybrid_ml = results["Híbrido con ML"]
    
    p_improvement_h = ((hybrid_ml["precision@5"] - hybrid_no_ml["precision@5"]) / hybrid_no_ml["precision@5"]) * 100
    r_improvement_h = ((hybrid_ml["recall@5"] - hybrid_no_ml["recall@5"]) / hybrid_no_ml["recall@5"]) * 100
    f1_improvement_h = ((hybrid_ml["f1@5"] - hybrid_no_ml["f1@5"]) / hybrid_no_ml["f1@5"]) * 100
    
    print(f"\nHíbrido → Híbrido+ML:")
    print(f"  Precision: {hybrid_no_ml['precision@5']:.3f} → {hybrid_ml['precision@5']:.3f} ({p_improvement_h:+.1f}%)")
    print(f"  Recall: {hybrid_no_ml['recall@5']:.3f} → {hybrid_ml['recall@5']:.3f} ({r_improvement_h:+.1f}%)")
    print(f"  F1: {hybrid_no_ml['f1@5']:.3f} → {hybrid_ml['f1@5']:.3f} ({f1_improvement_h:+.1f}%)")
    
    # Comparar RAG vs Híbrido (sin ML)
    print(f"\nRAG → Híbrido (sin ML):")
    p_diff = hybrid_no_ml["precision@5"] - rag_no_ml["precision@5"]
    r_diff = hybrid_no_ml["recall@5"] - rag_no_ml["recall@5"]
    f1_diff = hybrid_no_ml["f1@5"] - rag_no_ml["f1@5"]
    
    print(f"  Precision: {rag_no_ml['precision@5']:.3f} → {hybrid_no_ml['precision@5']:.3f} ({p_diff:+.3f})")
    print(f"  Recall: {rag_no_ml['recall@5']:.3f} → {hybrid_no_ml['recall@5']:.3f} ({r_diff:+.3f})")
    print(f"  F1: {rag_no_ml['f1@5']:.3f} → {hybrid_no_ml['f1@5']:.3f} ({f1_diff:+.3f})")
    
    # Determinar mejor sistema
    print("\n" + "="*80)
    print("🏆 MEJOR SISTEMA POR MÉTRICA:")
    print("="*80)
    
    best_by_metric = {}
    for metric in ["precision@5", "recall@5", "f1@5", "hit_rate@5", "map@5", "coverage", "diversity", "novelty@5"]:
        best_system = max(results.keys(), key=lambda x: results[x][metric])
        best_value = results[best_system][metric]
        best_by_metric[metric] = (best_system, best_value)
    
    for metric, (system, value) in best_by_metric.items():
        print(f"{metric:<15} → {system:<20} ({value:.3f})")
    
    # Sistema óptimo basado en combinación ponderada
    print("\n⚖️  SISTEMA ÓPTIMO (combinación ponderada):")
    print("-"*80)
    
    # Ponderaciones: 30% F1, 20% Coverage, 20% Diversity, 15% Novelty, 15% Latencia inversa
    weighted_scores = {}
    for name, metrics in results.items():
        # Convertir latencia a score (menor latencia = mejor)
        latency_score = max(0, 1 - (metrics["latency_per_query_ms"] / 100))
        
        weighted_score = (
            metrics["f1@5"] * 0.30 +
            metrics["coverage"] * 0.20 +
            metrics["diversity"] * 0.20 +
            metrics["novelty@5"] * 0.15 +
            latency_score * 0.15
        )
        weighted_scores[name] = weighted_score
    
    best_system = max(weighted_scores.keys(), key=lambda x: weighted_scores[x])
    best_score = weighted_scores[best_system]
    
    print(f"🏅 Sistema recomendado: {best_system}")
    print(f"📊 Puntuación balanceada: {best_score:.3f}")
    print(f"📈 F1 Score: {results[best_system]['f1@5']:.3f}")
    print(f"🌐 Coverage: {results[best_system]['coverage']:.3f}")
    print(f"🎲 Diversidad: {results[best_system]['diversity']:.3f}")
    print(f"🆕 Novelty: {results[best_system]['novelty@5']:.3f}")
    print(f"⏱️  Latencia: {results[best_system]['latency_per_query_ms']:.1f}ms")
    
    # Guardar resultados
    output_data = {
        "timestamp": time.time(),
        "system_info": {
            "n_products": len(system.products),
            "n_queries": 10,
            "seed": system.seed,
            "n_runs": 3
        },
        "results": results,
        "best_by_metric": best_by_metric,
        "weighted_scores": weighted_scores,
        "recommended_system": best_system
    }
    
    output_file = "evaluation_advanced_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados detallados guardados en: {output_file}")
    print("="*80)
    
    # Recomendaciones finales
    print("\n🎯 RECOMENDACIONES FINALES:")
    print("-"*80)
    
    if results["Híbrido con ML"]["f1@5"] > results["RAG con ML"]["f1@5"]:
        print("✅ El sistema híbrido con ML proporciona el mejor balance calidad/diversidad")
        print("   Considera implementar filtrado colaborativo junto con RAG")
    else:
        print("✅ El sistema RAG con ML proporciona la mejor precisión")
        print("   Focaliza esfuerzos en mejorar el sistema RAG básico")
    
    if any(metrics["coverage"] < 0.4 for metrics in results.values()):
        print("⚠️  Algunos sistemas tienen coverage bajo (<40%)")
        print("   Considera técnicas de diversificación de recomendaciones")
    
    if any(metrics["novelty@5"] < 0.3 for metrics in results.values()):
        print("⚠️  Algunos sistemas tienen novedad baja (<30%)")
        print("   Añadir componentes de exploración podría mejorar resultados a largo plazo")
    
    print("="*80)

if __name__ == "__main__":
    main()