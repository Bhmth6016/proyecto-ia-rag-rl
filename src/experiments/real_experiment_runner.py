# src/experiments/real_experiment_runner.py
"""
Ejecuta experimentos REALES con métricas REALES - VERSIÓN CORREGIDA
"""
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# Importación corregida
from src.data.interaction_logger import RealInteractionLogger
from src.evaluation.real_evaluator import RealEvaluator

logger = logging.getLogger(__name__)

class RealExperimentRunner:
    """Ejecuta experimentos REALES - VERSIÓN CORREGIDA"""
    
    def __init__(self, system):
        self.system = system
        self.interaction_logger = RealInteractionLogger()
        self.evaluator = RealEvaluator(self.interaction_logger)
        self.results_dir = Path("results/real_experiments")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🔬 RealExperimentRunner inicializado (CORREGIDO)")
    
    def run_real_experiment(self, experiment_name: str, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta un experimento REAL - VERSIÓN CORREGIDA"""
        logger.info(f"\n🔬 EJECUTANDO EXPERIMENTO REAL: {experiment_name}")
        logger.info(f"   Configuración: {experiment_config}")
        
        # Queries de prueba REALES
        test_queries = [
            "car parts",
            "led lights for cars",
            "wireless headphones",
            "laptop for programming", 
            "kitchen blender",
            "running shoes",
            "skin care products",
            "men's jeans",
            "smartphone case",
            "office chair"
        ]
        
        # Determinar modo del experimento
        mode_mapping = {
            "baseline_real": "baseline",
            "with_features_real": "with_features",
            "with_rlhf_real": "with_rlhf"
        }
        
        mode = mode_mapping.get(experiment_name, "baseline")
        
        experiment_results = {
            "experiment_name": experiment_name,
            "mode": mode,
            "description": experiment_config.get("description", ""),
            "config": experiment_config,
            "timestamp": datetime.now().isoformat(),
            "query_results": {},
            "overall_metrics": {}
        }
        
        all_metrics = []
        
        for query in test_queries:
            try:
                logger.info(f"\n   🔍 Procesando query REAL: '{query}' (modo: {mode})")
                
                # Procesar query con el modo correcto
                start_time = time.time()
                response = self.system._process_query_mode(query, mode)
                query_time_ms = (time.time() - start_time) * 1000
                
                if response.get("success"):
                    # Extraer IDs de productos rankeados
                    ranked_products = [p.get("id") for p in response["products"]]
                    
                    # Calcular métricas REALES usando el evaluador
                    metrics = self.evaluator.evaluate_ranking(
                        query=query,
                        ranked_products=ranked_products,
                        mode=mode,
                        k=5
                    )
                    
                    # Añadir tiempo de query
                    metrics["query_time_ms"] = query_time_ms
                    
                    # Guardar resultados
                    experiment_results["query_results"][query] = {
                        "query": query,
                        "mode": mode,
                        "ranked_products_count": len(ranked_products),
                        **metrics
                    }
                    
                    if metrics.get("has_ground_truth", False):
                        all_metrics.append(metrics)
                    
                    logger.info(f"   ✅ Query procesada - NDCG@5: {metrics.get('ndcg@k', 0):.3f}")
                    
                    # Registrar interacción para ground truth futuro
                    self.interaction_logger.log_interaction(
                        session_id=f"exp_{experiment_name}",
                        mode=mode,
                        query=query,
                        results=response.get("products", []),
                        feedback_type="shown",
                        additional_context={
                            "experiment": experiment_name,
                            "has_ground_truth": metrics.get("has_ground_truth", False)
                        }
                    )
                    
                    # Simular feedback para RLHF si es necesario
                    if mode == "with_rlhf" and metrics.get("has_ground_truth", False):
                        self._simulate_feedback_for_rl(query, ranked_products[:3], mode)
                    
                else:
                    logger.error(f"   ❌ Error procesando query: {response.get('error')}")
                    
            except Exception as e:
                logger.error(f"   ❌ Error en query '{query}': {e}")
        
        # Calcular métricas globales
        if all_metrics:
            self._calculate_overall_metrics(all_metrics, experiment_results)
        
        # Guardar resultados
        self._save_experiment_results(experiment_name, experiment_results)
        
        logger.info(f"\n✅ Experimento '{experiment_name}' COMPLETADO")
        logger.info(f"   Queries procesadas: {len(experiment_results['query_results'])}")
        logger.info(f"   Queries con ground truth: {len(all_metrics)}")
        
        if experiment_results.get("overall_metrics", {}).get("ndcg@k_mean"):
            logger.info(f"   NDCG@5 promedio: {experiment_results['overall_metrics']['ndcg@k_mean']:.3f}")
        
        return experiment_results
    
    def _calculate_overall_metrics(self, all_metrics, experiment_results):
        """Calcula métricas globales"""
        metric_names = ['ndcg@k', 'mrr', 'precision@k', 'recall@k']
        overall_metrics = {}
        
        for metric_name in metric_names:
            values = [m.get(metric_name, 0) for m in all_metrics]
            if values:
                overall_metrics[f"{metric_name}_mean"] = float(np.mean(values))
                overall_metrics[f"{metric_name}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
                overall_metrics[f"{metric_name}_min"] = float(np.min(values))
                overall_metrics[f"{metric_name}_max"] = float(np.max(values))
        
        # Añadir estadísticas generales
        overall_metrics["total_queries"] = len(all_metrics)
        overall_metrics["queries_with_ground_truth"] = len(all_metrics)
        
        experiment_results["overall_metrics"] = overall_metrics
    
    def _simulate_feedback_for_rl(self, query: str, top_products: List[str], mode: str):
        """Simula feedback para RLHF basado en interacciones previas"""
        if not top_products or mode != "with_rlhf":
            return
        
        # Verificar si hay RL ranker
        if not hasattr(self.system, 'rl_ranker') or not self.system.rl_ranker:
            return
        
        try:
            # Obtener productos relevantes
            relevant_products = self.evaluator._get_relevant_products_for_query(query)
            
            # Simular feedback para cada producto top
            for position, product_id in enumerate(top_products[:3], 1):
                is_relevant = product_id in relevant_products
                reward = 1.0 if is_relevant else -0.2
                
                # Aprender del feedback
                self.system.rl_ranker.learn_from_feedback(
                    query_features={"query_text": query},
                    selected_product_id=product_id,
                    reward=reward,
                    context={
                        "query": query,
                        "position": position,
                        "mode": mode,
                        "is_relevant": is_relevant,
                        "is_simulated": True
                    }
                )
                
                logger.debug(f"   📝 Feedback simulado RL: {product_id} -> "
                           f"{'RELEVANTE' if is_relevant else 'NO RELEVANTE'} "
                           f"(reward={reward:.2f})")
                
        except Exception as e:
            logger.debug(f"   ⚠️  Error en feedback RL: {e}")
    
    def _save_experiment_results(self, experiment_name: str, results: Dict[str, Any]):
        """Guarda resultados del experimento"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"real_experiment_{experiment_name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Resultados guardados: {filepath}")
    
    def run_all_real_experiments(self) -> Dict[str, Any]:
        """Ejecuta todos los experimentos REALES"""
        logger.info("\n" + "="*80)
        logger.info("🚀 EJECUTANDO EXPERIMENTOS REALES COMPLETOS")
        logger.info("="*80)
        
        # Configuraciones de experimentos
        experiment_configs = {
            "baseline_real": {
                "description": "Solo FAISS + similitud coseno (REAL)",
                "use_rlhf": False,
                "use_features": False
            },
            "with_features_real": {
                "description": "FAISS + características heurísticas (REAL)",
                "use_rlhf": False,
                "use_features": True
            },
            "with_rlhf_real": {
                "description": "FAISS + características + RLHF (REAL)",
                "use_rlhf": True,
                "use_features": True
            }
        }
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "experiments": {}
        }
        
        for exp_name, exp_config in experiment_configs.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 EJECUTANDO: {exp_name}")
            logger.info(f"{'='*60}")
            
            results = self.run_real_experiment(exp_name, exp_config)
            all_results["experiments"][exp_name] = results
            
            # Pequeña pausa entre experimentos
            time.sleep(1)
        
        # Calcular comparativa
        comparison = self._calculate_experiment_comparison(all_results)
        all_results["comparison"] = comparison
        
        # Guardar resultados combinados
        combined_filename = f"real_experiments_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        combined_filepath = self.results_dir / combined_filename
        
        with open(combined_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n💾 Resultados combinados guardados: {combined_filepath}")
        
        # Mostrar resumen comparativo
        self._print_comparison_summary(comparison)
        
        return all_results
    
    def _calculate_experiment_comparison(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula comparativa entre experimentos"""
        comparison = {}
        
        for exp_name, exp_data in all_results["experiments"].items():
            if "overall_metrics" in exp_data:
                metrics = exp_data["overall_metrics"]
                
                comparison[exp_name] = {
                    "ndcg@5": metrics.get("ndcg@k_mean", 0),
                    "mrr": metrics.get("mrr_mean", 0),
                    "precision@5": metrics.get("precision@k_mean", 0),
                    "recall@5": metrics.get("recall@k_mean", 0),
                    "query_count": len(exp_data.get("query_results", {})),
                    "description": exp_data.get("description", ""),
                    "mode": exp_data.get("mode", "")
                }
        
        return comparison
    
    def _print_comparison_summary(self, comparison: Dict[str, Any]):
        """Imprime resumen comparativo de experimentos"""
        logger.info("\n" + "="*80)
        logger.info("📊 RESUMEN COMPARATIVO DE EXPERIMENTOS")
        logger.info("="*80)
        
        # Encabezado de tabla
        header = f"{'Experimento':<25} | {'NDCG@5':<8} | {'MRR':<8} | {'P@5':<8} | {'R@5':<8} | {'Queries':<8}"
        logger.info(header)
        logger.info("-" * len(header))
        
        # Filas de datos
        for exp_name, exp_data in comparison.items():
            row = f"{exp_name:<25} | " \
                  f"{exp_data['ndcg@5']:.4f} | " \
                  f"{exp_data['mrr']:.4f} | " \
                  f"{exp_data['precision@5']:.4f} | " \
                  f"{exp_data['recall@5']:.4f} | " \
                  f"{exp_data['query_count']:<8}"
            logger.info(row)
        
        # Encontrar mejor experimento por NDCG@5
        if comparison:
            best_exp = max(comparison.items(), key=lambda x: x[1]['ndcg@5'])
            logger.info("\n" + "="*80)
            logger.info(f"🏆 MEJOR EXPERIMENTO: {best_exp[0]}")
            logger.info(f"   NDCG@5: {best_exp[1]['ndcg@5']:.4f}")
            logger.info(f"   Descripción: {best_exp[1]['description']}")
            logger.info("="*80)
    
    def run_single_query_experiment(self, query: str, mode: str = "baseline") -> Dict[str, Any]:
        """Ejecuta experimento para una sola query"""
        logger.info(f"\n🔍 Ejecutando experimento para query única: '{query}' (modo: {mode})")
        
        # Procesar query
        start_time = time.time()
        response = self.system._process_query_mode(query, mode)
        query_time_ms = (time.time() - start_time) * 1000
        
        if not response.get("success"):
            return {"error": response.get("error")}
        
        # Extraer productos rankeados
        ranked_products = [p.get("id") for p in response.get("products", [])]
        
        # Calcular métricas
        metrics = self.evaluator.evaluate_ranking(
            query=query,
            ranked_products=ranked_products,
            mode=mode,
            k=5
        )
        
        # Añadir tiempo de query
        metrics["query_time_ms"] = query_time_ms
        
        # Registrar interacción
        self.interaction_logger.log_interaction(
            session_id=f"single_query_exp",
            mode=mode,
            query=query,
            results=response.get("products", []),
            feedback_type="shown",
            additional_context={
                "is_single_query_experiment": True,
                "has_ground_truth": metrics.get("has_ground_truth", False)
            }
        )
        
        # Simular feedback para RLHF si es necesario
        if mode == "with_rlhf" and metrics.get("has_ground_truth", False):
            self._simulate_feedback_for_rl(query, ranked_products[:3], mode)
        
        return {
            "query": query,
            "mode": mode,
            "ranked_products": ranked_products,
            "ranked_count": len(ranked_products),
            "metrics": metrics,
            "response_summary": {
                "success": True,
                "query_time_ms": query_time_ms
            }
        }