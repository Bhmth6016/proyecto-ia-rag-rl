# src/evaluator/evaluator.py
"""
Evaluador científico para los 4 puntos del paper
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime
import json
import sys
import os

# Añadir src al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# Importaciones absolutas
try:
    from data.canonicalizer import CanonicalProduct
    from ranking.ranking_engine import StaticRankingEngine
    from ranking.rlhf_agent import RLHFAgent
    from features.features import StaticFeatures
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.canonicalizer import CanonicalProduct
    from ranking.ranking_engine import StaticRankingEngine
    from ranking.rlhf_agent import RLHFAgent
    from features.features import StaticFeatures

class ScientificEvaluator:
    """Evaluador para experimentos controlados"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.results = {}
        self.configurations = {
            1: {"name": "Baseline", "ml": False, "ner": False, "rlhf": False},
            2: {"name": "+NER", "ml": False, "ner": True, "rlhf": False},
            3: {"name": "+StaticML", "ml": True, "ner": False, "rlhf": False},
            4: {"name": "+RLHF", "ml": True, "ner": True, "rlhf": True}
        }
    
    def evaluate_configuration(
        self,
        config_id: int,
        query_embedding: np.ndarray,
        query_category: str,
        products: List[CanonicalProduct],
        rlhf_agent: Optional[RLHFAgent] = None
    ) -> Dict[str, Any]:
        """
        Evalúa una configuración específica
        
        Args:
            config_id: 1, 2, 3 o 4
            query_embedding: Embedding de la query
            query_category: Categoría de la query
            products: Productos candidatos
            rlhf_agent: Agente RLHF (solo para config 4)
        """
        config = self.configurations[config_id]
        logger.info(f"🧪 Evaluando Config {config_id}: {config['name']}")
        
        # 1. Baseline ranking (siempre presente)
        baseline_engine = StaticRankingEngine()
        baseline_results = baseline_engine.rank_products(
            query_embedding, query_category, products, top_k=10
        )
        
        if config_id == 1:
            # Solo baseline
            return {
                "config": config,
                "results": baseline_results,
                "ranking": [p.id for p in baseline_results],
                "scores": [1.0 / (i+1) for i in range(len(baseline_results))]  # Discounted scores
            }
        
        elif config_id == 2:
            # Baseline + NER (simulado)
            # Aquí podrías modificar el ranking basado en NER
            ner_boosted = self._apply_ner_boost(query_embedding, baseline_results)
            return {
                "config": config,
                "results": ner_boosted,
                "ranking": [p.id for p in ner_boosted],
                "scores": [1.0 / (i+1) for i in range(len(ner_boosted))]
            }
        
        elif config_id == 3:
            # Static ML features
            ml_engine = StaticRankingEngine(weights={
                "content_similarity": 0.3,
                "title_similarity": 0.2,
                "category_exact_match": 0.2,
                "rating_normalized": 0.15,
                "price_available": 0.1,
                "has_popularity": 0.05
            })
            ml_results = ml_engine.rank_products(
                query_embedding, query_category, products, top_k=10
            )
            return {
                "config": config,
                "results": ml_results,
                "ranking": [p.id for p in ml_results],
                "scores": [1.0 / (i+1) for i in range(len(ml_results))]
            }
        
        elif config_id == 4:
            # RLHF
            if not rlhf_agent:
                raise ValueError("RLHF agent required for configuration 4")
            
            # Extraer features de query para RLHF
            query_features = {"dummy": 1.0}  # Reemplazar con features reales
            
            # Obtener ranking RLHF
            baseline_indices = list(range(len(baseline_results)))
            rlhf_ranking = rlhf_agent.select_ranking(
                query_features, products, baseline_indices
            )
            
            rlhf_results = [products[idx] for idx in rlhf_ranking[:10]]
            
            return {
                "config": config,
                "results": rlhf_results,
                "ranking": [p.id for p in rlhf_results],
                "scores": rlhf_agent.get_learning_curve()[-10:] if rlhf_agent.get_learning_curve() else [0.5]*10
            }
    
    def _apply_ner_boost(self, query_embedding: np.ndarray, 
                        products: List[CanonicalProduct]) -> List[CanonicalProduct]:  # <-- CORREGIDO
        """Simula boost basado en NER"""
        # Implementación simple: boost productos con títulos más largos
        boosted = sorted(products, 
                        key=lambda p: len(p.title) * 0.1 + np.dot(query_embedding, p.content_embedding),
                        reverse=True)
        return boosted[:10]
    
    def compute_metrics(self, ranked_products: List[CanonicalProduct], 
                       relevance_scores: List[float]) -> Dict[str, float]:
        """Calcula métricas de ranking"""
        if not ranked_products or not relevance_scores:
            return {}
        
        # NDCG
        dcg = 0.0
        for i, score in enumerate(relevance_scores[:10]):
            dcg += score / np.log2(i + 2)
        
        # Ideal DCG
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_scores[:10]):
            idcg += score / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        # MRR
        mrr = 0.0
        for i, score in enumerate(relevance_scores[:10]):
            if score > 0.7:  # Umbral de relevancia
                mrr = 1.0 / (i + 1)
                break
        
        # Precision@k
        precisions = {}
        for k in [1, 3, 5, 10]:
            relevant = sum(1 for score in relevance_scores[:k] if score > 0.7)
            precisions[f"P@{k}"] = relevant / k if k > 0 else 0.0
        
        return {
            "NDCG@10": ndcg,
            "MRR": mrr,
            **precisions
        }
    
    def run_full_experiment(self, test_data: List[Tuple], num_runs: int = 10):
        """Ejecuta experimento completo"""
        all_results = {}
        
        for config_id in [1, 2, 3, 4]:
            config_results = []
            
            for run in range(num_runs):
                run_results = []
                
                for query_embedding, query_category, products in test_data:
                    result = self.evaluate_configuration(
                        config_id, query_embedding, query_category, products
                    )
                    
                    # Calcular métricas (usando scores simulados)
                    metrics = self.compute_metrics(
                        result["results"], result["scores"]
                    )
                    
                    run_results.append({
                        "query": f"query_{len(run_results)}",
                        "metrics": metrics,
                        "ranking": result["ranking"][:5]
                    })
                
                # Promediar métricas del run
                avg_metrics = {}
                for metric in ["NDCG@10", "MRR", "P@1", "P@3", "P@5", "P@10"]:
                    values = [r["metrics"].get(metric, 0) for r in run_results]
                    avg_metrics[metric] = np.mean(values) if values else 0.0
                
                config_results.append({
                    "run": run,
                    "avg_metrics": avg_metrics,
                    "details": run_results
                })
            
            # Promediar entre runs
            final_metrics = {}
            for metric in ["NDCG@10", "MRR", "P@1", "P@3", "P@5", "P@10"]:
                values = [r["avg_metrics"][metric] for r in config_results]
                final_metrics[metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "values": values
                }
            
            all_results[config_id] = {
                "config": self.configurations[config_id],
                "final_metrics": final_metrics,
                "all_runs": config_results
            }
        
        self.results = all_results
        return all_results
    
    def save_results(self, path: str):
        """Guarda resultados del experimento"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "configurations": self.configurations,
            "results": self.results,
            "summary": self._generate_summary()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Resultados guardados en: {path}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Genera resumen ejecutivo"""
        if not self.results:
            return {}
        
        summary = {}
        
        for config_id, result in self.results.items():
            config_name = self.configurations[config_id]["name"]
            metrics = result["final_metrics"]
            
            summary[config_name] = {
                "NDCG@10": f"{metrics['NDCG@10']['mean']:.3f} ± {metrics['NDCG@10']['std']:.3f}",
                "MRR": f"{metrics['MRR']['mean']:.3f} ± {metrics['MRR']['std']:.3f}",
                "P@1": f"{metrics['P@1']['mean']:.3f} ± {metrics['P@1']['std']:.3f}",
                "P@5": f"{metrics['P@5']['mean']:.3f} ± {metrics['P@5']['std']:.3f}"
            }
        
        return summary