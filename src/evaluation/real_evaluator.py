# src/evaluation/real_evaluator.py
"""
Evaluador REAL usando interacciones REALES como ground truth
"""
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class RealEvaluator:
    def __init__(self, interaction_logger=None):
        self.interaction_logger = interaction_logger
        self.relevance_cache = {}  # Cache de relevancia
        
    def set_interaction_logger(self, interaction_logger):
        """Establece el logger de interacciones"""
        self.interaction_logger = interaction_logger
    
    def evaluate_ranking(
        self, 
        query: str, 
        ranked_products: List[str],  # IDs de productos rankeados
        mode: str,
        k: int = 5
    ) -> Dict[str, float]:
        """
        Evalúa ranking usando clicks REALES como ground truth
        """
        # Obtener ground truth REAL: productos clickeados para esta query
        relevant_products = self._get_relevant_products_for_query(query)
        
        if not relevant_products:
            # Si no hay clicks para esta query, no podemos evaluar
            return {
                'ndcg@k': 0.0,
                'mrr': 0.0,
                'precision@k': 0.0,
                'recall@k': 0.0,
                'has_ground_truth': False,
                'relevant_products_count': 0,
                'query': query,
                'mode': mode,
                'k': k
            }
        
        # Calcular métricas
        ndcg_score = self.calculate_ndcg(ranked_products, relevant_products, k)
        mrr_score = self.calculate_mrr(ranked_products, relevant_products)
        precision_score = self.calculate_precision(ranked_products, relevant_products, k)
        recall_score = self.calculate_recall(ranked_products, relevant_products, k)
        
        return {
            'ndcg@k': ndcg_score,
            'mrr': mrr_score,
            'precision@k': precision_score,
            'recall@k': recall_score,
            'has_ground_truth': True,
            'relevant_products_count': len(relevant_products),
            'mode': mode,
            'query': query,
            'k': k
        }
    
    def _get_relevant_products_for_query(self, query: str) -> List[str]:
        """Obtiene productos relevantes para una query desde logs reales"""
        if not self.interaction_logger:
            return []
        
        # Usar cache para mejorar rendimiento
        if query in self.relevance_cache:
            return self.relevance_cache[query]
        
        # Obtener ground truth del logger
        try:
            if hasattr(self.interaction_logger, 'get_relevance_labels'):
                relevance_dict = self.interaction_logger.get_relevance_labels()
                relevant_products = relevance_dict.get(query, [])
                self.relevance_cache[query] = relevant_products
                return relevant_products
        except Exception as e:
            logger.error(f"Error obteniendo relevancia para query '{query}': {e}")
        
        return []
    
    def calculate_ndcg(self, ranked: List[str], relevant: List[str], k: int = 5) -> float:
        """Calcula NDCG@k"""
        if not relevant or k == 0:
            return 0.0
        
        # DCG
        dcg = 0.0
        for i, product_id in enumerate(ranked[:k]):
            if product_id in relevant:
                # i+2 porque empezamos desde 1 y log2(1) = 0, así que usamos i+2
                dcg += 1.0 / np.log2(i + 2)
        
        # IDCG (ideal DCG)
        ideal_relevance = [1] * min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(len(ideal_relevance)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_mrr(self, ranked: List[str], relevant: List[str]) -> float:
        """Calcula MRR (Mean Reciprocal Rank)"""
        for i, product_id in enumerate(ranked):
            if product_id in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def calculate_precision(self, ranked: List[str], relevant: List[str], k: int = 5) -> float:
        """Calcula Precision@k"""
        if k == 0 or not ranked:
            return 0.0
        
        relevant_in_top_k = sum(1 for pid in ranked[:k] if pid in relevant)
        return relevant_in_top_k / k
    
    def calculate_recall(self, ranked: List[str], relevant: List[str], k: int = 5) -> float:
        """Calcula Recall@k"""
        if not relevant:
            return 0.0
        
        relevant_in_top_k = sum(1 for pid in ranked[:k] if pid in relevant)
        return relevant_in_top_k / len(relevant)
    
    def evaluate_batch(
        self, 
        queries_results: Dict[str, List[str]],  # query -> ranked products
        modes: Optional[Dict[str, str]] = None  # query -> mode
    ) -> Dict[str, Any]:
        """Evalúa un batch de queries"""
        results = {
            'queries': {},
            'overall': {},
            'by_mode': defaultdict(list)
        }
        
        for query, ranked_products in queries_results.items():
            mode = modes.get(query, 'unknown') if modes else 'unknown'
            
            metrics = self.evaluate_ranking(query, ranked_products, mode)
            results['queries'][query] = metrics
            
            if metrics['has_ground_truth']:
                for metric in ['ndcg@k', 'mrr', 'precision@k', 'recall@k']:
                    results['by_mode'][mode].append(metrics.get(metric, 0.0))
        
        # Calcular métricas agregadas
        for mode, values in results['by_mode'].items():
            if values:
                results['overall'][mode] = {
                    'ndcg@k_mean': float(np.mean(values)),
                    'ndcg@k_std': float(np.std(values)) if len(values) > 1 else 0.0,
                    'num_queries': len(values)
                }
        
        return results
    
    def clear_cache(self):
        """Limpia el cache de relevancia"""
        self.relevance_cache.clear()


class SimpleEvaluator:
    """Evaluador simple para testing cuando no hay logs reales"""
    
    @staticmethod
    def evaluate_ranking_simple(
        ranked_products: List[str],
        relevant_products: List[str]
    ) -> Dict[str, float]:
        """Evaluación simple sin dependencias"""
        if not relevant_products:
            return {
                'ndcg@5': 0.0,
                'mrr': 0.0,
                'precision@5': 0.0,
                'recall@5': 0.0
            }
        
        evaluator = RealEvaluator()
        
        return {
            'ndcg@5': evaluator.calculate_ndcg(ranked_products, relevant_products, 5),
            'mrr': evaluator.calculate_mrr(ranked_products, relevant_products),
            'precision@5': evaluator.calculate_precision(ranked_products, relevant_products, 5),
            'recall@5': evaluator.calculate_recall(ranked_products, relevant_products, 5)
        }