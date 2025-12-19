# src/evaluator/metrics_calculator.py
"""
Calculador de métricas adicionales para el paper
"""
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class AdvancedMetricsCalculator:
    """Calcula métricas avanzadas que los revisores esperan"""
    
    @staticmethod
    def calculate_all_metrics(ranked_products: List, relevance_scores: List[float], 
                            query_embedding: np.ndarray = None) -> Dict[str, float]:
        """Calcula todas las métricas"""
        metrics = {}
        
        # Métricas clásicas
        metrics.update(AdvancedMetricsCalculator.calculate_classic_metrics(
            ranked_products, relevance_scores
        ))
        
        # Métricas RLHF específicas
        metrics.update(AdvancedMetricsCalculator.calculate_rlhf_metrics(
            relevance_scores
        ))
        
        # Métricas de separación (si hay query embedding)
        if query_embedding is not None:
            metrics.update(AdvancedMetricsCalculator.calculate_separation_metrics(
                ranked_products, query_embedding
            ))
        
        return metrics
    
    @staticmethod
    def calculate_classic_metrics(ranked_products: List, 
                                relevance_scores: List[float]) -> Dict[str, float]:
        """Métricas clásicas de ranking"""
        # precision@k
        precisions = {}
        for k in [1, 3, 5, 10]:
            if len(relevance_scores) >= k:
                relevant = sum(1 for score in relevance_scores[:k] if score > 0.7)
                precisions[f"precision@{k}"] = relevant / k
        
        # ndcg@k
        ndcg_scores = {}
        for k in [5, 10]:
            if len(relevance_scores) >= k:
                dcg = 0.0
                for i, score in enumerate(relevance_scores[:k]):
                    dcg += score / np.log2(i + 2)
                
                ideal_scores = sorted(relevance_scores[:k], reverse=True)
                idcg = 0.0
                for i, score in enumerate(ideal_scores):
                    idcg += score / np.log2(i + 2)
                
                ndcg_scores[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
        
        return {**precisions, **ndcg_scores}
    
    @staticmethod
    def calculate_rlhf_metrics(rewards_history: List[float]) -> Dict[str, float]:
        """Métricas específicas de RLHF"""
        if not rewards_history:
            return {}
        
        rewards = np.array(rewards_history)
        
        return {
            "cumulative_reward": np.sum(rewards),
            "average_reward_per_episode": np.mean(rewards),
            "reward_slope": AdvancedMetricsCalculator._calculate_slope(rewards),
            "reward_variance": np.var(rewards),
            "exploitation_ratio": np.mean(rewards[-len(rewards)//4:]) / np.mean(rewards) if np.mean(rewards) > 0 else 0.0
        }
    
    @staticmethod
    def calculate_separation_metrics(ranked_products: List, 
                                   query_embedding: np.ndarray) -> Dict[str, float]:
        """Métricas de separación y estabilidad"""
        if not ranked_products or len(ranked_products) < 2:
            return {}
        
        # Calcular similitudes entre productos top-k
        k = min(5, len(ranked_products))
        top_products = ranked_products[:k]
        
        similarities = []
        for i in range(k):
            for j in range(i+1, k):
                if hasattr(top_products[i], 'content_embedding') and hasattr(top_products[j], 'content_embedding'):
                    sim = np.dot(top_products[i].content_embedding, top_products[j].content_embedding)
                    similarities.append(sim)
        
        return {
            "retrieval_stability": np.mean(similarities) if similarities else 0.0,
            "diversity_score": 1.0 - np.mean(similarities) if similarities else 1.0,
            "query_product_alignment": AdvancedMetricsCalculator._calculate_alignment(
                query_embedding, ranked_products[:k]
            )
        }
    
    @staticmethod
    def _calculate_slope(data: np.ndarray) -> float:
        """Calcula pendiente de tendencia"""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        slope, _ = np.polyfit(x, data, 1)
        return float(slope)
    
    @staticmethod
    def _calculate_alignment(query_embedding: np.ndarray, 
                           products: List) -> float:
        """Calcula alineamiento query-productos"""
        if not products:
            return 0.0
        
        alignments = []
        for product in products:
            if hasattr(product, 'content_embedding'):
                alignment = np.dot(query_embedding, product.content_embedding)
                alignments.append(alignment)
        
        return np.mean(alignments) if alignments else 0.0