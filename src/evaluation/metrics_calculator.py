# src/evaluation/metrics_calculator.py
"""
Calcula métricas REALES de evaluación
"""
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RealMetricsCalculator:
    """Calcula métricas reales de evaluación"""
    
    @staticmethod
    def calculate_ndcg(ranked_products: List[str], relevant_products: List[str], k: int = 5) -> float:
        """Calcula NDCG@k REAL"""
        if not relevant_products:
            return 0.0
        
        # Calcular DCG
        dcg = 0.0
        for i, product_id in enumerate(ranked_products[:k]):
            if product_id in relevant_products:
                # Relevancia binaria (1 si es relevante, 0 si no)
                relevance = 1.0
                dcg += relevance / np.log2(i + 2)  # i+2 porque el índice empieza en 0
        
        # Calcular IDCG (ranking ideal)
        ideal_relevance = [1.0] * min(len(relevant_products), k)
        idcg = sum(relevance / np.log2(i + 2) for i, relevance in enumerate(ideal_relevance))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def calculate_mrr(ranked_products: List[str], relevant_products: List[str]) -> float:
        """Calcula MRR REAL"""
        for i, product_id in enumerate(ranked_products):
            if product_id in relevant_products:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def calculate_precision(ranked_products: List[str], relevant_products: List[str], k: int = 10) -> float:
        """Calcula Precisión@k REAL"""
        if not ranked_products[:k]:
            return 0.0
        
        relevant_in_top_k = sum(1 for pid in ranked_products[:k] if pid in relevant_products)
        return relevant_in_top_k / k
    
    @staticmethod
    def calculate_recall(ranked_products: List[str], relevant_products: List[str], k: int = 10) -> float:
        """Calcula Recall@k REAL"""
        if not relevant_products:
            return 0.0
        
        relevant_in_top_k = sum(1 for pid in ranked_products[:k] if pid in relevant_products)
        return relevant_in_top_k / len(relevant_products)
    
    @staticmethod
    def calculate_all_metrics(ranked_products: List, relevant_products: List[str]) -> Dict[str, float]:
        """Calcula todas las métricas REALES"""
        ranked_ids = [p.id if hasattr(p, 'id') else str(p) for p in ranked_products]
        
        return {
            "ndcg@5": RealMetricsCalculator.calculate_ndcg(ranked_ids, relevant_products, k=5),
            "ndcg@10": RealMetricsCalculator.calculate_ndcg(ranked_ids, relevant_products, k=10),
            "mrr": RealMetricsCalculator.calculate_mrr(ranked_ids, relevant_products),
            "precision@5": RealMetricsCalculator.calculate_precision(ranked_ids, relevant_products, k=5),
            "precision@10": RealMetricsCalculator.calculate_precision(ranked_ids, relevant_products, k=10),
            "recall@5": RealMetricsCalculator.calculate_recall(ranked_ids, relevant_products, k=5),
            "recall@10": RealMetricsCalculator.calculate_recall(ranked_ids, relevant_products, k=10)
        }