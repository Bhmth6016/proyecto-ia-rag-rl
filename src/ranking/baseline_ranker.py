# src/ranking/baseline_ranker.py
"""
Baseline Ranker - Ranking estático sin aprendizaje
"""
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaselineRanker:
    """Ranking estático basado en características heurísticas"""
    
    def __init__(self, weights: Dict[str, float] = None):
        # Pesos por defecto para características
        self.weights = weights or {
            "content_similarity": 0.4,
            "title_similarity": 0.2,
            "category_exact_match": 0.15,
            "rating_normalized": 0.1,
            "price_available": 0.05,
            "has_brand": 0.05,
            "description_length": 0.05
        }
        
        # Normalizar pesos
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
        
        logger.info(f"📊 BaselineRanker inicializado con {len(self.weights)} características")
    
    def rank(
        self, 
        products: List, 
        query_features: Dict[str, Any],
        product_features: List[Dict[str, float]]
    ) -> List[float]:
        """
        Calcula scores de ranking basados en características
        
        Args:
            products: Lista de productos
            query_features: Características de la query
            product_features: Lista de características por producto
        
        Returns:
            Lista de scores
        """
        if not products or not product_features:
            return []
        
        scores = []
        
        for i, (product, features) in enumerate(zip(products, product_features)):
            score = 0.0
            
            # Aplicar pesos a características disponibles
            for feature_name, weight in self.weights.items():
                if feature_name in features:
                    score += features[feature_name] * weight
            
            # Bonus por características específicas de la query
            score += self._query_specific_bonus(product, query_features)
            
            scores.append(score)
        
        return scores
    
    def _query_specific_bonus(self, product, query_features: Dict[str, Any]) -> float:
        """Bonus basado en match con características de la query"""
        bonus = 0.0
        
        # Match de categoría
        if hasattr(product, 'category') and 'category' in query_features:
            if product.category == query_features['category']:
                bonus += 0.1
        
        # Productos con rating alto para queries de recomendación
        if query_features.get('intent') == 'recommend' and hasattr(product, 'rating'):
            if product.rating and product.rating > 4.0:
                bonus += 0.05
        
        # Productos con precio para queries de compra
        if query_features.get('intent') == 'purchase' and hasattr(product, 'price'):
            if product.price:
                bonus += 0.03
        
        return bonus
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Obtiene importancia de características"""
        return self.weights.copy()