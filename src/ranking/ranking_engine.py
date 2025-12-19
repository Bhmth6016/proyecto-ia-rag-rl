# src/ranking/ranking_engine.py
"""
Ranking estático - Baseline para el paper
NO aprende, solo combina características con pesos fijos
"""
import numpy as np
from typing import List, Dict, Any
import sys
import os

# Añadir src al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importaciones absolutas
try:
    from data.canonicalizer import CanonicalProduct
    from features.features import StaticFeatures
except ImportError:
    # Fallback
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.canonicalizer import CanonicalProduct
    from features.features import StaticFeatures

class StaticRankingEngine:
    """Motor de ranking estático - Baseline"""
    
    def __init__(self, weights: Dict[str, float] = None):
        # Pesos fijos predefinidos
        self.weights = weights or {
            "content_similarity": 0.4,
            "title_similarity": 0.2,
            "category_exact_match": 0.15,
            "rating_normalized": 0.1,
            "price_available": 0.05,
            "has_popularity": 0.05,
            "title_length": 0.025,
            "desc_length": 0.025
        }
        
        # Normalizar pesos
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
    
    def rank_products(
        self,
        query_embedding: np.ndarray,
        query_category: str,
        products: List[CanonicalProduct],
        top_k: int = 10
    ) -> List[CanonicalProduct]:
        """Ranking estático de productos"""
        scored_products = []
        
        for product in products:
            # Extraer características
            features = StaticFeatures.extract_all_features(
                query_embedding, query_category, product
            )
            
            # Calcular score (combinación lineal)
            score = 0.0
            for feature_name, weight in self.weights.items():
                if feature_name in features:
                    score += features[feature_name] * weight
            
            # Añadir producto con score
            scored_products.append((score, product))
        
        # Ordenar por score
        scored_products.sort(key=lambda x: x[0], reverse=True)
        
        # Devolver top-k
        return [p for _, p in scored_products[:top_k]]
    
    def get_score_breakdown(
        self,
        query_embedding: np.ndarray,
        query_category: str,
        product: CanonicalProduct
    ) -> Dict[str, Any]:
        """Obtiene desglose de score para análisis"""
        features = StaticFeatures.extract_all_features(
            query_embedding, query_category, product
        )
        
        breakdown = {"features": features, "weights": self.weights}
        
        # Calcular contribuciones
        contributions = {}
        total_score = 0.0
        for feature_name, weight in self.weights.items():
            if feature_name in features:
                contribution = features[feature_name] * weight
                contributions[feature_name] = {
                    "value": features[feature_name],
                    "weight": weight,
                    "contribution": contribution
                }
                total_score += contribution
        
        breakdown["contributions"] = contributions
        breakdown["total_score"] = total_score
        
        return breakdown