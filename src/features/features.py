# src/features/features.py
"""
Características estáticas para ranking - NO aprenden
"""
import numpy as np
from typing import List, Dict, Any
import sys
import os

# Añadir src al path para importaciones absolutas
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importación absoluta
try:
    from data.canonicalizer import CanonicalProduct
except ImportError:
    # Fallback para desarrollo
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.canonicalizer import CanonicalProduct

class StaticFeatures:
    """Características estáticas predefinidas"""
    
    @staticmethod
    def extract_product_features(product: CanonicalProduct) -> Dict[str, float]:
        """Extrae características de un producto"""
        features = {}
        
        # 1. Disponibilidad de información
        features["price_available"] = 1.0 if product.price is not None else 0.0
        features["has_rating"] = 1.0 if product.rating is not None else 0.0
        
        # 2. Calidad (rating)
        if product.rating:
            features["rating_normalized"] = product.rating / 5.0
            features["has_popularity"] = 1.0 if product.rating_count and product.rating_count > 10 else 0.0
        else:
            features["rating_normalized"] = 0.0
            features["has_popularity"] = 0.0
        
        # 3. Longitud de información (proxy de completitud)
        features["title_length"] = min(1.0, len(product.title) / 100)
        features["desc_length"] = min(1.0, len(product.description) / 500)
        
        return features
    
    @staticmethod
    def compute_similarity_features(
        query_embedding: np.ndarray,
        product: CanonicalProduct
    ) -> Dict[str, float]:
        """Calcula características de similitud"""
        features = {}
        
        # Similitud con título
        title_sim = np.dot(query_embedding, product.title_embedding)
        features["title_similarity"] = float(title_sim)
        
        # Similitud con contenido completo
        content_sim = np.dot(query_embedding, product.content_embedding)
        features["content_similarity"] = float(content_sim)
        
        return features
    
    @staticmethod
    def compute_category_match(
        query_category: str,
        product: CanonicalProduct
    ) -> Dict[str, float]:
        """Características de match de categoría"""
        features = {}
        
        # Match exacto
        features["category_exact_match"] = 1.0 if query_category == product.category else 0.0
        
        # Match parcial (categorías similares)
        category_similarities = {
            ("Electronics", "Computers"): 0.8,
            ("Electronics", "Video Games"): 0.6,
            ("Clothing", "Shoes"): 0.7,
            ("Books", "Educational"): 0.6,
            ("Home & Kitchen", "Furniture"): 0.7,
        }
        
        features["category_partial_match"] = 0.0
        for (cat1, cat2), sim in category_similarities.items():
            if (query_category == cat1 and product.category == cat2) or \
               (query_category == cat2 and product.category == cat1):
                features["category_partial_match"] = sim
                break
        
        return features
    
    @staticmethod
    def extract_all_features(
        query_embedding: np.ndarray,
        query_category: str,
        product: CanonicalProduct
    ) -> Dict[str, float]:
        """Extrae TODAS las características"""
        features = {}
        
        # Product features
        features.update(StaticFeatures.extract_product_features(product))
        
        # Similarity features
        features.update(StaticFeatures.compute_similarity_features(query_embedding, product))
        
        # Category features
        features.update(StaticFeatures.compute_category_match(query_category, product))
        
        return features