# src/features/extractor.py
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class FeatureEngineer:
    
    def __init__(self):
        self.feature_count = 0
        logger.info("⚙️  FeatureEngineer inicializado")
    
    def extract_query_features(
        self, 
        query_text: str, 
        query_embedding: np.ndarray,
        query_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        features = {
            'query_length': min(1.0, len(query_text) / 100),
            'num_keywords': min(1.0, len(query_analysis.get('keywords', [])) / 10),
            'num_entities': min(1.0, len(query_analysis.get('entities', [])) / 5),
            
            'intent_search': 1.0 if query_analysis.get('intent') == 'search' else 0.0,
            'intent_purchase': 1.0 if query_analysis.get('intent') == 'purchase' else 0.0,
            'intent_compare': 1.0 if query_analysis.get('intent') == 'compare' else 0.0,
            'intent_recommend': 1.0 if query_analysis.get('intent') == 'recommendation' else 0.0,
            
            'specificity': query_analysis.get('specificity', {}).get('specificity_score', 0.0),
            'is_specific': 1.0 if query_analysis.get('is_specific', False) else 0.0,
            
            'category_general': 1.0 if query_analysis.get('category') == 'General' else 0.0,
            'category_electronics': 1.0 if query_analysis.get('category') == 'Electronics' else 0.0,
            'category_books': 1.0 if query_analysis.get('category') == 'Books' else 0.0,
        }
        
        if query_embedding is not None:
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            for i in range(min(3, len(query_norm))):
                features[f'embedding_dim_{i}'] = float(query_norm[i])
        
        self.feature_count += 1
        return features
    
    def extract_product_features(
        self, 
        product, 
        query_features: Dict[str, Any]
    ) -> Dict[str, float]:
        features = {}
        
        features['price_available'] = 1.0 if hasattr(product, 'price') and product.price is not None else 0.0
        features['has_rating'] = 1.0 if hasattr(product, 'rating') and product.rating is not None else 0.0
        features['has_brand'] = 1.0 if hasattr(product, 'brand') and product.brand else 0.0
        
        if hasattr(product, 'rating') and product.rating is not None:
            features['rating_normalized'] = product.rating / 5.0
            features['rating_high'] = 1.0 if product.rating > 4.0 else 0.0
            features['rating_low'] = 1.0 if product.rating < 2.5 else 0.0
        else:
            features['rating_normalized'] = 0.0
            features['rating_high'] = 0.0
            features['rating_low'] = 0.0
        
        if hasattr(product, 'title'):
            features['title_length'] = min(1.0, len(product.title) / 100)
        
        if hasattr(product, 'description'):
            features['desc_length'] = min(1.0, len(product.description) / 50000)
        
        if hasattr(product, 'category') and 'category' in query_features:
            category = query_features.get('category', '')
            features['category_exact_match'] = 1.0 if product.category == category else 0.0
            
            category_pairs = [
                ('Electronics', 'Computers'),
                ('Electronics', 'Video Games'),
                ('Books', 'Educational'),
                ('Clothing', 'Shoes')
            ]
            
            features['category_partial_match'] = 0.0
            for cat1, cat2 in category_pairs:
                if (category == cat1 and product.category == cat2) or \
                   (category == cat2 and product.category == cat1):
                    features['category_partial_match'] = 0.7
                    break
        
        if hasattr(product, 'title_embedding') and 'embedding_dim_0' in query_features:
            pass  # Se calcula en el ranking engine
        
        if hasattr(product, 'price') and product.price is not None:
            features['price_normalized'] = min(1.0, product.price / 1000)
            features['price_low'] = 1.0 if product.price < 50 else 0.0
            features['price_high'] = 1.0 if product.price > 500 else 0.0
        
        return features
    
    def normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        normalized = {}
        
        for key, value in features.items():
            if any(substr in key for substr in ('normalized', 'match', 'available')):
                normalized[key] = max(0.0, min(1.0, value))
            elif 'length' in key:
                normalized[key] = value
            elif 'price' in key and 'normalized' not in key:
                normalized[key] = 1.0 if value > 0 else 0.0
            else:
                normalized[key] = 1.0 if value > 0.5 else 0.0
        
        return normalized

    
    def get_feature_stats(self) -> dict:
        return {
            'feature_extractions': self.feature_count,
            'purpose': 'ranking_features_only',
            'principles': ['no_retrieval_modification', 'no_embedding_generation']
        }