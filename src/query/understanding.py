# src/query/understanding.py
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)
class QueryAnalyzer:
    def __init__(self):
        self.categories = {
            'Automotive': ['car', 'auto', 'vehicle', 'truck'],
            'Beauty': ['beauty', 'makeup', 'skincare'],
            'Electronics': ['phone', 'laptop', 'tablet'],
            'Home': ['home', 'furniture', 'kitchen'],
            'Sports': ['sport', 'fitness', 'outdoor']
        }
        
        self.intents = {
            'purchase': ['buy', 'purchase', 'order'],
            'compare': ['compare', 'versus', 'better'],
            'recommend': ['best', 'recommend', 'good']
        }
        
        logger.info(" QueryAnalyzer initialized (features only)")
    
    def extract(self, query: str) -> Dict[str, Any]:
        query_lower = query.lower().strip()
        
        features = {
            'query': query,
            'query_length': len(query_lower.split()),
            'user_intent': self._classify_intent(query_lower),
            'product_category': self._classify_category(query_lower),
            'specificity_score': self._calculate_specificity(query_lower),
            'keywords': self._extract_keywords(query_lower),
            'analysis_timestamp': 'now',
            'module_type': 'feature_extractor'
        }
        
        logger.debug(f"Extracted features from query: {query[:30]}...")
        return features
    
    def extract_features(self, query: str) -> Dict[str, Any]:
        return self.extract(query)
    
    def _classify_intent(self, query: str) -> str:
        for intent, keywords in self.intents.items():
            if any(keyword in query for keyword in keywords):
                return intent
        return 'exploratory'
    
    def _classify_category(self, query: str) -> str:
        for category, keywords in self.categories.items():
            if any(keyword in query for keyword in keywords):
                return category
        return 'General'
    
    def _calculate_specificity(self, query: str) -> float:
        words = query.split()
        if not words:
            return 0.0
        
        specific_terms = {'specific', 'exact', 'model', 'brand', 'color', 'size'}
        found_terms = sum(1 for word in words if word in specific_terms)
        
        return min(1.0, found_terms / 3)
    
    def _extract_keywords(self, query: str) -> List[str]:
        common = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for'}
        words = query.split()
        
        return [
            word for word in words
            if word not in common and len(word) > 2
        ][:5]
    
    def get_module_info(self) -> Dict[str, str]:
        return {
            'name': 'QueryAnalyzer',
            'purpose': 'Extract ranking features from queries',
            'architecture_role': 'Feature provider',
            'independence_level': 'complete'
        }
QueryUnderstanding = QueryAnalyzer