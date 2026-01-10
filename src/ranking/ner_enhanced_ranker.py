# src/ranking/ner_enhanced_ranker.py
from typing import List, Dict, Any
import logging
logger = logging.getLogger(__name__)
class NEREnhancedRanker:
    def __init__(self, ner_weight: float = 0.25):
        self.ner_weight = ner_weight
        self.feature_weights = {}
        self.has_learned = False
        self.bonus_applied_count = 0
        
        logger.info(f" NER Enhanced Ranker (weight={ner_weight})")
    
    def _clean_query(self, query: str) -> str:
        cleaned = query.replace('"', '').replace("'", "")
        cleaned = cleaned.replace('...', '')
        cleaned = ' '.join(cleaned.split())
        return cleaned.strip().lower()
    
    def rank_with_ner(self, products: List, query: str,
                    baseline_scores: List[float]) -> List:
        if not products:
            return []
        
        query_cleaned = self._clean_query(query)
        
        query_intent = self._extract_query_intent_aggressive(query_cleaned)
        
        if query_intent:
            logger.info(f"    Query: '{query_cleaned}' → Intent: {list(query_intent.keys())}")
        
        scored = []
        bonus_applied = 0
        
        for i, product in enumerate(products):
            baseline_score = baseline_scores[i] if i < len(baseline_scores) else 0.0
            
            ner_bonus = self._calculate_ner_bonus_improved(
                product, query_cleaned, query_intent
            )
            
            if ner_bonus > 0:
                bonus_applied += 1
            
            final_score = baseline_score + (ner_bonus * self.ner_weight)
            final_score = max(0.0, min(1.0, final_score))
            
            scored.append((product, final_score, ner_bonus))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        if bonus_applied > 0:
            logger.info(f"    NER bonus aplicado a {bonus_applied}/{len(products)} productos")
        else:
            logger.warning(f"    NER bonus NO aplicado para: '{query_cleaned}'")
        
        return [product for product, _, _ in scored]
    
    def _extract_query_intent_aggressive(self, query: str) -> Dict[str, List[str]]:
        query_lower = query.lower()
        intent = {}
        
        categories = {
            'gaming': ['game', 'gaming', 'play', 'nintendo', 'playstation', 'xbox', 'switch', 'pc gaming'],
            'electronics': ['electronic', 'computer', 'laptop', 'pc', 'phone', 'tablet', 'device'],
            'sports': ['sport', 'equipment', 'fitness', 'outdoor', 'exercise'],
            'automotive': ['car', 'auto', 'vehicle', 'part', 'tire', 'engine'],
            'beauty': ['beauty', 'makeup', 'cosmetic', 'skin', 'hair'],
            'kitchen': ['kitchen', 'cooking', 'utensil', 'cookware'],
            'toys': ['toy', 'kid', 'child', 'baby', 'play']
        }
        
        found_categories = []
        for category, keywords in categories.items():
            if any(kw in query_lower for kw in keywords):
                found_categories.append(category)
        
        if found_categories:
            intent['category'] = found_categories
        
        platforms = {
            'playstation': ['playstation', 'ps4', 'ps5', 'ps3', 'ps '],
            'xbox': ['xbox', 'x-box', 'xb'],
            'nintendo': ['nintendo', 'switch', 'wii', 'ds', '3ds'],
            'pc': ['pc', 'computer', 'windows', 'steam'],
            'mobile': ['mobile', 'ios', 'android', 'phone']
        }
        
        found_platforms = []
        for platform, keywords in platforms.items():
            if any(kw in query_lower for kw in keywords):
                found_platforms.append(platform)
        
        if found_platforms:
            intent['platform'] = found_platforms
        
        genres_keywords = {
            'action': ['action', 'fighting', 'combat'],
            'adventure': ['adventure', 'exploration'],
            'rpg': ['rpg', 'role playing', 'role-playing'],
            'strategy': ['strategy', 'tactical'],
            'sports': ['sports', 'football', 'soccer', 'basketball', 'racing'],
            'survival': ['survival', 'zombie', 'apocalypse'],
            'shooter': ['shooter', 'fps', 'shooting'],
            'simulation': ['simulation', 'sim', 'simulator'],
            'puzzle': ['puzzle', 'brain', 'logic']
        }
        
        found_genres = []
        for genre, keywords in genres_keywords.items():
            if any(kw in query_lower for kw in keywords):
                found_genres.append(genre)
        
        if found_genres:
            intent['genre'] = found_genres
        
        franchises = {
            'mario': ['mario'],
            'zelda': ['zelda'],
            'pokemon': ['pokemon', 'pokémon'],
            'sonic': ['sonic'],
            'minecraft': ['minecraft'],
            'fortnite': ['fortnite'],
            'call of duty': ['call of duty', 'cod']
        }
        
        found_franchises = []
        for franchise, keywords in franchises.items():
            if any(kw in query_lower for kw in keywords):
                found_franchises.append(franchise)
        
        if found_franchises:
            intent['franchise'] = found_franchises
        
        return intent
    
    def _calculate_ner_bonus_improved(self, product, query: str, 
                                     query_intent: Dict) -> float:
        product_attrs = getattr(product, 'ner_attributes', {})
        
        if not product_attrs:
            return 0.0
        
        # MODO 1: Match por intención
        intent_score = 0.0
        if query_intent:
            for intent_key, intent_values in query_intent.items():
                if intent_key in product_attrs:
                    product_values = product_attrs[intent_key]
                    
                    for intent_val in intent_values:
                        for product_val in product_values:
                            if self._fuzzy_match(str(intent_val), str(product_val)):
                                intent_score += 0.3
        
        keyword_score = 0.0
        query_lower = query.lower()
        title = getattr(product, 'title', '').lower()
        
        for attr_type, attr_values in product_attrs.items():
            for attr_val in attr_values:
                attr_lower = str(attr_val).lower()
                if attr_lower in query_lower and len(attr_lower) > 2:
                    keyword_score += 0.2
                elif any(word in title for word in attr_lower.split() if len(word) > 3):
                    keyword_score += 0.1
        
        specificity_score = min(0.2, len(product_attrs) * 0.05)
        
        total_score = intent_score + keyword_score + specificity_score
        
        return min(1.0, total_score)
    
    def _fuzzy_match(self, str1: str, str2: str) -> bool:
        s1_lower = str1.lower().strip()
        s2_lower = str2.lower().strip()
        
        if s1_lower == s2_lower:
            return True
        
        if s1_lower in s2_lower or s2_lower in s1_lower:
            return True
        
        words1 = set(s1_lower.split())
        words2 = set(s2_lower.split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2)
        return overlap / min(len(words1), len(words2)) >= 0.5
    
    def rank_products(self, products: List, query: str, 
                     baseline_scores: List[float]) -> List:
        return self.rank_with_ner(products, query, baseline_scores)
    
    def rank_with_human_preferences(self, products: List, query: str,
                                   baseline_scores: List[float]) -> List:
        return self.rank_with_ner(products, query, baseline_scores)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'ranker_type': 'NER_Enhanced_v2',
            'ner_weight': self.ner_weight,
            'has_learned': self.has_learned,
            'bonus_applied_count': self.bonus_applied_count
        }
    
    def get_learning_health(self) -> Dict[str, Any]:
        return {
            'score': 3,
            'status': ' Estático',
            'issues': ['No learning model, only rule-based NER']
        }