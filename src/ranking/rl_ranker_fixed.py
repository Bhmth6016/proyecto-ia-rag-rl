# src/ranking/rl_ranker_fixed.py
"""
RLHFRankerFixed - Sistema de Pesos Dinámicos y Adaptativos
Aprende automáticamente cuánto peso dar a cada tipo de feature
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import pickle
import logging

logger = logging.getLogger(__name__)

class RLHFRankerFixed:
    """RLHF con pesos dinámicos que se ajustan según éxito real"""
    
    def __init__(self, learning_rate: float = 0.3, match_rating_balance: float = 1.5):
        """
        Sistema de pesos adaptativos
        
        Args:
            learning_rate: Tasa de aprendizaje base
            match_rating_balance: Balance inicial (se ajusta dinámicamente)
        """
        # Features aprendidas
        self.feature_weights = defaultdict(float)
        self.feature_counts = defaultdict(int)
        self.feedback_count = 0
        self.learning_rate = learning_rate
        self.match_rating_balance = match_rating_balance
        self.has_learned = False
        
        # NUEVO: Sistema de pesos dinámicos
        self.dynamic_weights = {
            'match_weight': 0.3,      # Peso inicial para matches
            'rating_weight': 0.3,     # Peso inicial para ratings
            'category_weight': 0.2,   # Peso inicial para categorías
            'specific_weight': 0.2    # Peso inicial para preferencias específicas
        }
        
        # Historial de éxito por tipo de feature
        self.feature_success = defaultdict(lambda: {'hits': 0, 'total': 0})
        
        # Estadísticas
        self.learning_history = []
        self.query_product_pairs = defaultdict(int)
        
        logger.info(f"🤖 RLHF Dinámico inicializado (lr={learning_rate})")
    
    def extract_semantic_features(self, product, query=""):
        """Extrae features con categorización clara"""
        features = {}
        
        if not query or not hasattr(product, 'title'):
            return features
        
        query_lower = query.lower()
        title_lower = product.title.lower() if product.title else ""
        
        # 1. FEATURES DE MATCH (tipo: match)
        query_words = set(query_lower.split())
        title_words = set(title_lower.split())
        
        if query_words:
            match_count = len(query_words.intersection(title_words))
            match_ratio = match_count / len(query_words)
            
            features['semantic_match_ratio'] = min(match_ratio, 1.0)
            features['_type_semantic_match_ratio'] = 'match'
            
            if match_ratio >= 0.9:
                features['excellent_semantic_match'] = 1.0
                features['_type_excellent_semantic_match'] = 'match'
            elif match_ratio >= 0.7:
                features['good_semantic_match'] = 1.0
                features['_type_good_semantic_match'] = 'match'
        
        # 2. FEATURES DE RATING (tipo: rating)
        if hasattr(product, 'rating') and product.rating:
            try:
                rating_val = float(product.rating)
                features['has_rating'] = 1.0
                features['_type_has_rating'] = 'rating'
                
                features['rating_value'] = rating_val / 5.0
                features['_type_rating_value'] = 'rating'
                
                if rating_val >= 4.5:
                    features['excellent_rating'] = 1.0
                    features['_type_excellent_rating'] = 'rating'
                elif rating_val >= 4.0:
                    features['good_rating'] = 1.0
                    features['_type_good_rating'] = 'rating'
            except (ValueError, TypeError):
                pass
        
        # 3. FEATURES DE CATEGORÍA (tipo: category)
        if hasattr(product, 'category'):
            cat_lower = str(product.category).lower()
            features['has_category'] = 1.0
            features['_type_has_category'] = 'category'
            
            # Match de categoría con query
            if any(word in cat_lower for word in query_lower.split()):
                features['category_match'] = 1.0
                features['_type_category_match'] = 'category'
        
        return features
    
    def learn_from_human_feedback(self, clicked_product, query, position, reward=1.0):
        """
        Aprendizaje con ajuste dinámico de pesos
        """
        self.feedback_count += 1
        
        # Extraer features
        features = self.extract_semantic_features(clicked_product, query)
        
        # Reward basado en posición (discovery bonus)
        if position == 1:
            position_factor = 0.5
        elif position <= 3:
            position_factor = 0.8
        elif position <= 10:
            position_factor = 1.2
        else:
            position_factor = 1.5
        
        adjusted_reward = reward * position_factor
        
        # APRENDIZAJE NORMAL
        feature_types_used = defaultdict(int)
        
        for feature_name, feature_value in features.items():
            if feature_name.startswith('_type_'):
                continue  # Skip metadata
            
            if isinstance(feature_value, (int, float)) and feature_value > 0:
                # Obtener tipo de feature
                feature_type = features.get(f'_type_{feature_name}', 'other')
                feature_types_used[feature_type] += 1
                
                # Aprender peso de la feature
                adjustment = adjusted_reward * feature_value * self.learning_rate
                
                # Aplicar balance inicial (se ajustará dinámicamente)
                if feature_type == 'match':
                    adjustment *= self.match_rating_balance
                
                self.feature_weights[feature_name] += adjustment
                self.feature_counts[feature_name] += 1
        
        # Feature específica query-producto
        if hasattr(clicked_product, 'id'):
            specific_feature = f"preference_{hash(query) % 1000}_{hash(clicked_product.id) % 1000}"
            self.feature_weights[specific_feature] += adjusted_reward * 0.3
            feature_types_used['specific'] += 1
        
        # ACTUALIZAR ESTADÍSTICAS DE ÉXITO
        for ftype in feature_types_used:
            self.feature_success[ftype]['hits'] += 1
            self.feature_success[ftype]['total'] += 1
        
        # Actualizar todos los tipos (incluso si no se usaron)
        for ftype in ['match', 'rating', 'category', 'specific']:
            if ftype not in feature_types_used:
                self.feature_success[ftype]['total'] += 1
        
        # AJUSTE DINÁMICO DE PESOS (cada 5 feedbacks)
        if self.feedback_count % 5 == 0:
            self._adjust_dynamic_weights()
        
        # Normalizar periódicamente
        if self.feedback_count % 20 == 0:
            self._normalize_weights_soft()
        
        # Marcar como aprendido
        if self.feedback_count >= 10 and len(self.feature_weights) >= 5:
            self.has_learned = True
        
        # Guardar en historial
        self.learning_history.append({
            'count': self.feedback_count,
            'query': query[:30],
            'position': position,
            'reward': adjusted_reward,
            'dynamic_weights': dict(self.dynamic_weights)
        })
        
        logger.info(f"📚 Human Feedback #{self.feedback_count}: '{query[:20]}...' pos {position}")
        
        return adjusted_reward
    
    def _adjust_dynamic_weights(self):
        """
        Ajusta dinámicamente los pesos según éxito real
        
        Principio: Features que aparecen en productos clickeados
        reciben más peso en el futuro
        """
        # Calcular tasas de éxito
        success_rates = {}
        for ftype, stats in self.feature_success.items():
            if stats['total'] > 0:
                success_rates[ftype] = stats['hits'] / stats['total']
            else:
                success_rates[ftype] = 0.0
        
        # Normalizar a suma = 1.0
        total = sum(success_rates.values())
        if total > 0:
            for ftype in success_rates:
                success_rates[ftype] /= total
        
        # Actualizar pesos dinámicos (suavizado)
        alpha = 0.3  # Factor de suavizado
        
        self.dynamic_weights['match_weight'] = (
            alpha * success_rates.get('match', 0.25) +
            (1 - alpha) * self.dynamic_weights['match_weight']
        )
        
        self.dynamic_weights['rating_weight'] = (
            alpha * success_rates.get('rating', 0.25) +
            (1 - alpha) * self.dynamic_weights['rating_weight']
        )
        
        self.dynamic_weights['category_weight'] = (
            alpha * success_rates.get('category', 0.25) +
            (1 - alpha) * self.dynamic_weights['category_weight']
        )
        
        self.dynamic_weights['specific_weight'] = (
            alpha * success_rates.get('specific', 0.25) +
            (1 - alpha) * self.dynamic_weights['specific_weight']
        )
        
        logger.debug(f"   🔄 Pesos dinámicos ajustados: {self.dynamic_weights}")
    
    def _normalize_weights_soft(self):
        """Normalización suave"""
        if not self.feature_weights:
            return
        
        max_weight = max(abs(w) for w in self.feature_weights.values())
        
        if max_weight > 10.0:
            scale = 8.0 / max_weight
            for key in self.feature_weights:
                self.feature_weights[key] *= scale
    
    def rank_with_human_preferences(self, products, query, baseline_scores=None):
        """
        Ranking con pesos dinámicos adaptativos
        """
        if not self.has_learned or not self.feature_weights:
            return self._baseline_ranking(products, query, baseline_scores)
        
        logger.info(f"   🧠 Aplicando Human Preferences ({len(self.feature_weights)} features)")
        logger.debug(f"   Pesos dinámicos: match={self.dynamic_weights['match_weight']:.2f}, "
                    f"rating={self.dynamic_weights['rating_weight']:.2f}, "
                    f"category={self.dynamic_weights['category_weight']:.2f}")
        
        scored_products = []
        
        for i, product in enumerate(products):
            # Score base
            if baseline_scores and i < len(baseline_scores):
                base_score = baseline_scores[i]
            else:
                base_score = 1.0 - (i / len(products))
            
            # Normalizar
            if baseline_scores and max(baseline_scores) > 0:
                base_score = base_score / max(baseline_scores)
            
            # Score de preferencias humanas con pesos dinámicos
            human_score = 0.0
            feature_contributions = defaultdict(float)
            
            features = self.extract_semantic_features(product, query)
            
            for feature_name, feature_value in features.items():
                if feature_name.startswith('_type_'):
                    continue
                
                if feature_name in self.feature_weights:
                    weight = self.feature_weights[feature_name]
                    feature_type = features.get(f'_type_{feature_name}', 'other')
                    
                    # Aplicar peso dinámico según tipo
                    if feature_type == 'match':
                        dynamic_weight = self.dynamic_weights['match_weight']
                    elif feature_type == 'rating':
                        dynamic_weight = self.dynamic_weights['rating_weight']
                    elif feature_type == 'category':
                        dynamic_weight = self.dynamic_weights['category_weight']
                    else:
                        dynamic_weight = 0.1
                    
                    contribution = feature_value * weight * dynamic_weight
                    human_score += contribution
                    feature_contributions[feature_type] += contribution
            
            # Preferencia específica
            if hasattr(product, 'id'):
                specific_feature = f"preference_{hash(query) % 1000}_{hash(product.id) % 1000}"
                if specific_feature in self.feature_weights:
                    contribution = self.feature_weights[specific_feature] * self.dynamic_weights['specific_weight']
                    human_score += contribution
            
            # Combinar: 70% baseline, 30% human (pero human_score NO limitado)
            combined_score = (base_score * 0.7) + (human_score * 0.3)
            
            scored_products.append((product, combined_score, human_score))
        
        # Ordenar
        scored_products.sort(key=lambda x: x[1], reverse=True)
        
        return [product for product, _, _ in scored_products]
    
    # Alias para compatibilidad
    def learn_from_feedback(self, *args, **kwargs):
        return self.learn_from_human_feedback(*args, **kwargs)
    
    def rank_products(self, *args, **kwargs):
        return self.rank_with_human_preferences(*args, **kwargs)
    
    def _baseline_ranking(self, products, query, baseline_scores):
        """Ranking baseline"""
        if baseline_scores and len(baseline_scores) == len(products):
            sorted_indices = np.argsort(baseline_scores)[::-1]
            return [products[i] for i in sorted_indices]
        return products
    
    def get_stats(self):
        """Estadísticas con pesos dinámicos"""
        stats = {
            'feedback_count': self.feedback_count,
            'weights_count': len(self.feature_weights),
            'has_learned': self.has_learned,
            'dynamic_weights': dict(self.dynamic_weights),
            'feature_success_rates': {}
        }
        
        # Tasas de éxito
        for ftype, data in self.feature_success.items():
            if data['total'] > 0:
                stats['feature_success_rates'][ftype] = data['hits'] / data['total']
            else:
                stats['feature_success_rates'][ftype] = 0.0
        
        return stats
    
    def save(self, path):
        """Guarda el ranker"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """Carga un ranker"""
        with open(path, 'rb') as f:
            return pickle.load(f)