# src/ranking/rl_ranker_fixed.py
"""
RL Ranker FIXED - Simple, pickleable y que prioriza matches sobre rating
"""
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict
import pickle

class RLHFRankerFixed:
    """RL Ranker simple que se puede picklear y funciona bien"""
    
    def __init__(self, learning_rate: float = 0.2):
        self.feature_weights = defaultdict(float)
        self.feature_counts = defaultdict(int)
        self.feedback_count = 0
        self.learning_rate = learning_rate
        self.has_learned = False
        
    def extract_features(self, product, query=""):
        """Extrae features del producto - MENOS AGRESIVO con matches"""
        features = {}
        
        # 1. MATCH FEATURES (importante pero no tanto)
        if query and hasattr(product, 'title'):
            query_lower = query.lower()
            title_lower = product.title.lower() if product.title else ""
            
            # Palabras en común (valor entre 0 y 1)
            query_words = set(query_lower.split())
            title_words = set(title_lower.split())
            
            match_count = len(query_words.intersection(title_words))
            if query_words:
                match_ratio = match_count / len(query_words)
                features['title_match_ratio'] = min(match_ratio, 1.0) * 0.5  # Reducido a 50%
                features['exact_match_count'] = min(match_count / 5.0, 1.0) * 0.3  # 30%
                
                # Bonus reducido por match alto
                if match_ratio >= 0.8:
                    features['near_perfect_match'] = 0.3  # Reducido
                if match_ratio >= 0.9:
                    features['excellent_match'] = 0.2  # Reducido
            
            # Match exacto de frase (poco común)
            if query_lower in title_lower:
                features['exact_phrase_match'] = 0.4  # Reducido
                features['perfect_match'] = 0.4  # Reducido
        
        # 2. RATING FEATURES (más peso relativo)
        if hasattr(product, 'rating') and product.rating:
            try:
                rating_val = float(product.rating)
                features['has_rating'] = 0.8  # Alto peso
                features['rating_value'] = min(rating_val / 5.0, 1.0) * 0.7  # 70%
                if rating_val >= 4.0:
                    features['high_rating'] = 0.6  # Peso medio-alto
            except (ValueError, TypeError):
                pass
        
        return features
    
    def learn_from_feedback(self, clicked_product, query, position, reward=1.0):
        """Aprende de feedback - APRENDER PRODUCTOS ESPECÍFICOS también"""
        self.feedback_count += 1
        
        # Extraer features
        features = self.extract_features(clicked_product, query)
        
        # Ajustar reward basado en posición
        if position == 1:
            position_factor = 1.5
        elif position <= 3:
            position_factor = 1.2
        elif position <= 5:
            position_factor = 1.0
        elif position <= 10:
            position_factor = 0.7
        else:
            position_factor = 0.3
        
        adjusted_reward = reward * position_factor
        
        # APRENDER: 1) Características genéricas
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, (int, float)) and feature_value > 0:
                adjustment = adjusted_reward * feature_value * self.learning_rate
                
                # ¡IMPORTANTE! Reducir MUCHO el peso de matches genéricos
                if 'match' in feature_name.lower() or 'perfect' in feature_name.lower():
                    adjustment *= 0.5  # Solo 50% del ajuste
                elif 'keyword' in feature_name.lower():
                    adjustment *= 0.3  # Solo 30%
                elif 'rating' in feature_name.lower():
                    adjustment *= 0.8  # 80% para rating
                
                self.feature_weights[feature_name] += adjustment
                self.feature_counts[feature_name] += 1
        
        # APRENDER: 2) Producto específico (esto es nuevo y crucial)
        if hasattr(clicked_product, 'id'):
            # Crear feature única para este producto+query
            product_query_feature = f"product_{clicked_product.id}_for_{hash(query) % 1000}"
            self.feature_weights[product_query_feature] = adjusted_reward * 0.5  # Peso moderado
        
        # Marcar como aprendido
        if self.feedback_count >= 5 and len(self.feature_weights) >= 3:
            self.has_learned = True
        
        # Normalizar periódicamente
        if self.feedback_count % 10 == 0:
            self._normalize_weights()
    
    def _normalize_weights(self):
        """Normaliza pesos para evitar explosión"""
        if not self.feature_weights:
            return
        
        max_weight = max(abs(w) for w in self.feature_weights.values())
        if max_weight > 5.0:
            scale = 3.0 / max_weight
            for key in self.feature_weights:
                self.feature_weights[key] *= scale
    
    def rank_products(self, products, query, baseline_scores=None):
        """Rankea productos - EXTREMADAMENTE CONSERVADOR"""
        if not self.has_learned or not self.feature_weights:
            return self._baseline_ranking(products, query, baseline_scores)
        
        # ¡CRÍTICO! Si el baseline ya es bueno (recupera > 50% relevantes), no cambiar mucho
        # Pero no podemos saber esto sin ground truth...
        # En su lugar: ser extremadamente conservador
        
        # 1. Solo ajustar top 10
        top_n = min(10, len(products))
        products_to_rank = products[:top_n]
        
        # 2. Calcular scores RL con impacto MÍNIMO
        rl_scores = []
        for product in products_to_rank:
            # Score base: mantener posición original (95% peso)
            idx = products_to_rank.index(product)
            base_score = (top_n - idx) / top_n * 0.95  # 95% al orden original
            
            # RL boost: solo 5% peso máximo
            rl_boost = 0.0
            
            # Boost por producto específico (si fue clickeado antes)
            if hasattr(product, 'id'):
                product_query_feature = f"product_{product.id}_for_{hash(query) % 1000}"
                if product_query_feature in self.feature_weights:
                    rl_boost += self.feature_weights[product_query_feature] * 0.2  # 20% del boost
            
            # Boost por características genéricas (el resto del boost)
            features = self.extract_features(product, query)
            generic_boost = 0.0
            for feature_name, feature_value in features.items():
                if feature_name in self.feature_weights:
                    weight = self.feature_weights[feature_name]
                    
                    # Impacto MUY reducido para características genéricas
                    if 'match' in feature_name.lower():
                        generic_boost += feature_value * weight * 0.01  # Solo 1%
                    elif 'rating' in feature_name.lower():
                        generic_boost += feature_value * weight * 0.02  # 2% para rating
                    elif 'keyword' in feature_name.lower():
                        generic_boost += feature_value * weight * 0.005  # 0.5%
            
            rl_boost += generic_boost * 0.8  # 80% del boost restante
            
            # Limitar boost total a 5%
            rl_boost = min(rl_boost, 0.05)
            
            final_score = base_score + rl_boost
            rl_scores.append(final_score)
        
        # 3. Solo reordenar si hay diferencias MUY claras
        if max(rl_scores) - min(rl_scores) > 0.15:  # Umbral alto
            sorted_indices = np.argsort(rl_scores)[::-1]
            ranked_top = [products_to_rank[i] for i in sorted_indices]
            
            # ¡LIMITAR! No mover productos más de 2 posiciones
            final_ranked = []
            for new_pos, (old_idx, product) in enumerate(zip(sorted_indices, ranked_top)):
                if abs(new_pos - old_idx) <= 2:  # Máximo 2 posiciones
                    final_ranked.append(product)
                else:
                    # Mantener cerca de posición original
                    final_ranked.insert(min(old_idx, len(final_ranked)), product)
            
            ranked_top = final_ranked
        else:
            ranked_top = products_to_rank
        
        # 4. Combinar
        ranked_products = ranked_top + products[top_n:]
        
        return ranked_products
    
    def _baseline_ranking(self, products, query, baseline_scores):
        """Ranking baseline cuando RL no ha aprendido"""
        if baseline_scores and len(baseline_scores) == len(products):
            sorted_indices = np.argsort(baseline_scores)[::-1]
            return [products[i] for i in sorted_indices]
        
        # Ordenar por match con query
        scores = []
        for product in products:
            score = 0.0
            
            # Match con título
            if query and hasattr(product, 'title'):
                query_lower = query.lower()
                title_lower = product.title.lower() if product.title else ""
                
                query_words = set(query_lower.split())
                title_words = set(title_lower.split())
                match_count = len(query_words.intersection(title_words))
                
                if query_words:
                    match_ratio = match_count / len(query_words)
                    score = match_ratio * 0.8  # 80% peso a match
            
            # Rating (20% peso)
            if hasattr(product, 'rating') and product.rating:
                try:
                    rating_val = float(product.rating)
                    score += (rating_val / 5.0) * 0.2
                except:
                    pass
            
            scores.append(score)
        
        sorted_indices = np.argsort(scores)[::-1]
        return [products[i] for i in sorted_indices]
    
    def get_stats(self):
        """Obtiene estadísticas"""
        stats = {
            'feedback_count': self.feedback_count,
            'weights_count': len(self.feature_weights),
            'has_learned': self.has_learned,
            'top_features': []
        }
        
        if self.feature_weights:
            sorted_weights = sorted(
                self.feature_weights.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            stats['top_features'] = sorted_weights[:10]
            
            # Clasificar features
            match_weight = sum(abs(w) for f, w in self.feature_weights.items() 
                             if 'match' in f.lower() or 'perfect' in f.lower())
            rating_weight = sum(abs(w) for f, w in self.feature_weights.items() 
                              if 'rating' in f.lower())
            
            stats['match_vs_rating_ratio'] = match_weight / rating_weight if rating_weight > 0 else float('inf')
        
        return stats
    
    def save(self, path):
        """Guarda el ranker entrenado"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """Carga un ranker entrenado"""
        with open(path, 'rb') as f:
            return pickle.load(f)