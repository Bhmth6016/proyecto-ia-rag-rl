# src/ranking/rl_ranker_fixed.py
"""
RLHFRankerFixed CORREGIDO Y COMPLETO - True Human Feedback Learning
Versión funcional con todos los métodos necesarios
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import pickle
import logging

logger = logging.getLogger(__name__)

class RLHFRankerFixed:
    """True RLHF: Aprende de PREFERENCIAS HUMANAS, no solo matches"""
    
    def __init__(self, learning_rate: float = 0.3, match_rating_balance: float = 1.5):
        """
        Args:
            learning_rate: Tasa de aprendizaje
            match_rating_balance: Balance entre match (2.0) vs rating (1.0)
        """
        self.feature_weights = defaultdict(float)
        self.feature_counts = defaultdict(int)
        self.feedback_count = 0
        self.learning_rate = learning_rate
        self.match_rating_balance = match_rating_balance
        self.has_learned = False
        
        # Estadísticas de aprendizaje
        self.learning_history = []
        self.query_product_pairs = defaultdict(int)  # (query, producto) -> clicks
        
        logger.info(f"🤖 RLHF True inicializado (balance match/rating: {match_rating_balance})")
    
    def extract_semantic_features(self, product, query=""):
        """Extrae features SEMÁNTICAS, no solo de palabras"""
        features = {}
        
        # 1. FEATURES SEMÁNTICAS (las más importantes)
        if query and hasattr(product, 'title'):
            query_lower = query.lower()
            title_lower = product.title.lower() if product.title else ""
            
            # A. Match semántico (palabras clave)
            query_words = set(query_lower.split())
            title_words = set(title_lower.split())
            
            match_count = len(query_words.intersection(title_words))
            if query_words:
                # Match ratio: importancia MODERADA
                match_ratio = match_count / len(query_words)
                features['semantic_match_ratio'] = min(match_ratio, 1.0) * 0.7  # 70% max
                
                # Bonus por match perfecto pero limitado
                if match_ratio >= 0.9:
                    features['excellent_semantic_match'] = 0.3  # 30% bonus
                elif match_ratio >= 0.7:
                    features['good_semantic_match'] = 0.2  # 20% bonus
            
            # B. Match exacto de frases importantes (menos peso)
            important_phrases = ['kit', 'replacement', 'genuine', 'professional']
            for phrase in important_phrases:
                if phrase in query_lower and phrase in title_lower:
                    features[f'contains_{phrase}'] = 0.1
        
        # 2. FEATURES DE CALIDAD (importancia moderada)
        if hasattr(product, 'rating') and product.rating:
            try:
                rating_val = float(product.rating)
                features['has_rating'] = 0.6  # 60% importancia
                features['rating_value'] = min(rating_val / 5.0, 1.0) * 0.5  # 50% max
                
                # Bonus por rating alto pero moderado
                if rating_val >= 4.5:
                    features['excellent_rating'] = 0.3
                elif rating_val >= 4.0:
                    features['good_rating'] = 0.2
            except (ValueError, TypeError):
                pass
        
        # 3. FEATURES DE RELEVANCIA (bajo peso)
        if hasattr(product, 'category'):
            cat_lower = str(product.category).lower()
            features['has_category'] = 0.3
            
            # Match de categoría con query
            if 'car' in query.lower() and any(x in cat_lower for x in ['auto', 'vehicle']):
                features['category_match'] = 0.4
            elif 'book' in query.lower() and 'book' in cat_lower:
                features['category_match'] = 0.4
        
        # 4. FEATURES DE PRECIO (muy bajo peso)
        if hasattr(product, 'price') and product.price:
            features['has_price'] = 0.2
            try:
                price_val = float(product.price)
                if 10 <= price_val <= 500:
                    features['reasonable_price'] = 0.1
            except:
                pass
        
        return features
    
    def learn_from_human_feedback(self, clicked_product, query, position, reward=1.0):
        """
        True Human Feedback Learning: Aprende de PREFERENCIAS, no solo matches
        
        Principios:
        1. Aprender más de productos clickeados en posiciones bajas (descubrimiento)
        2. Aprender menos de productos obvios (top 1)
        3. Balancear matches vs calidad
        """
        self.feedback_count += 1
        
        # Extraer features SEMÁNTICAS
        features = self.extract_semantic_features(clicked_product, query)
        
        # ESTRATEGIA DE REWARD BASADA EN POSICIÓN
        if position == 1:
            # Producto obvio: menos learning (ya está bien rankeado)
            position_factor = 0.5
        elif position <= 3:
            # Productos top: learning moderado
            position_factor = 0.8
        elif position <= 10:
            # Productos descubiertos: HIGH learning (esto es importante!)
            position_factor = 1.2
        elif position <= 30:
            # Productos profundos: VERY HIGH learning (descubrimiento valioso)
            position_factor = 1.5
        else:
            # Productos muy abajo: máximo learning
            position_factor = 2.0
        
        adjusted_reward = reward * position_factor
        
        # Registrar par query-producto
        if hasattr(clicked_product, 'id'):
            key = f"{query}_{clicked_product.id}"
            self.query_product_pairs[key] += 1
            
            # Si es click repetido, más reward
            repeat_bonus = min(self.query_product_pairs[key] * 0.2, 1.0)
            adjusted_reward *= (1.0 + repeat_bonus)
        
        # APRENDIZAJE BALANCEADO
        learning_updates = []
        
        for feature_name, feature_value in features.items():
            if isinstance(feature_value, (int, float)) and feature_value > 0:
                # Ajuste base
                adjustment = adjusted_reward * feature_value * self.learning_rate
                
                # APLICAR BALANCE match/rating
                if 'match' in feature_name.lower():
                    # Matches: aplicar balance (no demasiado)
                    adjustment *= min(self.match_rating_balance, 3.0)  # Máximo 3x
                elif 'rating' in feature_name.lower():
                    # Rating: mantener peso completo
                    adjustment *= 1.0
                elif 'category' in feature_name.lower():
                    # Categoría: peso moderado
                    adjustment *= 0.8
                elif 'price' in feature_name.lower():
                    # Precio: bajo peso
                    adjustment *= 0.3
                else:
                    # Otras features: peso medio
                    adjustment *= 0.5
                
                old_weight = self.feature_weights[feature_name]
                self.feature_weights[feature_name] += adjustment
                new_weight = self.feature_weights[feature_name]
                
                learning_updates.append((feature_name, old_weight, new_weight, adjustment))
                self.feature_counts[feature_name] += 1
        
        # Feature específica para este producto en esta query
        if hasattr(clicked_product, 'id'):
            specific_feature = f"preference_{hash(query) % 1000}_{hash(clicked_product.id) % 1000}"
            self.feature_weights[specific_feature] += adjusted_reward * 0.3
        
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
            'top_features': self._get_top_features(3)
        })
        
        logger.info(f"📚 Human Feedback #{self.feedback_count}: '{query[:20]}...' pos {position}")
        
        return adjusted_reward
    
    def _normalize_weights_soft(self):
        """Normalización suave para evitar explosión pero mantener aprendizaje"""
        if not self.feature_weights:
            return
        
        max_weight = max(abs(w) for w in self.feature_weights.values())
        
        # Solo normalizar si es muy grande
        if max_weight > 10.0:
            scale = 8.0 / max_weight
            for key in self.feature_weights:
                self.feature_weights[key] *= scale
    
    def rank_with_human_preferences(self, products, query, baseline_scores=None):
        """Ranking que respeta preferencias humanas aprendidas"""
        if not self.has_learned or not self.feature_weights:
            return self._baseline_ranking(products, query, baseline_scores)
        
        logger.info(f"   🧠 Aplicando Human Preferences ({len(self.feature_weights)} features)")
        
        # Calcular scores humanos
        human_scores = []
        
        for i, product in enumerate(products):
            # Score base del baseline (70% peso)
            if baseline_scores and i < len(baseline_scores):
                base_score = baseline_scores[i] / max(baseline_scores) if max(baseline_scores) > 0 else 0
            else:
                base_score = (len(products) - i) / len(products)  # Fallback
            
            # Score de preferencias humanas (30% peso máximo)
            human_score = 0.0
            
            # 1. Features semánticas
            features = self.extract_semantic_features(product, query)
            for feature_name, feature_value in features.items():
                if feature_name in self.feature_weights:
                    weight = self.feature_weights[feature_name]
                    
                    # Aplicar según tipo
                    if 'match' in feature_name.lower():
                        human_score += feature_value * weight * 0.4  # 40% de features
                    elif 'rating' in feature_name.lower():
                        human_score += feature_value * weight * 0.3  # 30%
                    else:
                        human_score += feature_value * weight * 0.2  # 20%
            
            # 2. Preferencia específica query-producto
            if hasattr(product, 'id'):
                specific_feature = f"preference_{hash(query) % 1000}_{hash(product.id) % 1000}"
                if specific_feature in self.feature_weights:
                    human_score += self.feature_weights[specific_feature] * 0.1
            
            # Limitar human_score a máximo 0.3
            human_score = min(human_score, 0.3)
            
            # Score combinado: 70% baseline, 30% human preferences
            combined_score = (base_score * 0.7) + human_score
            
            human_scores.append(combined_score)
        
        # Ordenar
        sorted_indices = np.argsort(human_scores)[::-1]
        ranked_products = [products[i] for i in sorted_indices]
        
        return ranked_products
    
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
    
    def _get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Obtiene las top n características aprendidas"""
        if not self.feature_weights:
            return []
        
        sorted_weights = sorted(
            self.feature_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_weights[:n]
    
    def get_stats(self):
        """Obtiene estadísticas detalladas"""
        stats = {
            'feedback_count': self.feedback_count,
            'weights_count': len(self.feature_weights),
            'has_learned': self.has_learned,
            'unique_query_product_pairs': len(self.query_product_pairs),
            'top_features': self._get_top_features(15),
            'feature_type_distribution': {}
        }
        
        if self.feature_weights:
            # Distribución por tipo
            type_counts = defaultdict(float)
            for feature, weight in self.feature_weights.items():
                if 'match' in feature.lower():
                    type_counts['match'] += abs(weight)
                elif 'rating' in feature.lower():
                    type_counts['rating'] += abs(weight)
                elif 'category' in feature.lower():
                    type_counts['category'] += abs(weight)
                elif 'price' in feature.lower():
                    type_counts['price'] += abs(weight)
                elif 'preference_' in feature:
                    type_counts['specific_preference'] += abs(weight)
                else:
                    type_counts['other'] += abs(weight)
            
            stats['feature_type_distribution'] = dict(type_counts)
            
            # Ratio match/rating
            match_total = type_counts.get('match', 0)
            rating_total = type_counts.get('rating', 0)
            stats['match_vs_rating_ratio'] = match_total / rating_total if rating_total > 0 else float('inf')
            
            # Health check
            stats['health'] = {
                'has_enough_feedback': self.feedback_count >= 20,
                'balanced_learning': 0.5 <= stats['match_vs_rating_ratio'] <= 3.0,
                'has_diverse_features': len(self.feature_weights) >= 10,
                'learning_active': len(self.learning_history) > 0
            }
        
        return stats
    
    def get_learning_health(self):
        """Diagnóstico de salud del aprendizaje"""
        stats = self.get_stats()
        
        health = {
            'score': 0,
            'issues': [],
            'recommendations': []
        }
        
        # Puntos por métricas saludables
        if stats['feedback_count'] >= 20:
            health['score'] += 2
        else:
            health['issues'].append(f"Poco feedback: {stats['feedback_count']}")
            health['recommendations'].append("Obtener más clicks reales")
        
        ratio = stats.get('match_vs_rating_ratio', 0)
        if 0.5 <= ratio <= 3.0:
            health['score'] += 2
        else:
            health['issues'].append(f"Ratio match/rating desbalanceado: {ratio:.2f}")
            health['recommendations'].append("Ajustar match_rating_balance en constructor")
        
        if len(stats['feature_type_distribution']) >= 3:
            health['score'] += 1
        else:
            health['issues'].append("Features poco diversas")
            health['recommendations'].append("Variar más las búsquedas")
        
        # Calificación
        if health['score'] >= 4:
            health['status'] = "✅ Saludable"
        elif health['score'] >= 2:
            health['status'] = "⚠️  Mejorable"
        else:
            health['status'] = "❌ Problemático"
        
        return health
    
    def save(self, path):
        """Guarda el ranker"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """Carga un ranker"""
        with open(path, 'rb') as f:
            return pickle.load(f)