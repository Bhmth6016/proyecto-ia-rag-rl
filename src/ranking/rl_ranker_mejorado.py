# src/ranking/rl_ranker_mejorado.py
"""
RL Ranker MEJORADO que aprende características REALES de productos
"""
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class RLHFRankerMejorado:
    """RL Ranker que realmente aprende de características de productos"""
    
    def __init__(self, alpha: float = 0.3, temperature: float = 1.0):  # Alpha más alto
        self.alpha = alpha  # Tasa de aprendizaje más alta
        self.temperature = temperature
        
        # Pesos aprendidos para características REALES
        self.feature_weights = defaultdict(float)
        
        # Estadísticas
        self.feedback_count = 0
        self.has_learned = False
        self.learning_history = []
        
        logger.info(f"🤖 RLHFRanker MEJORADO (alpha={alpha})")
    
    def extract_product_features_for_learning(self, product, query: str = "") -> Dict[str, float]:
        """Extrae características REALES para aprendizaje RL"""
        features = {}
        
        # 1. Características de rating (MUY IMPORTANTE)
        if hasattr(product, 'rating') and product.rating:
            try:
                rating_val = float(product.rating)
                features['has_rating'] = 1.0
                features['rating_value'] = rating_val / 5.0
                if rating_val >= 4.0:
                    features['high_rating'] = 1.0
                if rating_val >= 4.5:
                    features['excellent_rating'] = 1.0
            except (ValueError, TypeError):
                pass
        
        # 2. Match con query (LO MÁS IMPORTANTE)
        if query and hasattr(product, 'title'):
            query_lower = query.lower()
            title_lower = product.title.lower() if product.title else ""
            
            # Palabras exactas en común
            query_words = set(query_lower.split())
            title_words = set(title_lower.split())
            
            match_count = len(query_words.intersection(title_words))
            features['exact_match_count'] = min(match_count / 5.0, 1.0)  # Normalizado
            
            if query_words:
                match_ratio = match_count / len(query_words)
                features['title_match_ratio'] = min(match_ratio, 1.0)
                
                # Bonus por match alto
                if match_ratio >= 0.8:
                    features['near_perfect_match'] = 1.0
            
            # Match exacto de frase completa
            if query_lower in title_lower:
                features['exact_phrase_match'] = 1.0
                features['perfect_match'] = 1.0
        
        # 3. Características de categoría
        if hasattr(product, 'category') and product.category:
            features['has_category'] = 1.0
            # Detectar categorías específicas
            cat_lower = str(product.category).lower()
            if any(word in cat_lower for word in ['auto', 'car', 'vehicle']):
                features['is_automotive'] = 1.0
            elif any(word in cat_lower for word in ['beauty', 'cosmetic', 'makeup']):
                features['is_beauty'] = 1.0
            elif any(word in cat_lower for word in ['electronic', 'computer', 'phone']):
                features['is_electronics'] = 1.0
        
        # 4. Características de precio
        if hasattr(product, 'price') and product.price:
            try:
                price_val = float(product.price)
                if price_val > 0:
                    features['has_price'] = 1.0
                    # Normalizar precio (asumiendo < $1000)
                    features['price_normalized'] = min(price_val / 1000.0, 1.0)
                    if price_val < 50:
                        features['low_price'] = 1.0
                    elif price_val > 200:
                        features['high_price'] = 1.0
            except (ValueError, TypeError):
                pass
        
        # 5. Características de reviews
        if hasattr(product, 'review_count') and product.review_count:
            try:
                review_count = int(product.review_count)
                features['has_reviews'] = 1.0
                if review_count > 100:
                    features['many_reviews'] = 1.0
                if review_count > 1000:
                    features['popular_item'] = 1.0
            except (ValueError, TypeError):
                pass
        
        return features
    
    def learn_from_feedback(
        self,
        clicked_product,  # Producto clickeado REAL
        query: str,       # Query REAL
        position: int,    # Posición REAL
        reward: float = 1.0
    ):
        """APRENDE de feedback REAL con características REALES - VERSIÓN CORREGIDA"""
        self.feedback_count += 1
        
        # 1. Extraer características REALES del producto clickeado
        clicked_features = self.extract_product_features_for_learning(clicked_product, query)
        
        if not clicked_features:
            logger.warning("⚠️ No se pudieron extraer características del producto clickeado")
            return
        
        # 2. Ajustar reward basado en posición (posiciones bajas = menos reward)
        position_factor = 1.0 / (1.0 + np.log1p(position))  # 1->1.0, 2->0.63, 5->0.41
        adjusted_reward = reward * position_factor
        
        # 3. Aprender de CADA característica (ESTO ES CLAVE)
        learning_updates = []
        
        for feature_name, feature_value in clicked_features.items():
            if isinstance(feature_value, (int, float)) and feature_value > 0:
                # Cuánto ajustar este peso
                adjustment = adjusted_reward * feature_value * self.alpha
                
                # CORRECCIÓN: Declarar old_weight ANTES de usarlo
                old_weight = self.feature_weights[feature_name]
                
                # Boost para características CRÍTICAS
                if 'match' in feature_name.lower() or 'exact' in feature_name.lower():
                    adjustment *= 3.0  # TRIPLE importancia para matches
                elif 'rating' in feature_name.lower():
                    adjustment *= 2.0  # DOBLE importancia para rating
                elif 'perfect' in feature_name.lower():
                    adjustment *= 4.0  # CUÁDRUPLE importancia para match perfecto
                
                # Aplicar ajuste
                self.feature_weights[feature_name] += adjustment
                new_weight = self.feature_weights[feature_name]
                
                learning_updates.append((feature_name, old_weight, new_weight, adjustment))
        
        # 4. Normalizar periódicamente para evitar explosión
        if self.feedback_count % 10 == 0:
            self._normalize_weights()
        
        # 5. Marcar como aprendido si hay suficiente feedback diverso
        if self.feedback_count >= 10 and len(self.feature_weights) >= 5:
            self.has_learned = True
        
        # 6. Guardar en historial
        self.learning_history.append({
            'feedback_count': self.feedback_count,
            'query': query,
            'position': position,
            'adjusted_reward': adjusted_reward,
            'features_learned': list(clicked_features.keys()),
            'top_features': self.get_top_features(3)
        })
        
        # 7. Loggear aprendizaje
        logger.info(f"📚 RL aprendió de feedback #{self.feedback_count}:")
        logger.info(f"   Query: '{query[:30]}...', Posición: {position}, Reward ajustado: {adjusted_reward:.2f}")
        logger.info(f"   Características extraídas: {len(clicked_features)}")
        
        if learning_updates:
            top_update = max(learning_updates, key=lambda x: abs(x[3]))
            logger.info(f"   Mayor ajuste: {top_update[0]} = {top_update[3]:+.3f}")
    
    def _normalize_weights(self):
        """Normaliza pesos para mantenerlos en rango razonable"""
        if not self.feature_weights:
            return
        
        max_weight = max(abs(w) for w in self.feature_weights.values())
        if max_weight > 5.0:  # Solo normalizar si es muy grande
            scale = 3.0 / max_weight
            for key in self.feature_weights:
                self.feature_weights[key] *= scale
    
    def rank_products(
        self, 
        products: List, 
        query: str, 
        baseline_scores: Optional[List[float]] = None
    ) -> List:
        """Rankea productos usando aprendizaje REAL - VERSIÓN CORREGIDA"""
        if not self.has_learned or not self.feature_weights:
            logger.info("   → RL sin aprendizaje suficiente, usando baseline")
            # Fallback: ordenar por baseline scores o similitud
            if baseline_scores and len(baseline_scores) == len(products):
                sorted_indices = np.argsort(baseline_scores)[::-1]
                return [products[i] for i in sorted_indices]
            return products
        
        logger.info(f"   → Aplicando RL con {len(self.feature_weights)} características aprendidas")
        
        # COPIAMOS el baseline ranking inicial
        if baseline_scores and len(baseline_scores) == len(products):
            baseline_ranked = [products[i] for i in np.argsort(baseline_scores)[::-1]]
        else:
            baseline_ranked = products
        
        # Solo ajustar TOP 20 productos (no todos)
        top_n = min(20, len(baseline_ranked))
        top_products = baseline_ranked[:top_n]
        rest_products = baseline_ranked[top_n:]
        
        scores = []
        
        for i, product in enumerate(top_products):
            # Score base de similitud (mantener orden original)
            base_score = (top_n - i) / top_n  # Score decreciente según posición
            
            # Extraer características de este producto
            features = self.extract_product_features_for_learning(product, query)
            
            # Calcular RL boost de manera MÁS CONSERVADORA
            rl_boost = 0.0
            
            # PRIORIZAR características de MATCH sobre rating
            for feature_name, feature_value in features.items():
                if feature_name in self.feature_weights and isinstance(feature_value, (int, float)):
                    weight = self.feature_weights[feature_name]
                    
                    # Ajustar impacto basado en tipo de característica
                    if 'match' in feature_name.lower() or 'exact' in feature_name.lower():
                        # MATCH es lo más importante - aplicar peso completo
                        rl_boost += feature_value * weight * 0.3
                    elif 'rating' in feature_name.lower():
                        # Rating es secundario - aplicar peso reducido
                        rl_boost += feature_value * weight * 0.1
                    else:
                        # Otras características - impacto mínimo
                        rl_boost += feature_value * weight * 0.05
            
            # Aplicar temperatura para suavizar ajustes
            rl_boost = rl_boost * (1.0 / self.temperature)
            
            # Limitar boost máximo
            rl_boost = np.clip(rl_boost, -0.5, 0.5)
            
            scores.append(base_score + rl_boost)
        
        # Ordenar solo los top productos por scores ajustados
        top_indices = np.argsort(scores)[::-1]
        reordered_top = [top_products[i] for i in top_indices]
        
        # Mantener el resto sin cambios
        ranked_products = reordered_top + rest_products
        
        # Solo mover productos significativamente mejorados hacia arriba
        # pero mantener la coherencia general del ranking
        return ranked_products
    
    def apply_rl_correction(self, baseline_products: List, query: str) -> List:
        """Aplica corrección RL de manera CONSERVADORA al baseline"""
        if not self.has_learned or len(self.feature_weights) < 3:
            return baseline_products
        
        # Tomar solo top 20 del baseline
        n_products = min(20, len(baseline_products))
        top_products = baseline_products[:n_products]
        
        # Calcular scores RL para cada producto
        rl_scores = []
        for product in top_products:
            score = 0.0
            
            # Extraer características
            features = self.extract_product_features_for_learning(product, query)
            
            # Aplicar pesos aprendidos con PARÁMETROS CONSERVADORES
            for feature_name, feature_value in features.items():
                if feature_name in self.feature_weights:
                    weight = self.feature_weights[feature_name]
                    
                    # Aplicar según tipo de característica
                    if 'perfect' in feature_name.lower() or 'exact_phrase' in feature_name.lower():
                        # Match perfecto: alta prioridad
                        score += feature_value * weight * 0.4
                    elif 'match' in feature_name.lower():
                        # Otros matches: prioridad media
                        score += feature_value * weight * 0.2
                    elif 'rating' in feature_name.lower():
                        # Rating: prioridad baja
                        score += feature_value * weight * 0.1
                    else:
                        # Otras: prioridad muy baja
                        score += feature_value * weight * 0.05
            
            rl_scores.append(score)
        
        # Ordenar por scores RL, pero solo si hay diferencias significativas
        if max(rl_scores) - min(rl_scores) > 0.1:  # Umbral mínimo
            sorted_indices = np.argsort(rl_scores)[::-1]
            reordered_top = [top_products[i] for i in sorted_indices]
            
            # Combinar con el resto
            ranked_products = reordered_top + baseline_products[n_products:]
            
            # Loggear cambios solo si son significativos
            if list(top_products) != reordered_top:
                logger.info(f"   🔀 RL ajustó ranking para '{query[:30]}...'")
                
            return ranked_products
        
        # Sin cambios significativos, mantener baseline
        return baseline_products
    
    def get_top_features(self, n: int = 5) -> List[tuple]:
        """Obtiene las top n características aprendidas"""
        sorted_weights = sorted(
            self.feature_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_weights[:n]
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del aprendizaje"""
        stats = {
            'feedback_count': self.feedback_count,
            'has_learned': self.has_learned,
            'features_learned_count': len(self.feature_weights),
            'top_features': self.get_top_features(10)
        }
        
        # Análisis de tipos de features
        if self.feature_weights:
            match_features = [f for f in self.feature_weights.keys() if 'match' in f.lower()]
            rating_features = [f for f in self.feature_weights.keys() if 'rating' in f.lower()]
            price_features = [f for f in self.feature_weights.keys() if 'price' in f.lower()]
            category_features = [f for f in self.feature_weights.keys() if any(x in f.lower() for x in ['category', 'auto', 'beauty', 'electronic'])]
            
            stats['match_features_count'] = len(match_features)
            stats['rating_features_count'] = len(rating_features)
            stats['price_features_count'] = len(price_features)
            stats['category_features_count'] = len(category_features)
        
        return stats
    
    def save(self, path: str):
        """Guarda el ranker entrenado"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str):
        """Carga un ranker entrenado"""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)