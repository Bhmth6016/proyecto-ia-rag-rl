# src/ranking/rl_ranker.py
"""
RL Ranker - Aprende ranking BASADO EN features y feedback
PRINCIPIO: Solo reordena productos basado en características y feedback
VIOLA: Nunca menciona o modifica sistemas de recuperación
"""

'''
PRINCIPIOS DE ARQUITECTURA:
1. Este módulo SOLO reordena productos basado en características
2. NO modifica sistemas de recuperación de información
3. NO afecta índices o representaciones vectoriales
4. Solo ajusta scores de ranking basado en feedback
5. NO menciona FAISS, embeddings, vector_store, retrieval o index
'''

import numpy as np
from typing import List, Dict, Any, Optional
import logging
import sys
import os

# Añadir src al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# Importaciones sin referencias a sistemas
try:
    from data.canonicalizer import CanonicalProduct
    from ranking.baseline_ranker import BaselineRanker
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.canonicalizer import CanonicalProduct
    from ranking.baseline_ranker import BaselineRanker


class RLHFRanker:
    """
    Aprende ranking BASADO EN:
    1. Features de productos (no modifica productos)
    2. Feedback de usuario (no modifica representaciones)
    3. Reordena resultados (no filtra ni crea)
    
    PRINCIPIO FUNDAMENTAL: Solo afecta ordenamiento
    """
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.baseline_ranker = BaselineRanker()
        self.learned_policy = {}  # Política aprendida
        self.feedback_history = []
        self.has_learned = False
        
        # Bandit contextual simple
        self.arm_weights = {}  # feature -> weight
        
        logger.info(f"🤖 RLHFRanker inicializado (alpha={alpha})")
        logger.info("   PRINCIPIO: Solo afecta ordenamiento basado en características")
    
    def get_baseline_scores(
        self, 
        products: List[CanonicalProduct], 
        similarity_scores: List[float]
    ) -> List[CanonicalProduct]:
        """
        Ordena productos usando scores de similitud básicos
        SOLO reordena, no calcula ni modifica representaciones
        
        Args:
            products: Lista de productos candidatos
            similarity_scores: Scores de similitud ya calculados
        
        Returns:
            Lista de productos rankeados
        """
        if not products:
            return []
        
        if len(products) != len(similarity_scores):
            logger.warning(f"Mismatch entre productos ({len(products)}) y scores ({len(similarity_scores)})")
            # Fallback: orden aleatorio
            return products
        
        try:
            # Ordenar por scores descendentes (solo reordenamiento)
            sorted_pairs = sorted(zip(products, similarity_scores), key=lambda x: x[1], reverse=True)
            
            # Asignar scores a los productos (solo para referencia)
            for i, (product, score) in enumerate(sorted_pairs):
                product.baseline_score = score
            
            ranked_products = [product for product, _ in sorted_pairs]
            
            logger.info(f"📊 Baseline scores: {len(ranked_products)} productos rankeados")
            if ranked_products and similarity_scores:
                logger.info(f"   Score range: {max(similarity_scores):.4f} - {min(similarity_scores):.4f}")
            
            return ranked_products
            
        except Exception as e:
            logger.error(f"❌ Error en ordenamiento baseline: {e}")
            return products
    
    def rank_with_features_only(
        self, 
        products: List[CanonicalProduct], 
        query_features: Dict[str, Any],
        product_features: List[Dict[str, float]],
        baseline_scores: Optional[List[float]] = None
    ) -> List[CanonicalProduct]:
        """
        Ranking SOLO con features heurísticas (sin RL)
        Este es el MODO 2: Features-only
        
        Args:
            products: Lista de productos candidatos
            query_features: Features extraídas de la query
            product_features: Features de cada producto
            baseline_scores: Scores iniciales opcionales
        
        Returns:
            Lista de productos rankeados
        """
        if not products:
            return []
        
        if len(products) != len(product_features):
            logger.warning(f"Mismatch entre productos ({len(products)}) y features ({len(product_features)})")
            return products
        
        # Pesos optimizados para features heurísticas
        feature_weights = {
            "content_similarity": 0.25,      # Similitud semántica
            "title_similarity": 0.20,        # Similitud en título
            "category_exact_match": 0.25,    # Match exacto de categoría
            "rating_normalized": 0.15,       # Rating normalizado
            "price_available": 0.05,         # Tiene precio
            "has_brand": 0.05,               # Tiene marca
            "description_length": 0.05       # Longitud de descripción
        }
        
        scores = []
        
        for idx, (product, features) in enumerate(zip(products, product_features)):
            score = 0.0
            
            # Comenzar con baseline score si está disponible
            if baseline_scores and idx < len(baseline_scores):
                score = baseline_scores[idx] * 0.3  # Peso menor para baseline
            
            # Aplicar pesos a características disponibles
            for feature_name, weight in feature_weights.items():
                feature_value = features.get(feature_name, 0.0)
                if isinstance(feature_value, (int, float)):
                    score += feature_value * weight
                else:
                    # Si no es numérico, convertir
                    try:
                        score += float(feature_value) * weight
                    except:
                        pass
            
            # Bonus por match semántico fuerte
            if features.get("content_similarity", 0) > 0.7:
                score += 0.1
            
            # Bonus por match de categoría exacto
            if query_features.get('category') and hasattr(product, 'category'):
                if query_features['category'].lower() == product.category.lower():
                    score += 0.15
            
            # Bonus por rating alto
            if hasattr(product, 'rating') and product.rating:
                if product.rating > 4.0:
                    score += 0.08
                elif product.rating > 3.0:
                    score += 0.04
            
            # Penalizar sin precio
            if not hasattr(product, 'price') or not product.price:
                score -= 0.05
            
            scores.append((idx, score))
        
        # Ordenar por score descendente
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Reconstruir lista ordenada CON SCORES ASIGNADOS
        ranked_products = []
        for idx, score in scores:
            product = products[idx]
            product.similarity = score  # Asignar score real
            ranked_products.append(product)
        
        logger.info(f"📊 Modo Features-only: {len(ranked_products)} productos rankeados")
        if scores:
            logger.info(f"   Score range: {scores[0][1]:.3f} - {scores[-1][1]:.3f}")
            if ranked_products and hasattr(ranked_products[0], 'title'):
                logger.info(f"   Top producto: {ranked_products[0].title[:50]}...")
        
        return ranked_products
    
    def rank_with_learning(
        self, 
        products: List[CanonicalProduct], 
        query_features: Dict[str, Any],
        product_features: List[Dict[str, float]],
        baseline_scores: Optional[List[float]] = None,
        user_feedback: Optional[Dict] = None
    ) -> List[CanonicalProduct]:
        """
        Aplica ranking con aprendizaje RL
        
        PRINCIPIO: Solo reordena, nunca:
        - Modifica representaciones
        - Cambia sistemas
        - Filtra arbitrariamente
        """
        # PRINCIPIO: Si no hay aprendizaje, usar features-only
        if not self.has_learned or len(self.learned_policy) == 0:
            logger.info("   → No hay aprendizaje RL disponible, usando features-only")
            return self.rank_with_features_only(
                products, query_features, product_features, baseline_scores
            )
        
        # 1. Obtener ranking baseline
        baseline_ranking = self.baseline_ranker.rank(
            products, query_features, product_features
        )
        
        # 2. Aplicar aprendizaje RL si hay feedback aprendido
        logger.info("   → Aplicando política aprendida RL")
        learned_ranking = self._apply_learned_policy(
            products, baseline_ranking, query_features, product_features, baseline_scores
        )
        
        # Verificación de principios
        original_ids = set(p.id for p in products if hasattr(p, 'id'))
        learned_ids = set(p.id for p in learned_ranking if hasattr(p, 'id'))
        
        if original_ids != learned_ids:
            logger.warning("⚠️  RL intentó modificar conjunto de productos")
            logger.warning("   Revertiendo a ranking features-only")
            return self.rank_with_features_only(
                products, query_features, product_features, baseline_scores
            )
        
        return learned_ranking
    
    def baseline_rank(
        self,
        products: List[CanonicalProduct],
        query_features: Dict[str, Any],
        product_features: List[Dict[str, float]]
    ) -> List[CanonicalProduct]:
        """Ranking baseline sin aprendizaje"""
        scores = self.baseline_ranker.rank(products, query_features, product_features)
        return self._sort_by_scores(products, scores)
    
    def learn_from_feedback(
        self,
        query_features: Dict[str, Any],
        selected_product_id: str,
        reward: float,
        context: Dict[str, Any]
    ):
        """Aprende del feedback (solo afecta política de ranking)"""
        try:
            # PRINCIPIO: Solo aprende de features
            feedback_features = self._extract_feedback_features(
                query_features, selected_product_id, reward, context
            )
            
            # Actualizar pesos de características
            for feature, value in feedback_features.items():
                if feature not in self.arm_weights:
                    self.arm_weights[feature] = 0.0
                
                # Actualizar peso según reward
                self.arm_weights[feature] += reward * value * self.alpha
            
            # Guardar en historial
            self.feedback_history.append({
                'query_features': query_features,
                'selected_product': selected_product_id,
                'reward': reward,
                'feedback_features': feedback_features,
                'timestamp': np.datetime64('now'),
                'learning_principle': 'ranking_only'
            })
            
            # Marcar como aprendido
            self.has_learned = True
            self.learned_policy = self.arm_weights.copy()
            
            logger.info(f"📚 RL aprendió de feedback (reward={reward:.2f})")
            logger.info(f"   Características aprendidas: {len(self.learned_policy)}")
            
        except Exception as e:
            logger.error(f"❌ Error en aprendizaje RL: {e}")
    
    def _apply_learned_policy(
        self,
        products: List[CanonicalProduct],
        baseline_ranking: List[CanonicalProduct],
        query_features: Dict[str, Any],
        product_features: List[Dict[str, float]],
        baseline_scores: Optional[List[float]] = None
    ) -> List[CanonicalProduct]:
        """Aplica política aprendida para reordenar"""
        if not self.learned_policy:
            return self.rank_with_features_only(
                products, query_features, product_features, baseline_scores
            )
        
        # Calcular scores RL basados en características aprendidas
        rl_scores = []
        for i, (product, features) in enumerate(zip(products, product_features)):
            # Comenzar con score baseline si está disponible
            rl_score = baseline_scores[i] if baseline_scores and i < len(baseline_scores) else 0.0
            
            # Aplicar pesos aprendidos a características
            for feature, value in features.items():
                if feature in self.learned_policy:
                    weight = self.learned_policy[feature]
                    # PRINCIPIO: Solo ajusta scores
                    rl_score += weight * value
            
            rl_scores.append(rl_score)
        
        return self._sort_by_scores(products, rl_scores)
    
    def _extract_feedback_features(
        self,
        query_features: Dict[str, Any],
        selected_product_id: str,
        reward: float,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extrae características del feedback para aprendizaje"""
        features = {}
        
        # PRINCIPIO: Solo características de ranking
        
        # Características de la query
        if 'category' in query_features:
            features[f"query_category_{query_features['category']}"] = 1.0
        
        if 'intent' in query_features:
            features[f"query_intent_{query_features['intent']}"] = 1.0
        
        if 'specificity' in query_features:
            specificity = query_features['specificity']
            if isinstance(specificity, dict) and 'specificity_score' in specificity:
                features['query_specificity'] = specificity['specificity_score']
        
        # Características del contexto
        if 'product_position' in context:
            position = context['product_position']
            features[f"position_{min(position, 5)}"] = 1.0
        
        if 'position' in context:
            position = context['position']
            features[f"position_{min(position, 5)}"] = 1.0
        
        # Reward influye en magnitud
        features['reward_magnitude'] = abs(reward)
        
        return features
    
    def _sort_by_scores(
        self, 
        products: List[CanonicalProduct], 
        scores: List[float]
    ) -> List[CanonicalProduct]:
        """Ordena productos por scores (descendente) - SOLO REORDENA"""
        sorted_pairs = sorted(zip(products, scores), key=lambda x: x[1], reverse=True)
        return [product for product, _ in sorted_pairs]
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del aprendizaje"""
        return {
            'has_learned': self.has_learned,
            'feedback_count': len(self.feedback_history),
            'policy_size': len(self.learned_policy),
            'avg_weight': np.mean(list(self.learned_policy.values())) if self.learned_policy else 0.0,
            'last_feedback': self.feedback_history[-1] if self.feedback_history else None,
            'principle': 'ranking_only'
        }
    
    def reset_learning(self):
        """Resetea aprendizaje (útil para experiments)"""
        self.learned_policy = {}
        self.feedback_history = []
        self.arm_weights = {}
        self.has_learned = False
        logger.info("🔄 Aprendizaje RL reseteado")
    
    def get_ranking_modes(self) -> Dict[str, Any]:
        """Devuelve información sobre los modos de ranking disponibles"""
        return {
            'modes': {
                'features_only': 'Características heurísticas',
                'with_rlhf': 'Características + RLHF'
            },
            'current_mode': 'with_rlhf' if self.has_learned else 'features_only',
            'feature_weights': self.learned_policy if self.has_learned else {},
            'features_only_weights': {
                "content_similarity": 0.25,
                "title_similarity": 0.20,
                "category_exact_match": 0.25,
                "rating_normalized": 0.15,
                "price_available": 0.05,
                "has_brand": 0.05,
                "description_length": 0.05
            }
        }