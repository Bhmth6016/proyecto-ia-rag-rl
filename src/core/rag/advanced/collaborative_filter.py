from __future__ import annotations
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import time
from src.core.data.user_models import UserProfile
from src.core.data.product import Product

logger = logging.getLogger(__name__)

class CollaborativeFilter:
    """
    Filtro colaborativo para recomendaciones cruzadas entre usuarios similares
    """
    
    def __init__(self, user_manager, min_similarity: float = 0.6):
        self.user_manager = user_manager
        self.min_similarity = min_similarity
        self.positive_feedback_cache: Dict[str, Dict[str, float]] = {}
        self.cache_ttl = 3600
        self.last_cache_update: Dict[str, float] = {}
        self.hybrid_weights = {'collaborative': 0.6, 'rag': 0.4}
        
    def _get_positive_feedback_scores(self, user: UserProfile, query: str) -> Dict[str, float]:
        """Obtiene SOLO feedback positivo con cache robusto"""
        # ✅ CACHE KEY ROBUSTO
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:12]
        cache_key = f"{user.user_id}_{query_hash}"
        current_time = time.time()
        
        # Verificar cache
        if (cache_key in self.positive_feedback_cache and 
            current_time - self.last_cache_update.get(cache_key, 0) < self.cache_ttl):
            return self.positive_feedback_cache[cache_key]
        
        feedback_scores = {}
        
        try:
            relevant_feedback = self._find_relevant_positive_feedback(user, query)
            
            for feedback in relevant_feedback:
                product_id = feedback.get('selected_product_id')
                if product_id:
                    rating = feedback.get('rating', 3)
                    if rating >= 4:  # Solo feedback positivo
                        normalized_score = (rating - 1) / 4.0
                        recency_weight = self._calculate_recency_weight(feedback.get('timestamp'))
                        query_similarity = feedback.get('query_similarity', 0.5)
                        
                        final_score = normalized_score * recency_weight * query_similarity
                        
                        if final_score > 0.4:
                            feedback_scores[product_id] = max(
                                feedback_scores.get(product_id, 0),
                                final_score
                            )
            
            # Actualizar cache
            self.positive_feedback_cache[cache_key] = feedback_scores
            self.last_cache_update[cache_key] = current_time
            
        except Exception as e:
            logger.debug(f"Error obteniendo feedback positivo de {user.user_id}: {e}")
        
        return feedback_scores

    def get_collaborative_scores(
            self,
            user_or_profile,
            query_or_candidates,
            target_user: Optional[UserProfile] = None,
            query: Optional[str] = None,
            candidate_products: Optional[List[Product]] = None,
            max_similar_users: int = 8):
        
        # 🔥 CORRECCIÓN: Manejar caso cuando user_or_profile es None
        if user_or_profile is None:
            return {}
            
        # ---- CASO 1: llamado externo (DeepEval) ----
        if isinstance(user_or_profile, str) and isinstance(query_or_candidates, list):
            user_id = user_or_profile
            product_ids = query_or_candidates

            try:
                # 🔥 CORRECCIÓN: Manejar caso cuando user_manager no está disponible
                if self.user_manager is None:
                    return {}
                    
                user = self.user_manager.get_user_profile(user_id)
            except Exception as e:
                logger.debug(f"No se pudo obtener perfil de usuario {user_id}: {e}")
                return {}

            if user is None:
                return {}

        # ---- CASO 2: llamado interno ----
        else:
            user = user_or_profile
            target_user = user_or_profile
            query = query_or_candidates

        # ahora sigue tu lógica normal...
        try:
            similar_users = self.user_manager.find_similar_users(
                target_user, self.min_similarity
            )[:max_similar_users]

            collaborative_scores = {}

            if similar_users:
                collaborative_scores = self._get_collaborative_from_similar_users(
                    target_user, similar_users, query, candidate_products
                )

            if not collaborative_scores:
                collaborative_scores = self._get_category_fallback_scores(
                    target_user, query, candidate_products
                )

            return collaborative_scores

        except Exception as e:
            logger.error(f"Error en filtro colaborativo: {e}")
            return {}

    def _get_collaborative_from_similar_users(self, target_user, similar_users, query, candidate_products):
        """Lógica colaborativa con usuarios similares"""
        collaborative_scores = defaultdict(float)
        similarity_weights = defaultdict(float)
        
        for similar_user in similar_users:
            similarity = target_user.calculate_similarity(similar_user)
            
            positive_feedback = self._get_positive_feedback_scores(similar_user, query)
            
            for product_id, feedback_score in positive_feedback.items():
                collaborative_scores[product_id] += feedback_score * similarity
                similarity_weights[product_id] += similarity
        
        return self._normalize_with_quality_filter(collaborative_scores, similarity_weights)

    def _get_category_fallback_scores(self, user: UserProfile, query: str, products: List[Product]) -> Dict[str, float]:
        """
        Fallback: da scores basados en categorías preferidas del usuario
        """
        fallback_scores = {}
        
        try:
            user_categories = set(user.preferred_categories)
            
            for product in products:
                score = 0.0
                
                # Verificar categoría del producto
                product_category = getattr(product, 'main_category', '').lower()
                product_title = getattr(product, 'title', '').lower()
                
                # Score por categoría preferida
                if any(cat in product_category for cat in user_categories):
                    score += 0.3
                
                # Score por términos en título
                query_terms = set(query.lower().split())
                title_terms = set(product_title.split())
                common_terms = query_terms & title_terms
                
                if common_terms:
                    score += len(common_terms) * 0.1
                
                # Score por rating alto
                rating = getattr(product, 'average_rating', 0)
                if rating and rating >= 4.0:
                    score += 0.2
                
                if score > 0:
                    fallback_scores[product.id] = min(score, 1.0)
            
            logger.debug(f"🔄 Fallback categórico: {len(fallback_scores)} productos con score")
            return fallback_scores
            
        except Exception as e:
            logger.debug(f"Error en fallback categórico: {e}")
            return {}
    
    def _get_user_feedback_scores(self, user: UserProfile, query: str) -> Dict[str, float]:
        """
        Obtiene scores de feedback de un usuario para productos relevantes a la query
        """
        feedback_scores = {}
        
        try:
            # Buscar en historial de feedback del usuario
            relevant_feedback = self._find_relevant_feedback(user, query)
            
            for feedback in relevant_feedback:
                product_id = feedback.get('selected_product_id')
                if product_id:
                    # Convertir rating 1-5 a score 0-1
                    rating = feedback.get('rating', 3)
                    normalized_score = (rating - 1) / 4.0  # 1→0, 5→1
                    
                    # Ponderar por antigüedad (feedback reciente vale más)
                    recency_weight = self._calculate_recency_weight(feedback.get('timestamp'))
                    
                    feedback_scores[product_id] = max(
                        feedback_scores.get(product_id, 0),
                        normalized_score * recency_weight
                    )
        
        except Exception as e:
            logger.debug(f"Error obteniendo feedback de usuario {user.user_id}: {e}")
        
        return feedback_scores
    
    def _find_relevant_feedback(self, user: UserProfile, query: str) -> List[Dict]:
        """
        Encuentra feedback relevante basado en similitud de queries
        """
        relevant_feedback = []
        query_terms = set(query.lower().split())
        
        for feedback_event in user.feedback_history:
            feedback_query = feedback_event.query.lower()
            feedback_terms = set(feedback_query.split())
            
            # Calcular similitud de Jaccard entre queries
            intersection = len(query_terms & feedback_terms)
            union = len(query_terms | feedback_terms)
            
            if union > 0:
                similarity = intersection / union
                if similarity > 0.3:  # Umbral de similitud
                    relevant_feedback.append({
                        'selected_product_id': feedback_event.selected_product,
                        'rating': feedback_event.rating,
                        'timestamp': feedback_event.timestamp,
                        'query_similarity': similarity
                    })
        
        # Ordenar por similitud y luego por recencia
        relevant_feedback.sort(key=lambda x: (
            x['query_similarity'], 
            x['timestamp']
        ), reverse=True)
        
        return relevant_feedback[:5]  # Top 5 más relevantes
    
    def _calculate_recency_weight(self, timestamp: datetime) -> float:
        """
        Calcula peso basado en recencia (feedback reciente vale más)
        """
        try:
            days_ago = (datetime.now() - timestamp).days
            # Feedback de última semana: peso 1.0, después decae exponencialmente
            if days_ago <= 7:
                return 1.0
            elif days_ago <= 30:
                return 0.7
            elif days_ago <= 90:
                return 0.4
            else:
                return 0.2
        except:
            return 0.5  # Peso por defecto
    
    def update_hybrid_weights(self, user: UserProfile, success_rate: float):
        """
        Ajusta pesos híbridos basado en rendimiento histórico del usuario
        """
        try:
            # Si el usuario tiene buen historial con colaborativo, aumentar peso
            if success_rate > 0.7:  # 70% de éxito
                self.hybrid_weights['collaborative'] = min(0.8, self.hybrid_weights['collaborative'] + 0.1)
                self.hybrid_weights['rag'] = max(0.2, self.hybrid_weights['rag'] - 0.1)
            elif success_rate < 0.3:  # 30% de éxito
                self.hybrid_weights['collaborative'] = max(0.2, self.hybrid_weights['collaborative'] - 0.1)
                self.hybrid_weights['rag'] = min(0.8, self.hybrid_weights['rag'] + 0.1)
                
            logger.info(f"🔧 Pesos híbridos ajustados: Collaborative={self.hybrid_weights['collaborative']}, RAG={self.hybrid_weights['rag']}")
            
        except Exception as e:
            logger.debug(f"Error ajustando pesos híbridos: {e}")
            
    def _normalize_with_quality_filter(self, collaborative_scores: Dict[str, float], 
                                 similarity_weights: Dict[str, float]) -> Dict[str, float]:
        """Normaliza scores aplicando filtros de calidad"""
        normalized_scores = {}
        
        for product_id, total_score in collaborative_scores.items():
            weight = similarity_weights[product_id]
            
            # Filtros de calidad
            if (weight > 0.3 and                    # Mínima evidencia agregada
                total_score > 0.2):                 # Score mínimo absoluto
                
                normalized_score = total_score / weight
                
                # Solo incluir scores de alta calidad
                if normalized_score > 0.5:          # Umbral de calidad
                    normalized_scores[product_id] = normalized_score
        
        return normalized_scores
    def _find_relevant_positive_feedback(self, user: UserProfile, query: str) -> List[Dict]:
        """Encuentra feedback relevante y positivo basado en similitud de queries"""
        relevant_feedback = []
        query_terms = set(query.lower().split())
        
        for feedback_event in user.feedback_history:
            # Solo considerar feedback positivo
            if feedback_event.rating < 4:
                continue
                
            feedback_query = feedback_event.query.lower()
            feedback_terms = set(feedback_query.split())
            
            # Calcular similitud de Jaccard entre queries
            intersection = len(query_terms & feedback_terms)
            union = len(query_terms | feedback_terms)
            
            if union > 0:
                similarity = intersection / union
                if similarity > 0.3:  # Umbral de similitud
                    relevant_feedback.append({
                        'selected_product_id': feedback_event.selected_product,
                        'rating': feedback_event.rating,
                        'timestamp': feedback_event.timestamp,
                        'query_similarity': similarity
                    })
        
        # Ordenar por similitud y luego por recencia
        relevant_feedback.sort(key=lambda x: (
            x['query_similarity'], 
            x['timestamp']
        ), reverse=True)
        
        return relevant_feedback[:8]  # Top 8 más relevantes
