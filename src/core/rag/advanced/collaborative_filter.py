from __future__ import annotations
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta

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
        self.feedback_cache = {}  # Cache de feedback agregado
        
    def get_collaborative_scores(self, 
                            target_user: UserProfile,
                            query: str,
                            candidate_products: List[Product],
                            max_similar_users: int = 10) -> Dict[str, float]:
        """
        Calcula scores colaborativos con fallback inteligente
        """
        try:
            # 1. Encontrar usuarios similares
            similar_users = self.user_manager.find_similar_users(
                target_user, 
                self.min_similarity
            )[:max_similar_users]
            
            if not similar_users:
                logger.debug("No se encontraron usuarios similares, usando fallback basado en categorías")
                return self._get_category_fallback_scores(target_user, query, candidate_products)
            
            logger.info(f"👥 Encontrados {len(similar_users)} usuarios similares para {target_user.user_id}")
            
            # 2. Agregar feedback de usuarios similares
            collaborative_scores = defaultdict(float)
            similarity_weights = defaultdict(float)
            
            for similar_user in similar_users:
                similarity = target_user.calculate_similarity(similar_user)
                
                # 3. Obtener feedback relevante del usuario similar
                user_feedback_scores = self._get_user_feedback_scores(similar_user, query)
                
                # 4. Combinar scores ponderados por similitud
                for product_id, feedback_score in user_feedback_scores.items():
                    collaborative_scores[product_id] += feedback_score * similarity
                    similarity_weights[product_id] += similarity
            
            # 5. Si no hay scores colaborativos, usar fallback
            if not collaborative_scores:
                logger.debug("No hay feedback colaborativo disponible, usando fallback")
                return self._get_category_fallback_scores(target_user, query, candidate_products)
            
            # 6. Normalizar scores
            normalized_scores = {}
            for product_id, total_score in collaborative_scores.items():
                if similarity_weights[product_id] > 0:
                    normalized_scores[product_id] = total_score / similarity_weights[product_id]
            
            logger.info(f"📊 Scores colaborativos calculados para {len(normalized_scores)} productos")
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error en filtro colaborativo, usando fallback: {e}")
            return self._get_category_fallback_scores(target_user, query, candidate_products)

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