from __future__ import annotations
import logging
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import hashlib
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.core.data.user_models import UserProfile
from src.core.data.product import Product

logger = logging.getLogger(__name__)

class CollaborativeFilter:
    """
    Filtro colaborativo para recomendaciones cruzadas entre usuarios similares
    """
    
    def __init__(self, user_manager, min_similarity: float = 0.6, use_ml_features: bool = False):
        self.user_manager = user_manager
        self.min_similarity = min_similarity
        self.use_ml_features = use_ml_features
        
        # Cache
        self.positive_feedback_cache: Dict[str, Dict[str, float]] = {}
        self.cache_ttl = 3600
        self.last_cache_update: Dict[str, float] = {}
        
        # Pesos base e híbridos
        self.base_weights = {'collaborative': 0.6, 'rag': 0.4}
        self.hybrid_weights = self.base_weights.copy()
        
        # 🔥 NUEVO: historial para ajustes dinámicos
        self.weight_history = []
        self.performance_history = []
        
        # 🔥 NUEVO: Modelo ML para embeddings (se inicializa cuando sea necesario)
        self.ml_model: Optional[Any] = None
        self.product_embeddings_cache: Dict[str, np.ndarray] = {}
        
        print(f"[CollaborativeFilter] Pesos iniciales: {self.hybrid_weights}")
        print(f"[CollaborativeFilter] ML Features habilitado: {use_ml_features}")

    def _initialize_ml_model(self, embeddings: List[np.ndarray]) -> None:
        """Inicializa el modelo ML para búsqueda de vecinos cercanos"""
        if not self.use_ml_features or not embeddings:
            return
            
        try:
            # Usar NearestNeighbors para búsqueda de productos similares basados en embeddings
            self.ml_model = NearestNeighbors(
                n_neighbors=min(50, len(embeddings)),
                metric='cosine',
                algorithm='auto'
            )
            
            # Ajustar el modelo con los embeddings disponibles
            embeddings_array = np.array(embeddings)
            self.ml_model.fit(embeddings_array)
            logger.info("✅ Modelo ML inicializado para Collaborative Filter")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando modelo ML: {e}")
            self.ml_model = None

    def _calculate_ml_collaborative_score(self, product: Product, similar_users: List[UserProfile]) -> float:
        """
        Calcula score colaborativo usando embeddings y features ML
        """
        if not self.use_ml_features or not similar_users:
            return 0.0
            
        try:
            ml_score = 0.0
            total_weight = 0.0
            
            # Verificar si el producto tiene embedding
            if not hasattr(product, 'embedding') or not product.embedding:
                return 0.0
                
            product_embedding = np.array(product.embedding)
            
            for user in similar_users:
                # Obtener embeddings de productos que el usuario ha preferido
                user_preferred_embeddings = self._get_user_preferred_embeddings(user)
                
                if user_preferred_embeddings:
                    # Calcular similitud promedio con productos preferidos por el usuario
                    similarities = []
                    for user_embedding in user_preferred_embeddings:
                        similarity = self._cosine_similarity(product_embedding, user_embedding)
                        similarities.append(similarity)
                    
                    avg_similarity = np.mean(similarities) if similarities else 0.0
                    
                    # Ponderar por similitud del usuario con el usuario objetivo
                    user_weight = 0.5  # Peso base, se podría ajustar según similitud de perfiles
                    ml_score += avg_similarity * user_weight
                    total_weight += user_weight
            
            # Normalizar el score
            final_score = ml_score / total_weight if total_weight > 0 else 0.0
            
            # Aplicar transformación sigmoide para suavizar el score
            return self._sigmoid(final_score * 3)  # Escalar y aplicar sigmoide
            
        except Exception as e:
            logger.debug(f"Error calculando score ML para producto {getattr(product, 'id', 'unknown')}: {e}")
            return 0.0

    def _get_user_preferred_embeddings(self, user: UserProfile) -> List[np.ndarray]:
        """
        Obtiene embeddings de productos que el usuario ha preferido (feedback positivo)
        """
        embeddings = []
        
        try:
            # Buscar productos con feedback positivo del usuario
            for feedback in user.feedback_history:
                if feedback.rating >= 4:  # Feedback positivo
                    # Intentar obtener el producto y su embedding
                    product_id = feedback.selected_product
                    
                    # Usar cache de embeddings para evitar múltiples consultas
                    if product_id in self.product_embeddings_cache:
                        embeddings.append(self.product_embeddings_cache[product_id])
                    else:
                        # Aquí necesitarías una forma de obtener el producto por ID
                        # Esto dependerá de tu arquitectura
                        # Por ahora, devolvemos lista vacía si no hay cache
                        pass
                        
        except Exception as e:
            logger.debug(f"Error obteniendo embeddings preferidos del usuario {user.user_id}: {e}")
        
        return embeddings

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcula similitud coseno entre dos vectores"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return dot_product / (norm1 * norm2)
        except:
            return 0.0

    def _sigmoid(self, x: float) -> float:
        """Función sigmoide para normalizar scores"""
        return 1 / (1 + np.exp(-x))

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
            
            # 🔥 NUEVO: Aplicar scoring ML si está habilitado
            if self.use_ml_features and candidate_products and similar_users:
                collaborative_scores = self._apply_ml_scoring(
                    collaborative_scores, candidate_products, similar_users
                )

            return collaborative_scores

        except Exception as e:
            logger.error(f"Error en filtro colaborativo: {e}")
            return {}

    def _apply_ml_scoring(self, collaborative_scores: Dict[str, float], 
                         candidate_products: List[Product], 
                         similar_users: List[UserProfile]) -> Dict[str, float]:
        """
        Aplica scoring ML a los scores colaborativos existentes
        """
        enhanced_scores = collaborative_scores.copy()
        
        for product in candidate_products:
            product_id = getattr(product, 'id', None)
            if not product_id:
                continue
                
            # Calcular score ML
            ml_score = self._calculate_ml_collaborative_score(product, similar_users)
            
            # Mezclar con score colaborativo existente
            if product_id in enhanced_scores:
                # Combinación ponderada: 70% colaborativo tradicional, 30% ML
                enhanced_scores[product_id] = enhanced_scores[product_id] * 0.7 + ml_score * 0.3
            else:
                # Si no hay score colaborativo tradicional, usar solo ML (con peso reducido)
                enhanced_scores[product_id] = ml_score * 0.3
        
        return enhanced_scores

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
        
    def adjust_weights_dynamically(
            self,
            user_id: str = None,
            query: str = None,
            rag_results: list = None,
            collab_results: list = None
        ) -> Dict[str, float]:

        # Valores base
        rag_confidence = 0.5
        collab_confidence = 0.5

        # 1) Calidad de la consulta → RAG
        if query:
            rag_confidence = self._estimate_query_suitability_for_rag(query)

        # 2) Adecuación del usuario → colaborativo
        if user_id and self.user_manager:
            collab_confidence = self._estimate_user_suitability_for_collab(user_id)

        # 3) Calidad de resultados
        if rag_results is not None:
            rag_confidence *= self._evaluate_results_quality(rag_results, query)

        if collab_results is not None:
            collab_confidence *= self._evaluate_results_quality(collab_results, query)

        # 4) Historial de performance
        if self.performance_history:
            recent = self.performance_history[-10:]

            rag_success_rate = sum(1 for p in recent if p.get("rag_effective")) / len(recent)
            collab_success_rate = sum(1 for p in recent if p.get("collab_effective")) / len(recent)

            rag_confidence = rag_confidence * 0.7 + rag_success_rate * 0.3
            collab_confidence = collab_confidence * 0.7 + collab_success_rate * 0.3

        # 5) Suavizado
        smoothing = 0.4
        new_rag_weight = rag_confidence * smoothing + self.hybrid_weights["rag"] * (1 - smoothing)
        new_collab_weight = collab_confidence * smoothing + self.hybrid_weights["collaborative"] * (1 - smoothing)

        # Normalizar entre 0.1 y 0.9
        total = new_rag_weight + new_collab_weight
        if total > 0:
            self.hybrid_weights = {
                "rag": max(0.1, min(0.9, new_rag_weight / total)),
                "collaborative": max(0.1, min(0.9, new_collab_weight / total)),
            }
        else:
            self.hybrid_weights = self.base_weights.copy()

        # Guardar en historial
        from datetime import datetime
        self.weight_history.append({
            "timestamp": datetime.now().isoformat(),
            "weights": self.hybrid_weights.copy(),
            "rag_confidence": rag_confidence,
            "collab_confidence": collab_confidence,
            "user_id": user_id,
            "query_preview": query[:50] if query else None
        })

        if len(self.weight_history) > 100:
            self.weight_history = self.weight_history[-100:]

        print(f"[CollaborativeFilter] Pesos ajustados: "
            f"RAG={self.hybrid_weights['rag']:.2f}, "
            f"Collaborative={self.hybrid_weights['collaborative']:.2f}")

        return self.hybrid_weights

    def _evaluate_results_quality(self, results: list, query: str) -> float:
        """
        Evalúa si los resultados parecen relevantes.
        Retorna un valor 0.1 a 1.0
        """
        if not results:
            return 0.2  # mala calidad

        score = 0.5  # base

        # Si la mayoría tiene query en el título (si existe)
        if query:
            q = query.lower().split()
            matches = 0
            total = 0

            for r in results:
                title = getattr(r, "title", "").lower()
                if title:
                    total += 1
                    if any(word in title for word in q):
                        matches += 1

            if total > 0:
                score += (matches / total) * 0.4

        return max(0.1, min(1.0, score))

    
    def _estimate_rag_confidence(self, query: str) -> float:
        """Estima confianza del componente RAG basado en la consulta."""
        if not query:
            return 0.5
        
        # Factores que aumentan confianza en RAG:
        confidence = 0.5  # Base
        
        # Consultas largas y específicas son mejores para RAG
        if len(query.split()) > 3:
            confidence += 0.2
        
        # Consultas con términos técnicos o específicos
        technical_terms = ['características', 'especificaciones', 'comparar', 'mejor', 
                          'recomendar', 'qué es', 'cómo funciona']
        if any(term in query.lower() for term in technical_terms):
            confidence += 0.15
        
        # Limitar entre 0.1 y 0.9
        return max(0.1, min(0.9, confidence))
    
    def _estimate_collab_confidence(self, user_id: str) -> float:
        """Estima confianza del filtro colaborativo basado en el usuario."""
        if not user_id or not self.user_manager:
            return 0.3
        
        try:
            # Obtener usuarios similares
            similar_users = self.user_manager.get_similar_users(user_id, k=5)
            
            # Más usuarios similares = mayor confianza
            if len(similar_users) >= 3:
                return 0.8
            elif len(similar_users) >= 1:
                return 0.6
            else:
                return 0.4
        except:
            return 0.3
    
    def get_weights(self, user_id: str = None, query: str = None) -> Dict[str, float]:
        """Obtiene pesos (ajustados dinámicamente si se proporciona contexto)."""
        if user_id or query:
            return self.adjust_weights_dynamically(user_id, query)
        return self.hybrid_weights
      
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
