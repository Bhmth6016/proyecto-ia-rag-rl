from __future__ import annotations
from src.core.config import settings
# src/core/rag/advanced/collaborative_filter.py
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
    
    def __init__(self, user_manager, product_service=None, min_similarity: float = 0.6, use_ml_features: bool = False):
        self.user_manager = user_manager
        self.product_service = product_service  # 🔥 NUEVO: Servicio de productos
        self.min_similarity = min_similarity
        self.use_ml_features = use_ml_features  # Puede usar settings.ML_ENABLED
        
        # Cache
        self.positive_feedback_cache: Dict[str, Dict[str, float]] = {}
        self.cache_ttl = 3600
        self.last_cache_update: Dict[str, float] = {}
        
        # 🔥 NUEVO: Cache de embeddings
        self.product_embeddings_cache: Dict[str, np.ndarray] = {}
        
        # Pesos base e híbridos
        self.base_weights = {'collaborative': 0.6, 'rag': 0.4}
        self.hybrid_weights = self.base_weights.copy()
        
        # 🔥 NUEVO: historial para ajustes dinámicos
        self.weight_history = []
        self.performance_history = []
        
        # 🔥 NUEVO: Modelo ML para embeddings (se inicializa cuando sea necesario)
        self.ml_model: Optional[Any] = None
        
        # Precargar embeddings
        self._preload_embeddings()
        
        print(f"[CollaborativeFilter] Pesos iniciales: {self.hybrid_weights}")
        print(f"[CollaborativeFilter] ML Features habilitado: {use_ml_features}")

    def _preload_embeddings(self):
        """Precarga embeddings de productos populares"""
        try:
            if not self.product_service:
                return
            
            # Obtener productos populares (top 100)
            popular_products = self.product_service.get_popular_products(100)
            
            for product in popular_products:
                if hasattr(product, 'embedding') and product.embedding:
                    self.product_embeddings_cache[product.id] = np.array(product.embedding)
            
            logger.info(f"📊 {len(self.product_embeddings_cache)} embeddings precargados")
            
        except Exception as e:
            logger.debug(f"⚠️ Error precargando embeddings: {e}")

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
    
    def _safe_get_product_id(self, feedback) -> Optional[str]:
        """Obtiene product_id de forma segura."""
        try:
            product_id = getattr(feedback, 'selected_product', None)
            
            # Validar que sea un string no vacío
            if product_id and isinstance(product_id, str) and product_id.strip():
                return product_id.strip()
            return None
        except Exception:
            return None
    
    def _get_user_preferred_embeddings(self, user: UserProfile) -> List[np.ndarray]:
        embeddings = []
        
        try:
            for feedback in user.feedback_history:
                if feedback.rating >= 4:
                    
                    # 🔥 USAR LA FUNCIÓN SEGURA para obtener el product_id
                    product_id = self._safe_get_product_id(feedback)
                    if not product_id:
                        continue  # Saltar si no hay ID válido
                    
                    # 1. Buscar en caché
                    if product_id in self.product_embeddings_cache:
                        embeddings.append(self.product_embeddings_cache[product_id])
                        continue
                    
                    # 2. Obtener desde servicio si no está en cache
                    if self.product_service:
                        product = self.product_service.get_product(product_id)
                        if product and hasattr(product, 'embedding') and product.embedding:
                            embedding_array = np.array(product.embedding)
                            self.product_embeddings_cache[product_id] = embedding_array
                            embeddings.append(embedding_array)

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
        
        # -------------------------------------------------
        # 🔥 VALIDACIONES INICIALES
        # -------------------------------------------------
        if user_or_profile is None or user_or_profile == "":
            return {}

        # Si es string → se interpreta como ID externo (DeepEval)
        if isinstance(user_or_profile, str):
            user_id = user_or_profile
            
            # No permitir strings vacíos o inválidos
            if not user_id:
                return {}

            try:
                user = self.user_manager.get_user_profile(user_id)

                if user is None:
                    # Se podría crear perfil temporal, pero solicitud indica devolver vacío
                    logger.debug(f"Usuario {user_id} no encontrado, usando perfil básico → return vacío")
                    return {}

                target_user = user   # Referencia estándar

            except Exception as e:
                logger.debug(f"No se pudo obtener perfil usuario {user_id}: {e}")
                return {}

        # -------------------------------------------------
        # CASO INTERNO (el user ya es perfil real)
        # -------------------------------------------------
        else:
            user = user_or_profile
            target_user = user_or_profile
            query = query_or_candidates  # DeepEval ≠ interno

        # -------------------------------------------------
        # 🔥 LÓGICA PRINCIPAL
        # -------------------------------------------------
        try:
            similar_users = self.user_manager.find_similar_users(
                target_user, self.min_similarity
            )[:max_similar_users]

            collaborative_scores = {}

            # 1) Si hay usuarios similares → filtrado colaborativo normal
            if similar_users:
                collaborative_scores = self._get_collaborative_from_similar_users(
                    target_user, similar_users, query, candidate_products
                )

            # 2) Si no hay resultados → fallback basado en categoría
            if not collaborative_scores:
                collaborative_scores = self._get_category_fallback_scores(
                    target_user, query, candidate_products
                )

            # 3) (Opcional) Re-rank con ML si está habilitado
            if self.use_ml_features and candidate_products and similar_users:
                collaborative_scores = self._apply_ml_scoring(
                    collaborative_scores, candidate_products, similar_users
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
            user_categories = set()
            if user.preferred_categories:
                for cat in user.preferred_categories:
                    # 🔥 CORRECCIÓN: Manejar categorías None o no strings
                    if cat and isinstance(cat, str):
                        user_categories.add(cat.lower())
            
            query_lower = ""
            query_terms = set()
            if query and isinstance(query, str):
                query_lower = query.lower()
                query_terms = set(query_lower.split())
            
            for product in products:
                score = 0.0
                
                # 🔥 CORRECCIÓN: Manejar main_category None
                product_category = getattr(product, 'main_category', '')
                if product_category is None:
                    product_category = ''
                if isinstance(product_category, str):
                    product_category = product_category.lower()
                else:
                    product_category = str(product_category).lower() if product_category else ''
                
                # 🔥 CORRECCIÓN: Manejar title None
                product_title = getattr(product, 'title', '')
                if product_title is None:
                    product_title = ''
                if isinstance(product_title, str):
                    product_title = product_title.lower()
                else:
                    product_title = str(product_title).lower() if product_title else ''
                
                # Score por categoría preferida
                if user_categories and product_category:
                    if any(cat in product_category for cat in user_categories):
                        score += 0.3
                
                # Score por términos en título
                if query_terms and product_title:
                    title_terms = set(product_title.split())
                    common_terms = query_terms & title_terms
                    
                    if common_terms:
                        score += len(common_terms) * 0.1
                
                # Score por rating alto
                rating = getattr(product, 'average_rating', 0)
                if rating and rating >= 4.0:
                    score += 0.2
                
                if score > 0 and hasattr(product, 'id'):
                    fallback_scores[product.id] = min(score, 1.0)
            
            logger.debug(f"🔄 Fallback categórico: {len(fallback_scores)} productos con score")
            return fallback_scores
            
        except Exception as e:
            logger.debug(f"Error en fallback categórico: {e}")
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

    def _estimate_query_suitability_for_rag(self, query: str) -> float:
        """Estima qué tan adecuada es la query para RAG"""
        if not query:
            return 0.5
        
        rag_confidence = 0.5
        
        # Consultas largas y específicas son mejores para RAG
        if len(query.split()) > 3:
            rag_confidence += 0.2
        
        # Consultas con términos técnicos o específicos
        technical_terms = ['características', 'especificaciones', 'comparar', 'mejor', 
                          'recomendar', 'qué es', 'cómo funciona']
        
        # 🔥 CORRECCIÓN: Verificar que query no sea None antes de usar .lower()
        query_lower = query.lower() if query else ""
        if any(term in query_lower for term in technical_terms):
            rag_confidence += 0.15
        
        return max(0.1, min(0.9, rag_confidence))

    def _estimate_user_suitability_for_collab(self, user_id: str) -> float:
        """Estima qué tan adecuado es el usuario para filtro colaborativo"""
        try:
            user = self.user_manager.get_user_profile(user_id)
            if not user:
                return 0.3
            
            # Usuarios con más historial son mejores para colaborativo
            feedback_count = len(user.feedback_history)
            if feedback_count > 20:
                return 0.8
            elif feedback_count > 10:
                return 0.6
            elif feedback_count > 5:
                return 0.5
            else:
                return 0.4
        except:
            return 0.3

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
                title = getattr(r, "title", "")
                # 🔥 CORRECCIÓN: Manejar title None
                if title:
                    title_lower = title.lower()
                    total += 1
                    if any(word in title_lower for word in q):
                        matches += 1

            if total > 0:
                score += (matches / total) * 0.4

        return max(0.1, min(1.0, score))
    
    def get_weights(self, user_id: str = None, query: str = None) -> Dict[str, float]:
        """Obtiene pesos (ajustados dinámicamente si se proporciona contexto)."""
        if user_id or query:
            return self.adjust_weights_dynamically(user_id, query)
        return self.hybrid_weights
      
    def _get_positive_feedback_scores(self, user: UserProfile, query: str) -> Dict[str, float]:
        """Obtiene SOLO feedback positivo con cache robusto"""
        # 🔥 CORRECCIÓN: Manejar query None
        query_safe = query if query else ""
        query_hash = hashlib.md5(query_safe.lower().encode()).hexdigest()[:12]
        cache_key = f"{user.user_id}_{query_hash}"
        current_time = time.time()
        
        # Verificar cache
        if (cache_key in self.positive_feedback_cache and 
            current_time - self.last_cache_update.get(cache_key, 0) < self.cache_ttl):
            return self.positive_feedback_cache[cache_key]
        
        feedback_scores = {}
        
        try:
            relevant_feedback = self._find_relevant_positive_feedback(user, query_safe)
            
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
    
    def _find_relevant_positive_feedback(self, user: UserProfile, query: str) -> List[Dict]:
        """Encuentra feedback relevante y positivo basado en similitud de queries"""
        relevant_feedback = []
        
        # 🔥 CORRECCIÓN: Manejar query None
        if not query:
            return relevant_feedback
            
        query_terms = set(query.lower().split())
        
        for feedback_event in user.feedback_history:
            # Solo considerar feedback positivo
            if feedback_event.rating < 4:
                continue
            
            # 🔥 CORRECCIÓN: Manejar feedback_event.query None
            feedback_query = getattr(feedback_event, 'query', '')
            if feedback_query:
                feedback_query = feedback_query.lower()
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
    
    def _calculate_recency_weight(self, timestamp: datetime) -> float:
        """
        Calcula peso basado en recencia (feedback reciente vale más)
        """
        try:
            if not timestamp:
                return 0.5
                
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