from __future__ import annotations

# src/core/rag/advanced/WorkingRAGAgent.py
import time
import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
from collections import deque
import google.generativeai as genai
# Local imports
# EN LA SECCIÓN DE IMPORTS, AÑADE:
from src.core.data.product_reference import ProductReference
from src.core.data.user_manager import UserManager
from src.core.rag.advanced.collaborative_filter import CollaborativeFilter
from src.core.data.user_models import UserProfile, Gender
from src.core.rag.advanced.RLHFMonitor import RLHFMonitor
from src.core.init import get_system
from src.core.rag.basic.retriever import Retriever
from src.core.data.product import Product
from src.core.config import settings
from src.core.rag.advanced.feedback_processor import FeedbackProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class RAGConfig:
    enable_reranking: bool = True
    enable_rlhf: bool = True
    max_retrieved: int = 50
    max_final: int = 5
    memory_window: int = 3
    domain: str = "videojuegos"
    use_advanced_features: bool = True

@dataclass
class RAGResponse:
    answer: str
    products: List[Product]
    quality_score: float
    retrieved_count: int
    used_llm: bool = False

    @property
    def text(self):
        return self.answer
    @property
    def recommended(self):
        return self.recommended_ids
    @property
    def recommended_ids(self):
        return self.products

    
# ===============================
# Memory
# ===============================
class ConversationBufferMemory:
    def __init__(self, max_length: int = 3):
        self.memory: deque = deque(maxlen=max_length)

    def add(self, query: str, answer: str):
        clean_answer = answer[:200] + "..." if len(answer) > 200 else answer
        self.memory.append({"query": query, "answer": clean_answer})

    def get_context(self) -> str:
        if not self.memory:
            return ""
        return "\n".join([f"Q: {m['query']}\nA: {m['answer']}" for m in self.memory])

    def clear(self):
        self.memory.clear()

# ===============================
# Advanced Evaluator
# ===============================
class AdvancedEvaluator:
    def evaluate_response(self, query: str, response: str, products: List[Product]) -> float:
        """Evaluación avanzada para respuestas de gaming"""
        if not response:
            return 0.0
            
        if not products:
            # Respuesta sin productos pero con sugerencias útiles
            if len(response) > 100 and any(word in response.lower() for word in ['sugerencias', 'prueba', 'intenta']):
                return 0.3
            return 0.0
        
        # Score basado en múltiples factores optimizados para gaming
        content_score = min(1.0, len(response) / 400)
        products_score = min(1.0, len(products) / 3)
        relevance_score = self._calculate_gaming_relevance(query, response, products)
        structure_score = self._calculate_structure_score(response)
        
        final_score = (content_score * 0.25 + 
                      products_score * 0.35 + 
                      relevance_score * 0.25 + 
                      structure_score * 0.15)
        
        return round(final_score, 2)
    
    def _calculate_gaming_relevance(self, query: str, response: str, products: List[Product]) -> float:
        """Calcula relevancia específica para gaming"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        response_lower = response.lower()
        
        # Palabras clave de gaming
        gaming_keywords = {'juego', 'videojuego', 'playstation', 'xbox', 'nintendo', 'switch', 
                          'ps4', 'ps5', 'consola', 'game', 'gaming', 'edición', 'versión'}
        
        # Coincidencia con consulta
        query_matches = sum(1 for word in query_words if len(word) > 2 and word in response_lower)
        query_relevance = min(1.0, query_matches / max(1, len(query_words)))
        
        # Coincidencia con términos de gaming
        gaming_matches = sum(1 for keyword in gaming_keywords if keyword in response_lower)
        gaming_relevance = min(1.0, gaming_matches / 5)
        
        # Relevancia de productos
        product_relevance = 0.0
        for product in products:
            title = str(getattr(product, 'title', '')).lower()
            if any(word in title for word in query_words if len(word) > 2):
                product_relevance += 0.2
        
        product_relevance = min(1.0, product_relevance)
        
        return (query_relevance + gaming_relevance + product_relevance) / 3
    
    def _calculate_structure_score(self, response: str) -> float:
        """Evalúa la estructura de la respuesta"""
        score = 0.0
        
        # Puntos por buen formato
        if "🎯" in response or "🎮" in response:
            score += 0.3
        if "💰" in response or "⭐" in response:
            score += 0.2
        if "PlayStation" in response or "Xbox" in response or "Nintendo" in response:
            score += 0.2
        if len(response.split('\n')) > 5:  # Bien estructurado
            score += 0.3
            
        return min(1.0, score)

# ===============================
# Gaming RLHF Trainer
# ===============================
class GamingRLHFTrainer:
    def __init__(self):
        self.models_dir = Path("models/rl_models")
        self.models_dir.mkdir(exist_ok=True, parents=True)
    
    def score_product_relevance(self, query: str, product: Product, user_profile: Dict) -> float:
        """Scoring optimizado para productos de gaming"""
        base_score = 0.3
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        title = getattr(product, "title", "").lower()
        category = str(getattr(product, 'main_category', '')).lower()
        price = getattr(product, "price", None)
        rating = getattr(product, "average_rating", 0)
        
        # Plataformas de gaming
        platforms = {
            'playstation': ['playstation', 'ps4', 'ps5', 'ps3'],
            'xbox': ['xbox', 'xbox one', 'xbox series'], 
            'nintendo': ['nintendo', 'switch', 'wii', 'ds'],
            'pc': ['pc', 'computer', 'steam']
        }
        
        # Score por plataforma detectada en query
        for platform, keywords in platforms.items():
            if any(keyword in query_lower for keyword in keywords):
                if any(keyword in title for keyword in keywords):
                    base_score += 0.3
                    break
        
        # Score por coincidencia en título
        title_matches = sum(1 for word in query_words if len(word) > 2 and word in title)
        base_score += title_matches * 0.2
        
        # Score por género de juego
        genres = {
            'acción': ['acción', 'action', 'shooter', 'fps'],
            'aventura': ['aventura', 'adventure', 'rpg', 'rol'],
            'deportes': ['deporte', 'sport', 'fútbol', 'fifa'],
            'estrategia': ['estrategia', 'strategy', 'táctica']
        }
        
        for genre, keywords in genres.items():
            if any(keyword in query_lower for keyword in keywords):
                if any(keyword in title for keyword in keywords):
                    base_score += 0.15
                    break
        
        # Score por precio disponible
        if price is not None:
            base_score += 0.05
            
        # Score por rating alto
        if rating and rating >= 4.0:
            base_score += 0.1
            
        return min(1.0, base_score)

# ===============================
# Main Agent - VERSIÓN FINAL OPTIMIZADA
# ===============================
class WorkingAdvancedRAGAgent:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.system = get_system()
        self.retriever = getattr(self.system, "retriever", Retriever())
        self.llm_model = genai.GenerativeModel('gemini-pro')  
        # 🔥 CAMBIO 1: Inicialización condicional de UserManager y CollaborativeFilter
        self.enable_user_features = bool(self.config and getattr(self.config, "use_advanced_features", False))

        # Clase DummyUserManager para fallback garantizado
        class DummyUserManager:
            def get_user_preferences(self, user_id):
                return {"default": True, "fallback": True}
            
            def track_interaction(self, user_id, product_id, rating):
                print(f"[DummyUserManager] Tracked: user={user_id}, product={product_id}, rating={rating}")
            
            def get_similar_users(self, user_id, k=5):
                return []
            
            def get_recommendations(self, user_id, k=10):
                return []

        # ======================================================
        # 🔥 SIEMPRE tener un user_manager (real o dummy)
        # ======================================================
        if self.enable_user_features:
            try:
                from src.core.user_manager import UserManager
                self.user_manager = UserManager()
                print("✅ UserManager real inicializado")
            except ImportError as e:
                print(f"⚠️ No se pudo importar UserManager real: {e}")
                self.user_manager = DummyUserManager()
                print("⚠️ Usando DummyUserManager como fallback")
        else:
            self.user_manager = DummyUserManager()
            print("✅ UserManager dummy inicializado (use_advanced_features = False)")

        # ======================================================
        # 🔥 CollaborativeFilter: solo se activa si user_manager es real
        # ======================================================
        try:
            if isinstance(self.user_manager, DummyUserManager):
                self.collaborative_filter = None
            else:
                self.collaborative_filter = CollaborativeFilter(self.user_manager)
        except Exception as e:
            print(f"⚠️ Error inicializando CollaborativeFilter: {e}")
            self.collaborative_filter = None

        
        # Componentes optimizados
        self.evaluator = AdvancedEvaluator()
        self.feedback_processor = FeedbackProcessor()
        self.rlhf_trainer = GamingRLHFTrainer()
        
        # Estado del usuario (MANTENER por compatibilidad temporal)
        self.user_profiles: Dict[str, Dict] = {}
        self.user_memory: Dict[str, ConversationBufferMemory] = {}
        
        # Configuración RLHF mejorada
        self.min_feedback_for_retrain = 5
        self.retrain_interval = 3600
        self.last_retrain_time = 0
        self.rlhf_monitor = RLHFMonitor()
        
        # 🔥 NUEVO: Configuración sistema híbrido
        self.hybrid_weights = {
            'collaborative': 0.6,  # Peso para feedback usuarios similares
            'rag': 0.4            # Peso para RAG tradicional
        }
        self.min_similarity_threshold = 0.6  # Similitud mínima entre usuarios
        
        self._check_and_retrain()
        
        logger.info(f"✅ WorkingAdvancedRAGAgent inicializado - Sistema Híbrido Activado")
        
    def _generate_with_llm(self, context: str, query: str, products: List[Product]) -> str:
        prompt = f"""
        Eres un experto en recomendaciones de videojuegos.
        
        CONTEXTO: {context}
        PRODUCTOS: {[p.title for p in products]}
        CONSULTA: {query}
        
        Genera una respuesta útil y atractiva recomendando estos productos.
        Incluye:
        - Títulos y plataformas
        - Precios y ratings cuando estén disponibles  
        - Explicación breve de por qué son relevantes
        - Formato amigable con emojis
        """
        
        try:
            response = self.llm_model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error LLM: {e}")
            return self._generate_advanced_gaming_response(query, products)  # Fallback
        
    def process_query(self, query: str, user_id: str = "default") -> RAGResponse:
        """Procesa consultas de gaming de forma optimizada - VERSIÓN CORREGIDA"""
        try:
            # 🔥 CORRECCIÓN COMPLETA: Debug detallado
            original_query = query
            logger.info(f"🔍 INICIO process_query - Tipo: {type(query)}, Valor: {str(query)[:100]}")

            # Extraer texto de consulta de forma segura
            if hasattr(query, 'lower'):
                query_text = query
            elif isinstance(query, dict):
                query_text = query.get('query', str(query))
            elif isinstance(query, str) and query.strip().startswith('{'):
                try:
                    query_data = json.loads(query)
                    query_text = query_data.get('query', str(query_data))
                except json.JSONDecodeError:
                    query_text = str(query)
            else:
                query_text = str(query)
                
            # 🔥 CORRECCIÓN CRÍTICA: Asegurar que query_text sea string y tenga método lower
            if not hasattr(query_text, 'lower'):
                query_text = str(query_text)
                
            logger.info(f"🔍 Query procesada: '{query_text}'")
                
            # 🔥 CORRECCIÓN: Verificar cada paso del procesamiento
            profile = self.get_or_create_user_profile(user_id)
            memory = self.get_or_create_memory(user_id)

            # Enriquecimiento inteligente para gaming - con manejo de errores
            try:
                enriched_query = self._enrich_gaming_query(query_text, profile)
                logger.info(f"🔍 Query enriquecida: '{enriched_query}'")
            except Exception as e:
                logger.error(f"❌ Error en enriquecimiento: {e}")
                enriched_query = query_text

            # 🔥 CAMBIO: Conversión a ProductReference durante la recuperación
            try:
                raw_candidates = self.retriever.retrieve(enriched_query, k=self.config.max_retrieved)
                logger.info(f"🔍 Candidatos recuperados: {len(raw_candidates)} - Tipo: {type(raw_candidates[0]) if raw_candidates else 'None'}")
                
                # 🔥 CONVERTIR A ProductReference
                candidates = []
                for product in raw_candidates:
                    if hasattr(product, 'id'):
                        # Calcular score si está disponible (ajusta según tu lógica)
                        score = getattr(product, 'score', 0.5)
                        pref = ProductReference.from_product(product, score=score, source="rag")
                        candidates.append(pref)
                    else:
                        candidates.append(product)
            except Exception as e:
                logger.error(f"❌ Error en recuperación: {e}")
                candidates = []

            # 🔥 CORRECCIÓN: Convertir IDs a objetos Product para el filtrado
            product_objects = []
            for candidate in candidates:
                if isinstance(candidate, str):
                    # Es un ID, crear un producto mínimo
                    product_objects.append(Product(id=candidate, title=f"Producto {candidate}"))
                elif isinstance(candidate, ProductReference):
                    # Convertir ProductReference a Product si es necesario
                    product_objects.append(candidate.to_product())
                else:
                    product_objects.append(candidate)

            # Filtrado con manejo de errores
            try:
                relevant_candidates = self._filter_gaming_products(product_objects, query_text)
                logger.info(f"🔍 Candidatos filtrados: {len(relevant_candidates)}")
            except Exception as e:
                logger.error(f"❌ Error en filtrado: {e}")
                relevant_candidates = product_objects

            # Reranking con manejo de errores
            try:
                ranked = self._rerank_with_rlhf(relevant_candidates, query_text, profile)
                logger.info(f"🔍 Productos rerankeados: {len(ranked)}")
            except Exception as e:
                logger.error(f"❌ Error en reranking: {e}")
                ranked = relevant_candidates

            final_products = ranked[:self.config.max_final]
            
            # Generación de respuesta
            context_for_generation = memory.get_context()
            full_context = f"{context_for_generation}\nNueva consulta: {query_text}" if context_for_generation else query_text
            
            response = self._generate_gaming_response(full_context, final_products, query_text)
            quality_score = self.evaluator.evaluate_response(query_text, response, final_products)
            
            # Guardar en memoria
            memory.add(query_text, response)
            
            logger.info(f"✅ Query COMPLETADA: '{query_text}' -> {len(final_products)} productos | Score: {quality_score}")
            
            return RAGResponse(
                answer=response,
                products=final_products,  # 🔥 CORRECCIÓN: Devolver productos completos, no solo IDs
                quality_score=quality_score,
                retrieved_count=len(ranked),
                used_llm=False
            )
            
        except Exception as e:
            logger.error(f"❌ ERROR CRÍTICO en process_query: {e}")
            logger.error(f"❌ Traceback completo:", exc_info=True)
            return RAGResponse(
                answer=self._error_response(str(query), e),
                products=[],
                quality_score=0.0,
                retrieved_count=0
            )
            
    def process_query_with_limit(self, query: str, limit: int = 5) -> List[Dict]:
        """Versión de process_query que acepta limit parameter"""
        results = self.process_query(query)
        return results[:limit] if results else []

    def _enrich_gaming_query(self, query: str, profile: Dict) -> str:
        """Enriquecimiento inteligente + aprendizaje desde feedback"""
        # --- 1) Enriquecimiento original basado en plataformas y géneros ---
        query_lower = query.lower()
        enriched_terms = []
        
        # Detectar y expandir plataformas
        platform_map = {
            'playstation': ['ps4', 'ps5', 'playstation 4', 'playstation 5'],
            'xbox': ['xbox one', 'xbox series x', 'xbox series s'],
            'nintendo': ['switch', 'nintendo switch', 'wii', '3ds'],
            'pc': ['computadora', 'steam', 'epic games']
        }
        
        for platform, expansions in platform_map.items():
            if platform in query_lower:
                enriched_terms.extend(expansions)
        
        # Detectar géneros
        genre_map = {
            'acción': ['shooter', 'fps', 'acción'],
            'aventura': ['rpg', 'rol', 'aventura'],
            'deportes': ['deporte', 'sports', 'fútbol'],
            'estrategia': ['estrategy', 'táctica']
        }
        
        for genre, expansions in genre_map.items():
            if genre in query_lower:
                enriched_terms.extend(expansions)

        # --- 2) Enriquecimiento dinámico con términos aprendidos del feedback ---
        learned_terms = self._get_successful_query_terms()
        if learned_terms:
            enriched_terms.extend(learned_terms)

        # --- 3) Si hay términos para enriquecer, devolver query expandida ---
        if enriched_terms:
            enriched_query = f"{query} {' '.join(enriched_terms)}"
            logger.debug(f"🔍 Query enriquecida con aprendizaje: '{query}' -> '{enriched_query}'")
            return enriched_query

        return query
    
    def _get_successful_query_terms(self) -> List[str]:
        """Extrae términos comunes de consultas con feedback positivo"""
        try:
            success_log = Path("data/feedback/success_queries.log")
            if not success_log.exists():
                return []

            with open(success_log, 'r', encoding='utf-8') as f:
                queries = [json.loads(line).get('query', '') for line in f]

            from collections import Counter
            all_terms = []
            for q in queries:
                all_terms.extend(q.lower().split())

            # Filtrar términos útiles
            common_terms = [
                term for term, count in Counter(all_terms).most_common(5)
                if len(term) > 3 and count > 1
            ]

            return common_terms

        except Exception:
            return []



    def _filter_gaming_products(self, products: List, query: str) -> List:
        """Filtrado inteligente para productos de gaming - VERSIÓN CORREGIDA"""
        if not products:
            return []
        
        # 🔥 CORRECCIÓN: Los productos pueden ser IDs strings u objetos Product
        filtered_products = []
        
        query_lower = query.lower()
        query_words = set(word for word in query_lower.split() if len(word) > 2)
        
        # Términos clave de gaming
        gaming_terms = {'playstation', 'xbox', 'nintendo', 'switch', 'game', 'juego', 
                    'edición', 'versión', 'ps4', 'ps5', 'xbox one', 'gaming'}
        
        for product in products:
            try:
                # 🔥 CORRECCIÓN: Manejar tanto IDs como objetos Product
                if isinstance(product, str):
                    # Es un ID de producto, no podemos filtrar por contenido
                    # Incluirlo y dejar que el reranking lo ordene
                    filtered_products.append(product)
                    continue
                    
                # Es un objeto Product, podemos filtrar por contenido
                title = str(getattr(product, 'title', '')).lower()
                category = str(getattr(product, 'main_category', '')).lower()
                
                # Coincidencia directa en título
                title_match = any(word in title for word in query_words)
                
                # Coincidencia con términos de gaming
                gaming_match = any(term in title for term in gaming_terms)
                
                # Coincidencia en categoría
                category_match = any(word in category for word in query_words)
                
                # Para gaming, ser más permisivo pero priorizar coincidencias
                if title_match or (gaming_match and (category_match or len(query_words) == 0)):
                    filtered_products.append(product)
                    
            except Exception as e:
                # 🔥 CORRECCIÓN: Si hay error, incluir el producto de todos modos
                logger.debug(f"Error filtrando producto: {e}")
                filtered_products.append(product)
        
        # Fallback: si no hay productos filtrados, devolver todos
        if not filtered_products:
            logger.info(f"🔄 Usando {len(products)} productos sin filtrar como fallback")
            return products
        
        return filtered_products

    def _rerank_with_rlhf(self, products: List, query: str, profile: Dict) -> List:
        """Reranking híbrido con soporte RLHF y fallback seguro"""
        if not products:
            return products

        # 🔥 Si el reranking está deshabilitado por configuración, devolver tal cual
        if not getattr(self.config, "enable_reranking", True):
            logger.debug("Reranking deshabilitado por configuración")
            return products

        try:
            # Asegurar que todos los productos son objetos Product
            scorable_products = []
            for product in products:
                if isinstance(product, str):
                    scorable_products.append(Product(id=product, title=f"Producto {product}"))
                else:
                    scorable_products.append(product)

            # Obtener perfil completo del usuario
            user_profile = self._get_or_create_user_profile_demographic(profile.get('user_id'))

            # 1️⃣ Score RAG / RLHF
            rag_scores = {}
            for product in scorable_products:
                if hasattr(self, 'rlhf_trainer') and self.rlhf_trainer is not None:
                    score = self.rlhf_trainer.score_product_relevance(query, product, profile)
                else:
                    # Fallback a scoring simple si no hay RLHF
                    score = 0.5
                rag_scores[getattr(product, 'id', 'unknown')] = score

            # 2️⃣ Score colaborativo
            collaborative_scores = {}
            if (self.collaborative_filter is not None and
                hasattr(self.collaborative_filter, 'get_collaborative_scores')):
                try:
                    collaborative_scores = self.collaborative_filter.get_collaborative_scores(
                        user_profile, query, scorable_products
                    )
                except Exception as e:
                    logger.warning(f"Error en filtro colaborativo: {e}")
                    collaborative_scores = {}

            # 3️⃣ Combinación híbrida con pesos configurables
            hybrid_scores = {}
            for product in scorable_products:
                pid = getattr(product, 'id', 'unknown')
                rag_score = rag_scores.get(pid, 0)
                collab_score = collaborative_scores.get(pid, 0)
                hybrid_score = (
                    self.hybrid_weights.get('collaborative', 0.4) * collab_score +
                    self.hybrid_weights.get('rag', 0.6) * rag_score
                )
                hybrid_scores[pid] = hybrid_score

            # 4️⃣ Ordenar y retornar productos por score híbrido
            scored_products = [(hybrid_scores.get(getattr(p, 'id', 'unknown'), 0), p) for p in scorable_products]
            scored_products.sort(key=lambda x: x[0], reverse=True)

            logger.info(f"🎯 Reranking híbrido: {len([s for s, _ in scored_products if s > 0])} productos con score positivo")
            return [p for _, p in scored_products]

        except Exception as e:
            logger.warning(f"Reranking híbrido falló, usando RAG tradicional: {e}")
            return self._rerank_fallback(products, query, profile)
            
    def _log_rlhf_data(self, query: str, response: List[Dict], score: float, user_id: str = None):
        """Genera datos de entrenamiento para RLHF - VERSIÓN SIMPLIFICADA"""
        try:
            # Crear directorio de feedback si no existe
            feedback_dir = Path("data/feedback")
            feedback_dir.mkdir(parents=True, exist_ok=True)
            
            # Log exitoso
            log_file = feedback_dir / "successful_queries.jsonl"
            log_data = {
                'query': query,
                'response': str(response[:2]),  # Limitar tamaño
                'score': score,
                'user_id': user_id,
                'timestamp': time.time()
            }
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
                
        except Exception as e:
            print(f"⚠️ Error logging RLHF data: {e}")
        
    def _get_or_create_user_profile_demographic(self, user_id: str) -> UserProfile:
        """Obtiene o crea perfil de usuario con datos demográficos (solo si está permitido)"""
        try:
            # 🔥 CAMBIO 3: No crear perfiles demográficos por defecto en cada llamada
            # Si no permitimos features de usuario, devolver perfil temporal no persistente
            if not self.enable_user_features or self.user_manager is None:
                # Crear perfil temporal (UserProfile o similar)
                return UserProfile(
                    user_id=user_id,
                    session_id=user_id,
                    age=25,
                    gender=Gender.MALE,
                    country="Unknown",
                    language="es"
                )

            # Intentar cargar perfil existente (no crear uno nuevo para cada query)
            existing_profile = self.user_manager.get_user_profile(user_id)
            if existing_profile:
                return existing_profile

            # Si no existe y se permiten features, CREAR solo si explícitamente deseado (evitar creación implícita masiva)
            # Para evitar creación por cada query, crear solo si user_id no es "default"
            if user_id and user_id != "default":
                default_profile = self.user_manager.create_user_profile(
                    age=25,
                    gender="male",
                    country="Spain",
                    language="es",
                    preferred_categories=["games", "videojuegos"],
                    preferred_brands=["Sony", "Microsoft", "Nintendo"]
                )
                # 🔥 CAMBIO 6: Reducir logs de creación de usuario
                logger.debug(f"👤 Creado perfil demográfico para {user_id}")
                return default_profile

            # Si user_id == "default", devolver perfil temporal
            return UserProfile(
                user_id=user_id,
                session_id=user_id,
                age=25,
                gender=Gender.MALE,
                country="Unknown",
                language="es"
            )

        except Exception as e:
            logger.error(f"Error obteniendo perfil demográfico: {e}")
            return UserProfile(
                user_id=user_id,
                session_id=user_id,
                age=25,
                gender=Gender.MALE,
                country="Unknown",
                language="es"
            )

    def _rerank_fallback(self, products: List, query: str, profile: Dict) -> List:
        """Fallback a RAG tradicional si el sistema híbrido falla"""
        try:
            # 🔥 CORRECCIÓN: Manejar tanto IDs como objetos Product en el fallback
            scorable_products = []
            for product in products:
                if isinstance(product, str):
                    scorable_products.append(Product(id=product, title=f"Producto {product}"))
                else:
                    scorable_products.append(product)
                    
            scored_products = [(self.rlhf_trainer.score_product_relevance(query, p, profile), p) 
                            for p in scorable_products]
            scored_products.sort(key=lambda x: x[0], reverse=True)
            return [p for _, p in scored_products]
        except Exception as e:
            logger.error(f"Fallback también falló: {e}")
            return products

    def _generate_gaming_response(self, context: str, products: List[Product], original_query: str) -> str:
        """Generación de respuesta optimizada para gaming"""
        if not products:
            return self._no_gaming_results_response(original_query)
        
        return self._generate_with_llm(context, original_query, products)

    def _generate_advanced_gaming_response(self, query: str, products: List[Product]) -> str:
        """Respuesta avanzada para gaming con formato enriquecido"""
        # Agrupar por plataforma de forma inteligente
        platforms = self._categorize_by_platform(products)
        
        lines = [f"🎮 **Recomendaciones de videojuegos para '{query}'**", ""]
        
        total_shown = 0
        max_platforms = 2
        max_games_per_platform = 3
        
        for platform, prods in list(platforms.items())[:max_platforms]:
            # Emoji para plataforma
            platform_emoji = {
                'PlayStation': '📀',
                'Xbox': '🎯', 
                'Nintendo': '🎴',
                'PC': '🖥️',
                'Otras plataformas': '🎪'
            }.get(platform, '🎮')
            
            lines.append(f"{platform_emoji} **{platform}**")
            
            for i, p in enumerate(prods[:max_games_per_platform], 1):
                title = getattr(p, 'title', 'Videojuego sin nombre')
                price = getattr(p, 'price', None)
                rating = getattr(p, 'average_rating', None)
                
                # Formato enriquecido
                price_str = f"💰 ${price:.2f}" if price else "💰 Precio no disponible"
                rating_str = f"⭐ {rating}/5" if rating else "⭐ Sin calificaciones"
                
                lines.append(f"  {i}. **{title}**")
                lines.append(f"     {price_str} | {rating_str}")
                total_shown += 1
            
            lines.append("")
        
        # Información adicional
        if len(products) > total_shown:
            lines.append(f"💡 *Y {len(products) - total_shown} juegos más disponibles...*")
        
        # Sugerencia contextual
        lines.append(self._get_contextual_suggestion(query, products))
        
        return "\n".join(lines)

    def _categorize_by_platform(self, products: List[Product]) -> Dict[str, List[Product]]:
        """Categorización inteligente por plataforma"""
        platforms = {}
        
        for product in products:
            title = str(getattr(product, 'title', '')).lower()
            platform = "Otras plataformas"
            
            # Detección de plataforma mejorada
            if any(term in title for term in ['playstation', 'ps4', 'ps5', 'ps3']):
                platform = "PlayStation"
            elif any(term in title for term in ['xbox', 'xbox one', 'xbox series']):
                platform = "Xbox"
            elif any(term in title for term in ['nintendo', 'switch', 'wii', 'ds']):
                platform = "Nintendo" 
            elif any(term in title for term in ['pc', 'computer', 'steam']):
                platform = "PC"
                
            platforms.setdefault(platform, []).append(product)
        
        return platforms

    def _get_contextual_suggestion(self, query: str, products: List[Product]) -> str:
        """Sugerencias contextuales inteligentes"""
        query_lower = query.lower()
        
        if 'barato' in query_lower or 'económico' in query_lower:
            return "💸 *Tip: Filtra por precio en tu próxima búsqueda*"
        elif 'nuevo' in query_lower or 'lanzamiento' in query_lower:
            return "🆕 *Tip: Busca 'últimos lanzamientos' para novedades*"
        elif any(term in query_lower for term in ['acción', 'shooter', 'fps']):
            return "🔫 *Tip: Prueba 'call of duty' o 'battlefield' para shooters*"
        elif any(term in query_lower for term in ['aventura', 'rpg', 'rol']):
            return "🗺️ *Tip: Prueba 'zelda' o 'final fantasy' para aventuras*"
        elif len(products) < 3:
            return "🔍 *Tip: Prueba términos más generales para más resultados*"
        else:
            return "🎯 *Tip: Especifica plataforma o género para mejores resultados*"

    def _no_gaming_results_response(self, query: str) -> str:
        """Respuesta cuando no hay resultados"""
        suggestions = [
            "Prueba con el nombre específico del juego",
            "Busca por plataforma: 'playstation', 'xbox', 'nintendo', 'pc'", 
            "Intenta con el género: 'acción', 'aventura', 'deportes', 'estrategia'",
            "Usa términos como 'mejor valorado', 'novedades' o 'clásicos'",
            "Verifica la ortografía o usa términos en inglés"
        ]
        
        return f"🎮 **No encontré videojuegos para '{query}'**\n\n**Sugerencias para gaming:**\n" + "\n".join(f"• {s}" for s in suggestions)

    def _error_response(self, query: str, error: Exception) -> str:
        return f"❌ **Error procesando tu búsqueda de '{query}'**\n\nDetalle: {str(error)[:100]}...\n\nPor favor, intenta con otros términos."

    def get_or_create_user_profile(self, user_id: str) -> Dict:
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "user_id": user_id, 
                "preferred_categories": ["games", "videojuegos"],
                "preferred_platforms": [],
                "search_history": [], 
                "purchase_history": [],
                "gaming_preferences": {}
            }
        return self.user_profiles[user_id]

    def get_or_create_memory(self, user_id: str) -> ConversationBufferMemory:
        if user_id not in self.user_memory:
            self.user_memory[user_id] = ConversationBufferMemory(max_length=self.config.memory_window)
        return self.user_memory[user_id]

    def clear_memory(self, user_id: str = "default"):
        """Limpia la memoria del usuario"""
        if user_id in self.user_memory:
            self.user_memory[user_id].clear()

    def _calculate_dynamic_weights(self, collaborative_count: int, total_products: int) -> Dict[str, float]:
        """Calcula pesos dinámicos NORMALIZADOS"""
        if total_products == 0:
            return self.hybrid_weights
        
        collaborative_ratio = collaborative_count / total_products
        
        # Ajustar pesos según evidencia
        if collaborative_ratio < 0.1:  # Poca evidencia
            rag_weight = 0.7
            collab_weight = 0.3
        elif collaborative_ratio < 0.3:  # Evidencia moderada
            rag_weight = 0.4
            collab_weight = 0.6
        else:  # Buena evidencia
            rag_weight = 0.3
            collab_weight = 0.7
        
        # ✅ NORMALIZAR SIEMPRE
        total = rag_weight + collab_weight
        return {
            'rag': rag_weight / total,
            'collaborative': collab_weight / total
        }

    def _infer_selected_product(self, answer: str, rating: int, user_query: str = "") -> Optional[str]:
        """
        Infiere qué producto seleccionó el usuario basado en su respuesta.
        Versión mejorada con múltiples estrategias.
        """
        import re
        import json

        # ==========================================================
        # ESTRATEGIA 1: Buscar ID explícito en formato especial
        # ==========================================================
        explicit_patterns = [
            r'\[PRODUCT:([A-Za-z0-9_\-]+)\]',      # [PRODUCT:ABC123]
            r'\[ID:([A-Za-z0-9_\-]+)\]',           # [ID:ABC123]
            r'Product ID:\s*([A-Za-z0-9_\-]+)',    # Product ID: ABC123
            r'ID\s*:\s*([A-Za-z0-9_\-]+)',         # ID: ABC123
            r'producto\s+([A-Z][A-Z0-9]{4,})',     # producto ABC123
            r'ref[:\s]*([A-Z][A-Z0-9]{4,})',       # ref: ABC123
        ]

        for pattern in explicit_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            if matches:
                selected_id = matches[0]
                print(f"[Feedback] Found explicit product ID via pattern '{pattern}': {selected_id}")
                return selected_id

        # ==========================================================
        # ESTRATEGIA 2: Buscar JSON embebido
        # ==========================================================
        if '{' in answer and '}' in answer:
            try:
                json_start = answer.find('{')
                json_end = answer.rfind('}') + 1
                json_str = answer[json_start:json_end]

                metadata = json.loads(json_str)

                if 'selected_product_id' in metadata:
                    print(f"[Feedback] Found ID in JSON metadata: {metadata['selected_product_id']}")
                    return metadata['selected_product_id']

                if 'product_id' in metadata:
                    print(f"[Feedback] Found ID as 'product_id' in JSON: {metadata['product_id']}")
                    return metadata['product_id']

            except json.JSONDecodeError:
                pass
            except KeyError:
                pass

        # ==========================================================
        # ESTRATEGIA 3: Buscar IDs conocidos en los productos actuales
        # ==========================================================
        if hasattr(self, "current_products") and self.current_products:
            products = self.current_products
        elif hasattr(self, "products") and self.products:
            products = self.products
        else:
            products = []

        if products:
            # Diccionario título → id
            title_to_id = {}
            for product in products:
                if hasattr(product, "title") and hasattr(product, "id"):
                    clean = product.title.lower().strip()
                    if len(clean) > 3:
                        title_to_id[clean] = product.id

            answer_lower = answer.lower()

            # Coincidencia por título completo
            for title, product_id in title_to_id.items():
                if title in answer_lower and len(title) > 5:
                    print(f"[Feedback] Found product by title match '{title[:30]}...': {product_id}")
                    return product_id

            # Coincidencia parcial por palabras clave
            for product in products:
                if hasattr(product, "title") and hasattr(product, "id"):
                    title_words = set(product.title.lower().split())
                    answer_words = set(answer_lower.split())

                    common = title_words.intersection(answer_words)
                    common_filtered = [w for w in common if len(w) > 3]

                    if len(common_filtered) >= 2:
                        print(f"[Feedback] Found product by keyword match: {common_filtered} → {product.id}")
                        return product.id

        # ==========================================================
        # ESTRATEGIA 4: Buscar patrones comunes de posibles IDs
        # ==========================================================
        id_patterns = [
            r'\b([A-Z][A-Z0-9]{3,})\b',
            r'\b([0-9]{4,}[A-Z]?)\b',
            r'\b([A-Z]{2,}-[0-9]{3,})\b',
        ]

        for pattern in id_patterns:
            matches = re.findall(pattern, answer.upper())
            if matches:
                if products:
                    existing = [getattr(p, "id", "") for p in products]
                    for match in matches:
                        if match in existing:
                            print(f"[Feedback] Found ID pattern match '{pattern}': {match}")
                            return match
                else:
                    print(f"[Feedback] Found ID pattern (no verification): {matches[0]}")
                    return matches[0]

        # ==========================================================
        # ESTRATEGIA 5: Si el rating es alto, asumir el primer producto
        # ==========================================================
        if rating >= 4 and products:
            first_id = getattr(products[0], "id", "unknown")
            print(f"[Feedback] Using first product due to high rating ({rating}): {first_id}")
            return first_id

        # ==========================================================
        # ESTRATEGIA 6: Fallback al método padre
        # ==========================================================
        try:
            import inspect
            if hasattr(super(), "_infer_selected_product"):
                result = super()._infer_selected_product(answer, rating, user_query)
                print(f"[Feedback] Using parent class method: {result}")
                return result
        except:
            pass

        # ==========================================================
        # ÚLTIMO RECURSO
        # ==========================================================
        print(f"[Feedback] WARNING: Could not infer product ID from answer")
        print(f"  Answer preview: {answer[:200]}...")
        print(f"  Query: {user_query}")
        print(f"  Rating: {rating}")

        if products:
            default_id = getattr(products[0], "id", "unknown")
            print(f"[Feedback] Using default (first) product ID: {default_id}")
            return default_id

        return "unknown_product_id"


    def _find_best_query_match(self, user_query: str, product_ids: List[str], answer: str) -> Optional[str]:
        """Encuentra el producto que mejor coincide con la query del usuario"""
        try:
            query_terms = set(user_query.lower().split())
            best_match = None
            best_score = 0
            
            # Buscar en la respuesta secciones que mencionen cada producto
            for product_id in product_ids:
                # Buscar contexto alrededor del product_id en la respuesta
                product_context = self._extract_product_context(answer, product_id)
                if product_context:
                    context_terms = set(product_context.lower().split())
                    common_terms = query_terms & context_terms
                    score = len(common_terms) / len(query_terms) if query_terms else 0
                    
                    if score > best_score:
                        best_score = score
                        best_match = product_id
            
            return best_match if best_score > 0.3 else None
            
        except Exception:
            return None

    def _extract_product_context(self, answer: str, product_id: str) -> str:
        """Extrae contexto alrededor de la mención de un producto"""
        try:
            # Buscar líneas que contengan el product_id
            lines = answer.split('\n')
            for i, line in enumerate(lines):
                if product_id in line:
                    # Tomar línea actual y anterior/siguiente
                    start = max(0, i - 1)
                    end = min(len(lines), i + 2)
                    context_lines = lines[start:end]
                    return ' '.join(context_lines)
            return ""
        except Exception:
            return ""

    def log_feedback(self, query: str, answer: str, rating: int, user_id: str = "default"):
        """Log de feedback con inferencia mejorada"""
        try:
            user_profile = self._get_or_create_user_profile_demographic(user_id)
            all_product_ids = self._extract_products_from_response(answer)
            
            # ✅ INFERENCIA MEJORADA
            selected_product_id = self._infer_selected_product(answer, rating, query)
            
            entry = {
                "timestamp": datetime.now().isoformat(), 
                "query": query, 
                "answer": answer, 
                "rating": rating, 
                "user_id": user_id,
                "user_age": user_profile.age,
                "user_gender": user_profile.gender.value,
                "user_country": user_profile.country,
                "products_shown": all_product_ids,
                "selected_product_id": selected_product_id,
                "inference_method": "multi_strategy",
                "domain": self.config.domain
            }
            
            # Guardar y actualizar pesos
            fdir = Path("data/feedback")
            fdir.mkdir(exist_ok=True, parents=True)
            fname = fdir / f"feedback_gaming_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(fname, "a", encoding='utf-8') as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            if selected_product_id:
                self.retriever.update_feedback_weights_immediately(
                    selected_product_id, 
                    rating, 
                    all_product_ids
                )
            
            # Actualizar perfil de usuario
            user_profile.add_feedback_event(
                query=query,
                response=answer,
                rating=rating,
                products_shown=all_product_ids,
                selected_product=selected_product_id
            )
            
            if self.enable_user_features and self.user_manager:
                self.user_manager.save_user_profile(user_profile)
            
            logger.info(f"📝 Feedback: rating {rating} para {user_id} (producto: {selected_product_id})")
            
        except Exception as e:
            logger.error(f"Error registrando feedback: {e}")

    def _extract_products_from_response(self, answer: str) -> List[str]:
        """Extrae IDs de productos mencionados en la respuesta"""
        # Implementación simple - en sistema real usaría regex más sofisticado
        import re
        product_ids = re.findall(r'[A-Z0-9]{10}', answer)
        return product_ids
            
    def _check_and_retrain(self):
        """Verifica y ejecuta reentrenamiento con mejores condiciones"""
        try:
            feedback_count = self._count_recent_feedback()
            has_enough_feedback = feedback_count >= self.min_feedback_for_retrain
            should_retrain_time = (time.time() - self.last_retrain_time) > self.retrain_interval
            
            # ✅ NUEVA CONDICIÓN: también reentrenar si hay nuevo feedback significativo
            has_significant_new_feedback = feedback_count > 0 and self.last_retrain_time == 0
            
            if has_enough_feedback and (should_retrain_time or has_significant_new_feedback):
                logger.info(f"🔁 Iniciando reentrenamiento con {feedback_count} feedbacks")
                success = self._retrain_with_feedback()
                if success:
                    self.last_retrain_time = time.time()
                    logger.info("✅ Reentrenamiento completado exitosamente")
                else:
                    logger.warning("⚠️ Reentrenamiento falló, se reintentará más tarde")
                    
        except Exception as e:
            logger.error(f"❌ Error en reentrenamiento automático: {e}")
    
    def _count_recent_feedback(self) -> int:
        """Cuenta feedback de los últimos 7 días"""
        count = 0
        feedback_dir = Path("data/feedback")
        
        for jsonl_file in feedback_dir.glob("feedback_*.jsonl"):
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        record = json.loads(line)
                        # Verificar si es reciente (últimos 7 días)
                        timestamp = record.get('timestamp', '')
                        if self._is_recent(timestamp, days=7):
                            count += 1
            except:
                continue
                
        return count
    
    def _retrain_with_feedback(self) -> bool:
        """Reentrena el modelo RLHF - versión mejorada"""
        try:
            from .trainer import RLHFTrainer
            
            trainer = RLHFTrainer()
            
            # ✅ BUSCAR ARCHIVOS MÁS FLEXIBLE
            feedback_dir = Path("data/feedback")
            failed_log = feedback_dir / "failed_queries.log"
            success_log = feedback_dir / "success_queries.log"
            
            # ✅ CREAR ARCHIVOS SI NO EXISTEN
            failed_log.parent.mkdir(parents=True, exist_ok=True)
            if not failed_log.exists():
                failed_log.touch()
            if not success_log.exists():
                success_log.touch()
            
            dataset = trainer.prepare_rlhf_dataset_from_logs(failed_log, success_log)
            
            if len(dataset) >= 3:
                start_time = time.time()
                trainer.train(dataset)
                training_time = time.time() - start_time
                
                # 📊 REGISTRAR MÉTRICAS
                self.rlhf_monitor.log_training_session(
                    examples_used=len(dataset),
                    previous_accuracy=0.0,  # TODO: calcular accuracy real
                    new_accuracy=0.1,       # TODO: calcular accuracy real  
                    training_time=training_time
                )
            else:
                logger.info(f"⏳ No suficiente data aún: {len(dataset)}/3 ejemplos")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error en reentrenamiento RLHF: {e}")
            return False
        
    def _is_recent(self, timestamp: str, days: int = 7) -> bool:
        """Verifica si un timestamp es reciente (últimos N días)"""
        try:
            from datetime import datetime, timezone
            record_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)
            time_diff = current_time - record_time
            return time_diff.days <= days
        except:
            return False
        
    def _find_explicit_mention(self, answer: str, product_ids: List[str]) -> Optional[str]:
        """Busca menciones explícitas de productos en la respuesta"""
        try:
            # Buscar patrones como "recomiendo el producto X", "te sugiero Y"
            patterns = [
                r'recomiendo.*?([A-Z0-9]{10})',
                r'sugiero.*?([A-Z0-9]{10})', 
                r'te recomiendo.*?([A-Z0-9]{10})',
                r'producto.*?([A-Z0-9]{10})'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, answer, re.IGNORECASE)
                for match in matches:
                    if match in product_ids:
                        return match
            
            return None
        except Exception:
            return None