from __future__ import annotations

# src/core/rag/advanced/WorkingRAGAgent.py
import time

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
from collections import deque

# Local imports
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
            title = getattr(product, 'title', '').lower()
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
        category = getattr(product, "main_category", "").lower()
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
        
        # Componentes optimizados
        self.evaluator = AdvancedEvaluator()
        self.feedback_processor = FeedbackProcessor()
        self.rlhf_trainer = GamingRLHFTrainer()
        
        # Estado del usuario
        self.user_profiles: Dict[str, Dict] = {}
        self.user_memory: Dict[str, ConversationBufferMemory] = {}
        self.min_feedback_for_retrain = 50  # Mínimo feedback para reentrenar
        self.retrain_interval = 24 * 3600   # Reentrenar cada 24 horas
        self.last_retrain_time = 0
        
        # Verificar si hay suficiente feedback para reentrenar
        self._check_and_retrain()
        
        logger.info(f"✅ WorkingAdvancedRAGAgent inicializado - Dominio: {self.config.domain}")

    def process_query(self, query: str, user_id: str = "default") -> RAGResponse:
        """Procesa consultas de gaming de forma optimizada"""
        try:
            profile = self.get_or_create_user_profile(user_id)
            memory = self.get_or_create_memory(user_id)
            
            # Enriquecimiento inteligente para gaming
            enriched_query = self._enrich_gaming_query(query, profile)
            candidates = self.retriever.retrieve(enriched_query, k=self.config.max_retrieved)
            
            # Filtrado y reranking avanzado
            relevant_candidates = self._filter_gaming_products(candidates, query)
            ranked = self._rerank_with_rlhf(relevant_candidates, query, profile)
            final_products = ranked[:self.config.max_final]
            
            # Generación de respuesta optimizada
            context_for_generation = memory.get_context()
            full_context = f"{context_for_generation}\nNueva consulta: {query}" if context_for_generation else query
            
            response = self._generate_gaming_response(full_context, final_products, query)
            quality_score = self.evaluator.evaluate_response(query, response, final_products)
            
            # Guardar en memoria
            memory.add(query, response)
            
            logger.info(f"✅ Query: '{query}' -> {len(final_products)} productos | Score: {quality_score}")
            
            return RAGResponse(
                answer=response, 
                products=final_products, 
                quality_score=quality_score, 
                retrieved_count=len(ranked),
                used_llm=False
            )
            
        except Exception as e:
            logger.error(f"❌ Error en process_query: {e}")
            return RAGResponse(
                answer=self._error_response(query, e),
                products=[],
                quality_score=0.0,
                retrieved_count=0
            )

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



    def _filter_gaming_products(self, products: List[Product], query: str) -> List[Product]:
        """Filtrado inteligente para productos de gaming"""
        if not products:
            return []
        
        query_lower = query.lower()
        query_words = set(word for word in query_lower.split() if len(word) > 2)
        
        # Términos clave de gaming
        gaming_terms = {'playstation', 'xbox', 'nintendo', 'switch', 'game', 'juego', 
                       'edición', 'versión', 'ps4', 'ps5', 'xbox one'}
        
        relevant_products = []
        for product in products:
            title = getattr(product, 'title', '').lower()
            category = getattr(product, 'main_category', '').lower()
            
            # Coincidencia directa en título
            title_match = any(word in title for word in query_words)
            
            # Coincidencia con términos de gaming
            gaming_match = any(term in title for term in gaming_terms)
            
            # Coincidencia en categoría
            category_match = any(word in category for word in query_words)
            
            # Para gaming, ser más permisivo pero priorizar coincidencias
            if title_match or (gaming_match and (category_match or len(query_words) == 0)):
                relevant_products.append(product)
        
        # Fallback: productos que al menos son de gaming
        if not relevant_products:
            gaming_products = [p for p in products if any(term in getattr(p, 'title', '').lower() 
                                                         for term in gaming_terms)]
            if gaming_products:
                logger.info(f"🔄 Usando {len(gaming_products)} productos gaming como fallback")
                return gaming_products[:10]
        
        return relevant_products

    def _rerank_with_rlhf(self, products: List[Product], query: str, profile: Dict) -> List[Product]:
        """Reranking optimizado para gaming"""
        if not products:
            return products
            
        try:
            scored_products = [(self.rlhf_trainer.score_product_relevance(query, p, profile), p) 
                             for p in products]
            scored_products.sort(key=lambda x: x[0], reverse=True)
            return [p for _, p in scored_products]
        except Exception as e:
            logger.warning(f"Reranking falló, usando orden original: {e}")
            return products

    def _generate_gaming_response(self, context: str, products: List[Product], original_query: str) -> str:
        """Generación de respuesta optimizada para gaming"""
        if not products:
            return self._no_gaming_results_response(original_query)
        
        return self._generate_advanced_gaming_response(original_query, products)

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
            title = getattr(product, 'title', '').lower()
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

    def log_feedback(self, query: str, answer: str, rating: int, user_id: str = "default"):
        """Log de feedback para mejora continua"""
        entry = {
            "timestamp": datetime.now().isoformat(), 
            "query": query, 
            "answer": answer, 
            "rating": rating, 
            "user_id": user_id,
            "domain": self.config.domain
        }
        try:
            fdir = Path("data/feedback")
            fdir.mkdir(exist_ok=True, parents=True)
            fname = fdir / f"feedback_gaming_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(fname, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"No se pudo guardar feedback: {e}")
            
    def _check_and_retrain(self):
        """Verifica si hay suficiente feedback para reentrenar el modelo"""
        try:
            feedback_count = self._count_recent_feedback()
            
            if (feedback_count >= self.min_feedback_for_retrain and 
                time.time() - self.last_retrain_time > self.retrain_interval):
                
                logger.info(f"🔁 Iniciando reentrenamiento con {feedback_count} feedbacks")
                self._retrain_with_feedback()
                self.last_retrain_time = time.time()
                
        except Exception as e:
            logger.error(f"Error en reentrenamiento automático: {e}")
    
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
    
    def _retrain_with_feedback(self):
        """Reentrena el modelo RLHF con feedback acumulado"""
        try:
            from .trainer import RLHFTrainer
            
            trainer = RLHFTrainer()
            
            # Preparar dataset desde logs
            failed_log = Path("data/feedback/failed_queries.log")
            success_log = Path("data/feedback/success_queries.log")
            
            dataset = trainer.prepare_rlhf_dataset_from_logs(failed_log, success_log)
            
            if len(dataset) >= 10:  # Mínimo para entrenar
                logger.info(f"🏋️ Entrenando con {len(dataset)} ejemplos")
                trainer.train(dataset)
                logger.info("✅ Modelo RLHF reentrenado")
                
        except Exception as e:
            logger.error(f"❌ Error en reentrenamiento RLHF: {e}")
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