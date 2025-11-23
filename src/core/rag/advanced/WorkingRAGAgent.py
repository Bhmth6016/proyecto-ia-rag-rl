from __future__ import annotations

# src/core/rag/advanced/WorkingRAGAgent.py

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

# Importar Google Generative AI para Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Google Generative AI not available. Install with: pip install google-generativeai")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class RAGConfig:
    enable_reranking: bool = True
    enable_rlhf: bool = True
    max_retrieved: int = 50
    max_final: int = 5
    memory_window: int = 3
    llm_fallback: str = "mock"
    gemini_model: str = "gemini-1.5-flash"
    use_gemini: bool = True
    domain: str = "videojuegos"  # Nuevo: especificar el dominio

@dataclass
class RAGResponse:
    answer: str
    products: List[Product]
    quality_score: float
    retrieved_count: int

# ===============================
# Gemini LLM Wrapper - CORREGIDO
# ===============================
class GeminiLLMWrapper:
    def __init__(self, model_name: str = None):
        # Modelos válidos para Gemini
        self.valid_models = {
            "gemini-1.0-pro": "gemini-1.0-pro",
            "gemini-1.5-pro": "gemini-1.5-pro", 
            "gemini-pro": "gemini-pro",
            "gemini-1.0-pro-001": "gemini-1.0-pro-001"
        }
        
        # Determinar modelo a usar
        self.model_name = self._get_valid_model(model_name)
        self.model = None
        self.is_initialized = False
        self._initialize_model()

    def _get_valid_model(self, requested_model: str) -> str:
        """Obtiene un modelo válido de Gemini"""
        # Si el modelo solicitado es válido, usarlo
        if requested_model and requested_model in self.valid_models:
            return self.valid_models[requested_model]
        
        # Si no, probar modelos en orden de preferencia
        for model in ["gemini-1.0-pro", "gemini-pro", "gemini-1.5-pro"]:
            if model in self.valid_models:
                logger.info(f"🔍 Usando modelo: {model} (fallback)")
                return self.valid_models[model]
        
        # Último recurso
        return "gemini-1.0-pro"

    def _initialize_model(self):
        """Inicializa el modelo de Gemini"""
        try:
            api_key = getattr(settings, 'GEMINI_API_KEY', None)
            
            if not api_key:
                logger.warning("GEMINI_API_KEY no configurada")
                return
            
            # Verificar que la API key tenga formato básico
            if len(api_key) < 10:
                logger.warning("GEMINI_API_KEY parece muy corta")
                return
                
            genai.configure(api_key=api_key)
            
            # Listar modelos disponibles para diagnóstico
            try:
                models = genai.list_models()
                available_models = [model.name for model in models]
                logger.info(f"📋 Modelos disponibles: {[m.split('/')[-1] for m in available_models[:3]]}...")
            except Exception as e:
                logger.warning(f"⚠️ No se pudieron listar modelos: {e}")
            
            # Intentar inicializar el modelo
            self.model = genai.GenerativeModel(self.model_name)
            
            # Test simple para verificar que funciona
            test_response = self.model.generate_content("Hola")
            if test_response.text:
                self.is_initialized = True
                logger.info(f"✅ Gemini inicializado: {self.model_name}")
            else:
                logger.error("❌ Test de Gemini falló - respuesta vacía")
                
        except Exception as e:
            logger.error(f"❌ Error inicializando Gemini model {self.model_name}: {e}")
            self.is_initialized = False

    def generate(self, prompt: str) -> str:
        """Genera texto usando la API de Gemini"""
        if not self.is_initialized or not self.model:
            raise RuntimeError("Gemini no está inicializado")
        
        try:
            # Configuración para gaming
            generation_config = {
                'temperature': 0.8,  # Más creativo para gaming
                'top_p': 0.9,
                'top_k': 40,
                'max_output_tokens': 800,
            }
            
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                },
            ]
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            if response.text:
                return response.text.strip()
            else:
                logger.warning(f"Gemini bloqueó la respuesta: {response.prompt_feedback}")
                raise RuntimeError("Respuesta bloqueada por seguridad")
                
        except Exception as e:
            logger.error(f"Error generando con Gemini: {e}")
            raise

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
# Basic evaluator OPTIMIZADO PARA VIDEOJUEGOS
# ===============================
class BasicEvaluator:
    def evaluate_response(self, query: str, response: str, products: List[Product]) -> float:
        if not response:
            return 0.0
            
        if not products:
            # Para videojuegos, dar score bajo cuando no hay productos
            if "no encontré" in response.lower() or "sugerencias" in response.lower():
                return 0.2
            return 0.0
        
        # Score basado en múltiples factores optimizado para videojuegos
        content_score = min(1.0, len(response) / 400)
        products_score = min(1.0, len(products) / 3)
        relevance_score = self._calculate_relevance_score(query, response, products)
        
        final_score = (content_score * 0.3 + products_score * 0.4 + relevance_score * 0.3)
        return round(final_score, 2)
    
    def _calculate_relevance_score(self, query: str, response: str, products: List[Product]) -> float:
        """Calcula score de relevancia para videojuegos"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        response_lower = response.lower()
        
        # Palabras clave comunes en videojuegos
        gaming_keywords = {'juego', 'videojuego', 'playstation', 'xbox', 'nintendo', 'pc', 
                          'consola', 'game', 'gaming', 'edición', 'versión', 'digital'}
        
        # Verificar coincidencia con consulta
        query_matches = sum(1 for word in query_words if len(word) > 2 and word in response_lower)
        query_relevance = min(1.0, query_matches / max(1, len(query_words)))
        
        # Verificar relevancia de productos para videojuegos
        product_relevance = 0.0
        relevant_products = 0
        
        for product in products:
            title = getattr(product, 'title', '').lower()
            
            # Para videojuegos, consideramos relevante si coincide con alguna palabra de la consulta
            # o si contiene palabras clave de gaming
            title_match = any(word in title for word in query_words if len(word) > 2)
            gaming_match = any(keyword in title for keyword in gaming_keywords)
            
            if title_match or gaming_match:
                relevant_products += 1
        
        product_relevance = min(1.0, relevant_products / len(products)) if products else 0
        
        return (query_relevance + product_relevance) / 2

# ===============================
# Compatible RLHFTrainer OPTIMIZADO PARA VIDEOJUEGOS
# ===============================
class CompatibleRLHFTrainer:
    def __init__(self):
        self.reward_model = None
        self.models_dir = Path("models/rl_models")
        self.models_dir.mkdir(exist_ok=True, parents=True)
    
    def score_product_relevance(self, query: str, product: Product, user_profile: Dict) -> float:
        base_score = 0.3
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        title = getattr(product, "title", "").lower()
        category = getattr(product, "main_category", "").lower()
        description = getattr(product, "description", "").lower()
        
        # Palabras clave de videojuegos para matching mejorado
        gaming_platforms = {'playstation', 'xbox', 'nintendo', 'switch', 'pc', 'steam', 'digital'}
        gaming_terms = {'juego', 'game', 'edición', 'versión', 'colección', 'remastered'}
        
        # Score por coincidencia en título (alto peso)
        title_matches = sum(1 for word in query_words if len(word) > 2 and word in title)
        base_score += title_matches * 0.25
        
        # Score adicional por plataformas de gaming
        platform_matches = sum(1 for platform in gaming_platforms if platform in title)
        base_score += platform_matches * 0.1
        
        # Score por términos de gaming
        gaming_term_matches = sum(1 for term in gaming_terms if term in title)
        base_score += gaming_term_matches * 0.05
        
        # Score por categoría
        category_matches = sum(1 for word in query_words if len(word) > 2 and word in category)
        base_score += category_matches * 0.15
        
        # Score por categorías preferidas
        preferred_cats = user_profile.get("preferred_categories", [])
        if category and category in [cat.lower() for cat in preferred_cats]:
            base_score += 0.2
        
        # Score por precio disponible
        if getattr(product, "price", None) is not None:
            base_score += 0.05
            
        # Score por rating alto
        rating = getattr(product, "average_rating", 0)
        if rating and rating >= 4.0:
            base_score += 0.1
            
        return min(1.0, base_score)
    
    def get_latest_model_path(self) -> Optional[Path]:
        if not self.models_dir.exists():
            return None
        model_files = list(self.models_dir.glob("*.pkl"))
        return max(model_files, key=lambda p: p.stat().st_mtime) if model_files else None

# ===============================
# Main Agent - OPTIMIZADO PARA VIDEOJUEGOS
# ===============================
class WorkingAdvancedRAGAgent:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.system = get_system()
        self.retriever = getattr(self.system, "retriever", Retriever())
        
        # Inicializar Gemini LLM
        self.gemini_llm = self._initialize_gemini_llm()
        
        # Inicializar evaluador
        self.evaluator = BasicEvaluator()
        
        self.feedback_processor = FeedbackProcessor()
        self.rlhf_trainer = self._create_compatible_trainer()
        self.user_profiles: Dict[str, Dict] = {}
        self.user_memory: Dict[str, ConversationBufferMemory] = {}
        
        logger.info(f"✅ WorkingAdvancedRAGAgent inicializado para dominio: {self.config.domain}")
        logger.info(f"📊 Gemini disponible: {self.gemini_llm is not None}")

    def _initialize_gemini_llm(self) -> Optional[GeminiLLMWrapper]:
        """Inicializa Gemini LLM"""
        if not GEMINI_AVAILABLE:
            logger.warning("❌ Google Generative AI no disponible")
            return None
        
        try:
            model_name = getattr(settings, 'MODEL_NAME', self.config.gemini_model)
            wrapper = GeminiLLMWrapper(model_name)
            return wrapper if wrapper.is_initialized else None
        except Exception as e:
            logger.error(f"❌ Error inicializando Gemini LLM: {e}")
            return None

    def _create_compatible_trainer(self):
        if not self.config.enable_rlhf:
            return None
        try:
            from src.core.rag.advanced.trainer import RLHFTrainer
            return RLHFTrainer()
        except (ImportError, AttributeError):
            logger.info("Usando CompatibleRLHFTrainer (fallback)")
            return CompatibleRLHFTrainer()

    def process_query(self, query: str, user_id: str = "default") -> RAGResponse:
        """Procesa una consulta - OPTIMIZADO PARA VIDEOJUEGOS"""
        try:
            profile = self.get_or_create_user_profile(user_id)
            memory = self.get_or_create_memory(user_id)
            
            # Enriquecer consulta para videojuegos
            enriched_query = self._enrich_gaming_query(query, profile)
            candidates = self.retriever.retrieve(enriched_query, k=self.config.max_retrieved)
            
            # Filtrar y rerankear productos
            relevant_candidates = self._filter_gaming_products(candidates, query)
            ranked = self._rerank_with_rlhf(relevant_candidates, query, profile)
            final_products = ranked[:self.config.max_final]
            
            # Generar respuesta
            context_for_generation = memory.get_context()
            full_context = f"{context_for_generation}\nNueva consulta: {query}" if context_for_generation else query
            
            response = self._generate_gaming_response(full_context, final_products, query)
            quality_score = self.evaluator.evaluate_response(query, response, final_products)
            
            # Guardar en memoria
            memory.add(query, response)
            
            logger.info(f"✅ Query procesada: '{query}' -> {len(final_products)} productos, score: {quality_score}")
            
            return RAGResponse(
                answer=response, 
                products=final_products, 
                quality_score=quality_score, 
                retrieved_count=len(ranked)
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
        """Enriquece consultas para videojuegos"""
        query_lower = query.lower()
        
        # Términos de gaming para enriquecimiento
        gaming_terms = []
        
        # Detectar plataformas
        platforms = {
            'playstation': ['playstation', 'ps4', 'ps5', 'ps3'],
            'xbox': ['xbox', 'xbox one', 'xbox series'],
            'nintendo': ['nintendo', 'switch', 'wii', 'ds'],
            'pc': ['pc', 'computadora', 'steam']
        }
        
        for platform, keywords in platforms.items():
            if any(keyword in query_lower for keyword in keywords):
                gaming_terms.append(platform)
        
        # Detectar géneros
        genres = {
            'acción': ['acción', 'action', 'shooter'],
            'aventura': ['aventura', 'adventure', 'rpg'], 
            'deportes': ['deporte', 'sport', 'fútbol', 'fifa'],
            'estrategia': ['estrategia', 'strategy']
        }
        
        for genre, keywords in genres.items():
            if any(keyword in query_lower for keyword in keywords):
                gaming_terms.append(genre)
        
        # Añadir términos de enriquecimiento
        if gaming_terms:
            enriched = f"{query} {' '.join(gaming_terms)}"
            logger.debug(f"Consulta enriquecida para gaming: '{query}' -> '{enriched}'")
            return enriched
        
        return query

    def _filter_gaming_products(self, products: List[Product], query: str) -> List[Product]:
        """Filtra productos de videojuegos de forma inteligente"""
        if not products:
            return []
        
        query_lower = query.lower()
        query_words = set(word for word in query_lower.split() if len(word) > 2)
        
        # Términos específicos de videojuegos
        gaming_keywords = {'playstation', 'xbox', 'nintendo', 'switch', 'wii', 'game', 
                          'juego', 'edición', 'versión', 'ps4', 'ps5', 'xbox one'}
        
        relevant_products = []
        for product in products:
            title = getattr(product, 'title', '').lower()
            category = getattr(product, 'main_category', '').lower()
            
            # Coincidencia directa en título
            title_match = any(word in title for word in query_words)
            
            # Coincidencia con términos de gaming
            gaming_match = any(keyword in title for keyword in gaming_keywords)
            
            # Coincidencia en categoría
            category_match = any(word in category for word in query_words)
            
            # Para videojuegos, ser más permisivo pero priorizar coincidencias
            if title_match or (gaming_match and category_match):
                relevant_products.append(product)
        
        # Si no hay coincidencias fuertes, devolver productos que al menos sean videojuegos
        if not relevant_products:
            gaming_products = [p for p in products if any(kw in getattr(p, 'title', '').lower() 
                                                         for kw in gaming_keywords)]
            if gaming_products:
                logger.info(f"⚠️ Usando {len(gaming_products)} productos gaming como fallback para '{query}'")
                return gaming_products[:10]
        
        return relevant_products

    def _rerank_with_rlhf(self, products: List[Product], query: str, profile: Dict) -> List[Product]:
        if not self.rlhf_trainer or not products:
            return products
        try:
            scored_products = [(self.rlhf_trainer.score_product_relevance(query, p, profile), p) for p in products]
            scored_products.sort(key=lambda x: x[0], reverse=True)
            return [p for _, p in scored_products]
        except Exception as e:
            logger.warning("RLHF reranking failed, fallback: %s", e)
            return products

    def _generate_gaming_response(self, context: str, products: List[Product], original_query: str) -> str:
        """Genera respuesta optimizada para videojuegos"""
        if not products:
            return self._no_gaming_results_response(original_query)
        
        # Usar Gemini si está disponible
        if self.gemini_llm and products:
            try:
                gemini_response = self._generate_gaming_with_gemini(original_query, products)
                if gemini_response and len(gemini_response) > 50:
                    logger.info("✅ Respuesta generada con Gemini para gaming")
                    return gemini_response
            except Exception as e:
                logger.error(f"❌ Error generating with Gemini: {e}")
        
        # Fallback al método básico para gaming
        return self._generate_basic_gaming_response(original_query, products)

    def _generate_gaming_with_gemini(self, query: str, products: List[Product]) -> str:
        """Genera respuesta para videojuegos usando Gemini"""
        products_info = []
        for i, product in enumerate(products[:5], 1):
            title = getattr(product, 'title', 'Videojuego sin nombre')
            price = getattr(product, 'price', None)
            rating = getattr(product, 'average_rating', None)
            category = getattr(product, 'main_category', 'Videojuegos')
            
            product_info = f"{i}. **{title}**"
            if price:
                product_info += f" - 💰 ${price:.2f}"
            if rating:
                product_info += f" - ⭐ {rating}/5"
            product_info += f" - 🎮 {category}"
                
            products_info.append(product_info)
        
        products_text = "\n\n".join(products_info)
        
        prompt = f"""
        Eres un experto en videojuegos ayudando a un usuario a encontrar juegos.
        
        CONSULTA: "{query}"
        
        VIDEOJUEGOS DISPONIBLES:
        {products_text}
        
        Genera una respuesta que:
        - Sea entusiasta y apasionada sobre videojuegos
        - Destaque los juegos más relevantes para la consulta
        - Mencione plataformas (PlayStation, Xbox, Nintendo, PC)
        - Incluye precios y valoraciones cuando estén disponibles
        - Usa emojis de gaming apropiados (🎮, 🕹️, 👾, 🎯)
        - Sé conciso pero útil (máximo 200 palabras)
        
        RESPUESTA:
        """
        
        return self.gemini_llm.generate(prompt)

    def _generate_basic_gaming_response(self, query: str, products: List[Product]) -> str:
        """Método básico de generación para videojuegos"""
        # Agrupar por plataforma si es posible
        platforms = {}
        for product in products:
            title = getattr(product, 'title', '').lower()
            platform = "Otras plataformas"
            
            if 'playstation' in title or 'ps4' in title or 'ps5' in title:
                platform = "PlayStation"
            elif 'xbox' in title:
                platform = "Xbox" 
            elif 'nintendo' in title or 'switch' in title:
                platform = "Nintendo"
            elif 'pc' in title:
                platform = "PC"
                
            platforms.setdefault(platform, []).append(product)
        
        lines = [f"🎮 **Recomendaciones de videojuegos para '{query}'**", ""]
        
        total_shown = 0
        for platform, prods in list(platforms.items())[:3]:
            lines.append(f"📀 **{platform}**")
            for i, p in enumerate(prods[:3], 1):
                title = getattr(p, 'title', 'Videojuego sin nombre')
                price = getattr(p, 'price', None)
                rating = getattr(p, 'average_rating', None)
                price_str = f"💰 ${price:.2f}" if price else "💰 Precio no disponible"
                rating_str = f"⭐ {rating}/5" if rating else "⭐ Sin calificaciones"
                lines.append(f"  {i}. **{title}**")
                lines.append(f"     {price_str} | {rating_str}")
                total_shown += 1
            lines.append("")
        
        if len(products) > total_shown:
            lines.append(f"💡 *Y {len(products) - total_shown} juegos más...*")
        
        return "\n".join(lines)

    def _no_gaming_results_response(self, query: str) -> str:
        """Respuesta cuando no hay videojuegos"""
        suggestions = [
            "Prueba con el nombre específico del juego",
            "Busca por plataforma: 'playstation', 'xbox', 'nintendo', 'pc'",
            "Intenta con el género: 'acción', 'aventura', 'deportes', 'estrategia'",
            "Usa términos como 'mejor valorado' o 'novedades'"
        ]
        return f"🎮 **No encontré videojuegos para '{query}'**\n\n**Sugerencias para gaming:**\n" + "\n".join(f"• {s}" for s in suggestions)

    def _error_response(self, query: str, error: Exception) -> str:
        return f"❌ **Error buscando videojuegos para '{query}'**\n\nDetalle: {str(error)}\n\nPor favor, intenta de nuevo."

    def get_or_create_user_profile(self, user_id: str) -> Dict:
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "user_id": user_id, 
                "preferred_categories": ["games", "videojuegos"],  # Por defecto para gaming
                "preferred_platforms": [],
                "search_history": [], 
                "purchase_history": []
            }
        return self.user_profiles[user_id]

    def get_or_create_memory(self, user_id: str) -> ConversationBufferMemory:
        if user_id not in self.user_memory:
            self.user_memory[user_id] = ConversationBufferMemory(max_length=self.config.memory_window)
        return self.user_memory[user_id]

    def clear_memory(self, user_id: str = "default"):
        if user_id in self.user_memory:
            self.user_memory[user_id].clear()

    def log_feedback(self, query: str, answer: str, rating: int, user_id: str = "default"):
        # Implementación existente
        pass