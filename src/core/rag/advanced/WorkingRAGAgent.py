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
    gemini_model: str = "gemini-1.5-flash"  # Modelo correcto

@dataclass
class RAGResponse:
    answer: str
    products: List[Product]
    quality_score: float
    retrieved_count: int

# ===============================
# Gemini LLM Wrapper CORREGIDO
# ===============================
class GeminiLLMWrapper:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Inicializa el modelo de Gemini"""
        try:
            api_key = getattr(settings, 'GEMINI_API_KEY', None)
            if not api_key:
                logger.warning("GEMINI_API_KEY no configurada")
                return
                
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini model inicializado: {self.model_name}")
        except Exception as e:
            logger.error(f"Error inicializando Gemini model: {e}")
            self.model = None

    def generate(self, prompt: str) -> str:
        """Genera texto usando la API de Gemini"""
        if not self.model:
            raise RuntimeError("Gemini model no está inicializado")
        
        try:
            response = self.model.generate_content(prompt)
            if response.text:
                return response.text.strip()
            else:
                raise RuntimeError(f"Respuesta vacía de Gemini: {response.prompt_feedback}")
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
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
# Basic evaluator fallback MEJORADO
# ===============================
class BasicEvaluator:
    def evaluate_response(self, query: str, response: str, products: List[Product]) -> float:
        if not response or not products:
            return 0.0
        
        # Score basado en múltiples factores
        content_score = min(1.0, len(response) / 300)  # Longitud razonable
        products_score = min(1.0, len(products) / 5)   # Número de productos
        relevance_score = self._calculate_relevance_score(query, response, products)
        
        final_score = (content_score * 0.3 + products_score * 0.4 + relevance_score * 0.3)
        return round(final_score, 2)
    
    def _calculate_relevance_score(self, query: str, response: str, products: List[Product]) -> float:
        """Calcula score de relevancia basado en la consulta y productos"""
        query_words = set(query.lower().split())
        response_lower = response.lower()
        
        # Verificar si la respuesta contiene palabras de la consulta
        query_matches = sum(1 for word in query_words if word in response_lower)
        query_relevance = min(1.0, query_matches / len(query_words)) if query_words else 0.5
        
        # Verificar relevancia de productos (títulos que coincidan con la consulta)
        product_relevance = 0.0
        for product in products:
            title = getattr(product, 'title', '').lower()
            if any(word in title for word in query_words if len(word) > 3):
                product_relevance += 0.2
        
        product_relevance = min(1.0, product_relevance)
        
        return (query_relevance + product_relevance) / 2

# ===============================
# Compatible RLHFTrainer fallback
# ===============================
class CompatibleRLHFTrainer:
    def __init__(self):
        self.reward_model = None
        self.models_dir = Path("models/rl_models")
        self.models_dir.mkdir(exist_ok=True, parents=True)
    
    def score_product_relevance(self, query: str, product: Product, user_profile: Dict) -> float:
        base_score = 0.3  # Score base más bajo
        
        # Score por categorías preferidas
        preferred_cats = user_profile.get("preferred_categories", [])
        product_category = getattr(product, "main_category", "")
        if product_category and product_category in preferred_cats:
            base_score += 0.4
        
        # Score por coincidencia de palabras en título
        query_words = set(query.lower().split())
        title_words = set(getattr(product, "title", "").lower().split())
        matches = len(query_words.intersection(title_words))
        base_score += matches * 0.15
        
        # Score por precio (preferir productos con precio)
        if getattr(product, "price", None) is not None:
            base_score += 0.1
            
        # Score por rating
        rating = getattr(product, "average_rating", 0)
        if rating and rating >= 4.0:
            base_score += 0.05
            
        return min(1.0, base_score)
    
    def get_latest_model_path(self) -> Optional[Path]:
        if not self.models_dir.exists():
            return None
        model_files = list(self.models_dir.glob("*.pkl"))
        return max(model_files, key=lambda p: p.stat().st_mtime) if model_files else None

# ===============================
# Main Agent - VERSIÓN CORREGIDA
# ===============================
class WorkingAdvancedRAGAgent:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.system = get_system()
        self.retriever = getattr(self.system, "retriever", Retriever())
        
        # Inicializar Gemini LLM
        self.gemini_llm = self._initialize_gemini_llm()
        
        # Inicializar evaluador (siempre BasicEvaluator por ahora)
        self.evaluator = BasicEvaluator()
        
        self.feedback_processor = FeedbackProcessor()
        self.rlhf_trainer = self._create_compatible_trainer()
        self.user_profiles: Dict[str, Dict] = {}
        self.user_memory: Dict[str, ConversationBufferMemory] = {}

    def _initialize_gemini_llm(self) -> Optional[GeminiLLMWrapper]:
        """Inicializa el cliente de Gemini LLM"""
        if not GEMINI_AVAILABLE:
            logger.warning("Google Generative AI no está disponible. Instala con: pip install google-generativeai")
            return None
        
        try:
            model_name = getattr(settings, 'MODEL_NAME', self.config.gemini_model)
            return GeminiLLMWrapper(model_name)
        except Exception as e:
            logger.error(f"Error inicializando Gemini LLM: {e}")
            return None

    def _create_compatible_trainer(self):
        if not self.config.enable_rlhf:
            return None
        try:
            from src.core.rag.advanced.trainer import RLHFTrainer
            return RLHFTrainer()
        except (ImportError, AttributeError):
            return CompatibleRLHFTrainer()

    def process_query(self, query: str, user_id: str = "default") -> RAGResponse:
        """Procesa una consulta"""
        try:
            profile = self.get_or_create_user_profile(user_id)
            memory = self.get_or_create_memory(user_id)
            
            # Usar consulta actual para retrieval
            enriched_query = self._enrich_query(query, profile)
            candidates = self.retriever.retrieve(enriched_query, k=self.config.max_retrieved)
            
            # Filtrar productos irrelevantes
            relevant_candidates = self._filter_irrelevant_products(candidates, query)
            ranked = self._rerank_with_rlhf(relevant_candidates, query, profile)
            final_products = ranked[:self.config.max_final]
            
            # Generar respuesta
            context_for_generation = memory.get_context()
            full_context = f"{context_for_generation}\nNueva consulta: {query}" if context_for_generation else query
            
            response = self._generate_rich_response(full_context, final_products, query)
            quality_score = self.evaluator.evaluate_response(query, response, final_products)
            
            # Guardar en memoria
            memory.add(query, response)
            
            return RAGResponse(
                answer=response, 
                products=final_products, 
                quality_score=quality_score, 
                retrieved_count=len(ranked)
            )
            
        except Exception as e:
            logger.error(f"Error en process_query: {e}")
            return RAGResponse(
                answer=f"Error procesando la consulta: {str(e)}",
                products=[],
                quality_score=0.0,
                retrieved_count=0
            )

    def _filter_irrelevant_products(self, products: List[Product], query: str) -> List[Product]:
        """Filtra productos claramente irrelevantes"""
        if not products:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        relevant_products = []
        for product in products:
            title = getattr(product, 'title', '').lower()
            category = getattr(product, 'main_category', '').lower()
            description = getattr(product, 'description', '').lower()
            
            # Verificar coincidencia en título, categoría o descripción
            title_match = any(word in title for word in query_words if len(word) > 3)
            category_match = any(word in category for word in query_words if len(word) > 3)
            description_match = any(word in description for word in query_words if len(word) > 3)
            
            if title_match or category_match or description_match:
                relevant_products.append(product)
        
        # Si no encontramos relevantes, devolver los primeros
        return relevant_products if relevant_products else products[:10]

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

    def _enrich_query(self, query: str, profile: Dict) -> str:
        """Enriquece la consulta con preferencias del usuario"""
        prefs = profile.get("preferred_categories", [])
        if prefs:
            # Solo añadir categorías relevantes
            query_lower = query.lower()
            relevant_prefs = [pref for pref in prefs if any(word in pref.lower() for word in query_lower.split())]
            if relevant_prefs:
                return f"{query} {' '.join(relevant_prefs)}"
        return query

    def _generate_rich_response(self, context: str, products: List[Product], original_query: str) -> str:
        """Genera respuesta usando Gemini o fallback básico"""
        if not products:
            return self._no_results_response(original_query)
        
        # Intentar usar Gemini LLM si está disponible
        if self.gemini_llm:
            try:
                return self._generate_with_gemini(original_query, products)
            except Exception as e:
                logger.error(f"Error generating with Gemini, using fallback: {e}")
        
        # Fallback al método básico
        return self._generate_basic_response(original_query, products)

    def _generate_with_gemini(self, query: str, products: List[Product]) -> str:
        """Genera respuesta enriquecida usando Gemini LLM"""
        products_info = []
        for i, product in enumerate(products[:5], 1):
            title = getattr(product, 'title', 'Producto sin nombre')
            price = getattr(product, 'price', None)
            rating = getattr(product, 'average_rating', None)
            category = getattr(product, 'main_category', 'General')
            description = getattr(product, 'description', '')[:100]  # Primeros 100 chars
            
            product_info = f"{i}. **{title}**"
            if price:
                product_info += f" - ${price:.2f}"
            if rating:
                product_info += f" - ⭐ {rating}/5"
            product_info += f" - {category}"
            if description:
                product_info += f"\n   📝 {description}..."
                
            products_info.append(product_info)
        
        products_text = "\n\n".join(products_info)
        
        prompt = f"""
        Eres un asistente especializado en recomendaciones de productos. 
        El usuario busca: "{query}"
        
        Aquí tienes los productos disponibles:
        {products_text}
        
        Por favor, genera una respuesta que:
        1. Sea útil y atractiva para el usuario
        2. Destaque los productos más relevantes para su búsqueda
        3. Incluye información clave como precio y valoraciones
        4. Usa emojis apropiados para hacerlo más visual
        5. Mantén un tono amigable y profesional
        6. Si algún producto no parece relevante, explícalo amablemente
        
        Formato: Usa markdown para organizar la información claramente.
        """
        
        response = self.gemini_llm.generate(prompt)
        return response if response else self._generate_basic_response(query, products)

    def _generate_basic_response(self, query: str, products: List[Product]) -> str:
        """Método básico de generación de respuestas"""
        categorized = {}
        for product in products:
            cat = getattr(product, "main_category", "General")
            categorized.setdefault(cat, []).append(product)
        
        lines = [f"🎯 **Recomendaciones para '{query}'**\n"]
        
        for cat, prods in list(categorized.items())[:3]:
            lines.append(f"📁 **{cat}**")
            for i, p in enumerate(prods[:3], 1):
                title = getattr(p, 'title', 'Producto sin nombre')
                price = getattr(p, 'price', None)
                rating = getattr(p, 'average_rating', None)
                price_str = f"${price:.2f}" if price else "Precio no disponible"
                rating_str = f"⭐ {rating}/5" if rating else "⭐ Sin calificaciones"
                lines.append(f"  {i}. **{title}** | {price_str} | {rating_str}")
            lines.append("")
        
        if len(products) > 6:
            lines.append(f"💡 *Encontré {len(products)} productos relevantes*")
        
        return "\n".join(lines)

    def _no_results_response(self, query: str) -> str:
        """Respuesta cuando no hay productos"""
        suggestions = [
            "Prueba con términos más generales o específicos",
            "Revisa la ortografía de tu búsqueda",
            "Intenta buscar por categoría como 'electrónica', 'hogar', 'libros'",
            "Usa palabras clave como 'mejor valorado' o 'económico'"
        ]
        return f"🔍 **No encontré productos para '{query}'**\n\n**Sugerencias:**\n" + "\n".join(f"• {s}" for s in suggestions)

    def get_or_create_user_profile(self, user_id: str) -> Dict:
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "user_id": user_id, 
                "preferred_categories": [], 
                "search_history": [], 
                "purchase_history": []
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
        entry = {
            "timestamp": datetime.now().isoformat(), 
            "query": query, 
            "answer": answer, 
            "rating": rating, 
            "user_id": user_id
        }
        try:
            fdir = Path("data/feedback")
            fdir.mkdir(exist_ok=True, parents=True)
            fname = fdir / f"feedback_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(fname, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            try:
                self.feedback_processor.save_feedback(
                    query=query, 
                    answer=answer, 
                    rating=rating, 
                    extra_meta={"user_id": user_id}
                )
            except Exception:
                pass
        except Exception as e:
            logger.debug("Failed to log feedback: %s", e)