# src/core/rag/advanced/WorkingRAGAgent.py
"""
WorkingRAGAgent - Agente RAG avanzado con configuración ML centralizada.
Usa ProductReference y settings como única fuente de verdad.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import torch
from pathlib import Path

# Importar configuración centralizada
from src.core.config import settings, get_settings
from src.core.data.product import Product
from src.core.data.product_reference import ProductReference, create_ml_enhanced_reference

logger = logging.getLogger(__name__)


class RAGMode(Enum):
    """Modos de operación del RAG."""
    BASIC = "basic"
    HYBRID = "hybrid"
    ML_ENHANCED = "ml_enhanced"
    LLM_ENHANCED = "llm_enhanced"


@dataclass
class RAGConfig:
    """Configuración del agente RAG."""
    # Modo de operación
    mode: RAGMode = RAGMode.HYBRID
    
    # Configuración de recuperación
    enable_reranking: bool = True
    max_retrieved: int = 15
    max_final: int = 5
    
    # Configuración ML (se hereda de settings)
    ml_enabled: bool = field(default_factory=lambda: settings.ML_ENABLED)
    ml_features: List[str] = field(default_factory=lambda: list(settings.ML_FEATURES))
    use_ml_embeddings: bool = field(default_factory=lambda: settings.ML_ENABLED and 'embedding' in settings.ML_FEATURES)
    ml_embedding_weight: float = field(default_factory=lambda: settings.ML_WEIGHT)
    
    # Configuración LLM
    local_llm_enabled: bool = field(default_factory=lambda: settings.LOCAL_LLM_ENABLED)
    local_llm_model: str = field(default_factory=lambda: settings.LOCAL_LLM_MODEL)
    use_llm_for_reranking: bool = False
    
    # Configuración de dominio
    domain: str = "general"
    use_advanced_features: bool = True
    
    # Ponderaciones para scoring híbrido
    semantic_weight: float = 0.6
    popularity_weight: float = 0.2
    diversity_weight: float = 0.1
    freshness_weight: float = 0.1


class WorkingAdvancedRAGAgent:
    """
    Agente RAG avanzado que usa configuración ML centralizada
    y ProductReference para manejo consistente.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        # 🔥 Usar configuración centralizada
        self.settings = get_settings()
        
        # Configuración del agente
        self.config = config or RAGConfig()
        
        # Componentes del sistema (lazy loaded)
        self._retriever = None
        self._llm_client = None
        self._embedding_model = None
        
        # Cache para embeddings de queries
        self._query_cache = {}
        
        # Inicializar logger
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 🔥 NUEVO: Pipeline RLHF
        self.rlhf_pipeline = None
        self.rlhf_model = None
        self._init_rlhf()
        
        # 🔥 NUEVO: Inicializar Collaborative Filter
        self._collaborative_filter = None
        self._init_collaborative_filter()
        
        self.logger.info(f"🚀 WorkingAdvancedRAGAgent inicializado")
        self.logger.info(f"   • Modo: {self.config.mode.value}")
        self.logger.info(f"   • ML: {'✅' if self.config.ml_enabled else '❌'}")
        self.logger.info(f"   • LLM Local: {'✅' if self.config.local_llm_enabled else '❌'}")
        self.logger.info(f"   • RLHF: {'✅' if self.rlhf_pipeline else '❌'}")
        self.logger.info(f"   • Collaborative Filter: {'✅' if self._collaborative_filter else '❌'}")

    def _init_rlhf(self):
        """Inicializar componente RLHF si está habilitado"""
        try:
            from src.core.rag.advanced.train_pipeline import RLHFTrainingPipeline
            self.rlhf_pipeline = RLHFTrainingPipeline()
            
            # Intentar cargar modelo existente
            if (Path("data/models/rlhf_model") / "pytorch_model.bin").exists():
                self.rlhf_model = self.rlhf_pipeline.load_model()
                logger.info("🧠 RLHF integrado (modelo cargado)")
            else:
                logger.info("🧠 RLHF integrado (sin modelo entrenado)")
                
        except ImportError as e:
            logger.warning(f"⚠️ RLHF no disponible: {e}")
            self.rlhf_pipeline = None
    
    # En WorkingRAGAgent._init_collaborative_filter()
    def _init_collaborative_filter(self):
        """Inicializar Collaborative Filter si está habilitado"""
        try:
            from src.core.rag.advanced.collaborative_filter import CollaborativeFilter
            from src.core.data.user_manager import UserManager
            from src.core.data.product_service import ProductService  # 🔥 NUEVO
            
            # Obtener gestor de usuarios
            user_manager = UserManager()
            
            # 🔥 NUEVO: Usar ProductService real
            product_service = ProductService()
            
            # Crear filtro colaborativo con servicio real
            self._collaborative_filter = CollaborativeFilter(
                user_manager=user_manager,
                product_service=product_service,  # 🔥 Pasar servicio real
                use_ml_features=self.config.ml_enabled
            )
            
            logger.info("🤝 Collaborative Filter integrado (con ProductService)")
            
        except ImportError as e:
            logger.warning(f"⚠️ Collaborative Filter no disponible: {e}")
            # Fallback al servicio simple
            self._init_simple_collaborative_filter()
        except Exception as e:
            logger.warning(f"⚠️ Error inicializando Collaborative Filter: {e}")
    
    # --------------------------------------------------
    # Propiedades lazy
    # --------------------------------------------------
    
    @property
    def retriever(self):
        """Retriever vectorial (lazy loading)."""
        if self._retriever is None:
            try:
                from src.core.rag.basic.retriever import Retriever
                self._retriever = Retriever(
                    index_path=settings.VECTOR_INDEX_PATH,
                    embedding_model=settings.EMBEDDING_MODEL,
                    device=settings.DEVICE
                )
                self.logger.info(f"✅ Retriever inicializado: {settings.EMBEDDING_MODEL}")
            except ImportError as e:
                self.logger.error(f"❌ No se pudo cargar Retriever: {e}")
                raise
        return self._retriever
    
    @property
    def llm_client(self):
        """Cliente LLM local (lazy loading)."""
        if self._llm_client is None and self.config.local_llm_enabled:
            try:
                from src.core.llm.local_llm import LocalLLMClient
                self._llm_client = LocalLLMClient(
                    model=settings.LOCAL_LLM_MODEL,
                    endpoint=settings.LOCAL_LLM_ENDPOINT,
                    temperature=settings.LOCAL_LLM_TEMPERATURE,
                    timeout=settings.LOCAL_LLM_TIMEOUT
                )
                self.logger.info(f"✅ LLM Client inicializado: {settings.LOCAL_LLM_MODEL}")
            except ImportError as e:
                self.logger.warning(f"⚠️ No se pudo cargar LocalLLMClient: {e}")
            except Exception as e:
                self.logger.error(f"❌ Error inicializando LLM: {e}")
        return self._llm_client
    
    @property
    def embedding_model(self):
        """Modelo de embeddings (lazy loading)."""
        if self._embedding_model is None and self.config.use_ml_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(settings.ML_EMBEDDING_MODEL)
                self.logger.info(f"✅ Embedding Model cargado: {settings.ML_EMBEDDING_MODEL}")
            except ImportError as e:
                self.logger.warning(f"⚠️ SentenceTransformer no disponible: {e}")
            except Exception as e:
                self.logger.error(f"❌ Error cargando embedding model: {e}")
        return self._embedding_model
    
    # --------------------------------------------------
    # Métodos principales
    # --------------------------------------------------
    
    def process_query(self, query: str, user_id: str = None) -> Dict[str, Any]:
        """
        Procesa una consulta completa usando RAG avanzado.
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"🔍 Procesando consulta: '{query[:50]}...'")
            
            # 1. Búsqueda semántica inicial
            initial_results = self._semantic_search(query)

            # 🔍 Opcional: imprimir resultados encontrados como solicitaban
            self.logger.debug(f"Encontrados {len(initial_results)} resultados iniciales")
            for i, ref in enumerate(initial_results[:3]):
                self.logger.debug(f"{i+1}. {ref.title[:50]}... (score: {ref.score})")

            # 2. Enrich con ML si está habilitado
            ml_enhanced_results = (
                self._enhance_with_ml(initial_results, query)
                if self.config.ml_enabled else initial_results
            )

            # 3. Re-ranking final
            final_results = (
                self._rerank_results(ml_enhanced_results, query, user_id)
                if self.config.enable_reranking else
                ml_enhanced_results[:self.config.max_final]
            )

            # 4. Generación de respuesta con LLM
            answer = self._generate_answer(query, final_results)

            # 5. Métricas
            processing_time = time.time() - start_time

            response = {
                "query": query,
                "answer": answer,
                "products": final_results,
                "stats": {
                    "processing_time": round(processing_time, 2),
                    "initial_results": len(initial_results),
                    "final_results": len(final_results),
                    "ml_enhanced": self.config.ml_enabled,
                    "reranking_enabled": self.config.enable_reranking,
                }
            }

            self.logger.info(f"✅ Consulta procesada en {processing_time:.2f}s")
            return response

        except Exception as e:
            import traceback
            self.logger.error(f"❌ Error procesando consulta: {e}")
            self.logger.error(traceback.format_exc())

            return {
                "query": query,
                "answer": "Lo siento, hubo un error procesando tu consulta.",
                "products": [],
                "error": str(e)
            }

    
    # REEMPLAZA el método _semantic_search con esta versión corregida:
    def _semantic_search(self, query: str) -> List[ProductReference]:
        """Búsqueda semántica usando embeddings."""
        try:
            # 🔥 SIMPLIFICADO: Usar search() del retriever
            raw_results = self.retriever.search(
                query=query,
                k=self.config.max_retrieved
            )
            
            product_references = []
            for product in raw_results:
                try:
                    # 🔥 Validar que el producto sea válido
                    if not product or not hasattr(product, 'title'):
                        continue
                    
                    # 🔥 Asegurar que el título no sea None
                    if not product.title:
                        continue
                    
                    # Calcular score
                    score = self._calculate_product_score(product, query)
                    
                    # Crear referencia
                    from src.core.data.product_reference import ProductReference
                    ref = ProductReference.from_product(
                        product=product,
                        score=score,
                        source="rag"
                    )
                    product_references.append(ref)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error procesando resultado: {e}")
                    continue
            
            # 🔥 Ordenar solo si hay referencias
            if product_references:
                product_references.sort(key=lambda x: x.score, reverse=True)
            
            return product_references
            
        except Exception as e:
            self.logger.error(f"❌ Error en búsqueda semántica: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    # 🔥 AÑADE este método que falta:
    def _calculate_initial_score(self, product: Product, query: str) -> float:
        """Calcula score inicial basado en similitud semántica."""
        if not product or not query:
            return 0.0
        
        try:
            # Método simple usando SequenceMatcher para similitud de texto
            from difflib import SequenceMatcher
            
            # Calcular similitud basada en título y descripción
            title = getattr(product, 'title', '')
            description = getattr(product, 'description', '')
            
            # Ponderar título más que descripción
            title_sim = SequenceMatcher(None, query.lower(), title.lower()).ratio()
            desc_sim = SequenceMatcher(None, query.lower(), description.lower()).ratio() if description else 0
            
            # Combinar scores (70% título, 30% descripción)
            score = (title_sim * 0.7) + (desc_sim * 0.3)
            
            # Ajustar con otros factores
            price_factor = self._calculate_price_factor(product)
            rating_factor = self._calculate_rating_factor(product)
            
            final_score = score * 0.6 + price_factor * 0.2 + rating_factor * 0.2
            return min(1.0, max(0.0, final_score))
            
        except Exception:
            return 0.3  # Score mínimo

    def _calculate_price_factor(self, product: Product) -> float:
        """Factor basado en precio (productos con precio definido son mejores)."""
        price = getattr(product, 'price', None)
        if price and isinstance(price, (int, float)) and price > 0:
            return 0.8  # Bueno
        return 0.3  # Malo

    def _calculate_rating_factor(self, product: Product) -> float:
        """Factor basado en rating."""
        rating = getattr(product, 'average_rating', None)
        if rating and isinstance(rating, (int, float)):
            # Normalizar a 0-1
            return min(1.0, rating / 5.0)
        return 0.5  # Neutral
    def _calculate_product_score(self, product: Any, query: str) -> float:
        """Calcula un score simple para el producto basado en la query."""
        try:
            # 🔥 Asegurar que product tenga atributos necesarios
            if not product or not hasattr(product, 'title'):
                return 0.1
            
            # Método simple: similitud de texto
            from difflib import SequenceMatcher
            
            # 🔥 Asegurar que title no sea None
            title = getattr(product, 'title', '') or ''
            
            text_sim = SequenceMatcher(None, query.lower(), title.lower()).ratio()
            
            # 🔥 Agregar factores adicionales con manejo seguro de None
            price_factor = 0.5  # Valor por defecto
            if hasattr(product, 'price') and product.price is not None:
                price = float(product.price) if product.price else 0.0
                # Productos con precio definido obtienen mejor score
                price_factor = 0.8 if price > 0 else 0.3
            
            rating_factor = 0.5  # Valor por defecto
            if hasattr(product, 'average_rating') and product.average_rating is not None:
                rating = float(product.average_rating) if product.average_rating else 0.0
                rating_factor = min(1.0, rating / 5.0)
            
            # Combinar scores (60% similitud, 20% precio, 20% rating)
            final_score = (text_sim * 0.6) + (price_factor * 0.2) + (rating_factor * 0.2)
            
            # 🔥 Asegurar que el score esté en rango [0, 1]
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            self.logger.warning(f"Error calculando score: {e}")
            return 0.1  # Score mínimo
    def _enhance_with_ml(self, 
                        results: List[ProductReference], 
                        query: str) -> List[ProductReference]:
        """
        Enriquece resultados con procesamiento ML.
        Usa settings como única fuente de verdad para configuración ML.
        """
        if not results or not self.config.ml_enabled:
            return results
        
        enhanced_results = []
        query_embedding = self._get_query_embedding(query)
        
        for ref in results:
            # Solo procesar si el producto no tiene ya ML features
            if not ref.is_ml_processed:
                enhanced_ref = self._apply_ml_to_reference(ref, query_embedding)
                enhanced_results.append(enhanced_ref)
            else:
                # Si ya tiene ML, calcular similitud adicional
                if query_embedding and ref.has_embedding:
                    similarity = self._calculate_similarity(
                        query_embedding, 
                        ref.embedding
                    )
                    ref.update_ml_features({
                        'query_similarity': similarity,
                        'ml_enhanced': True
                    })
                enhanced_results.append(ref)
        
        # Ordenar por puntaje ML mejorado
        if self.config.use_ml_embeddings and query_embedding:
            enhanced_results.sort(
                key=lambda x: self._calculate_ml_score(x, query_embedding),
                reverse=True
            )
        
        self.logger.debug(f"🤖 ML Enhancement aplicado a {len(enhanced_results)} productos")
        return enhanced_results
    
    def _apply_ml_to_reference(self, 
                              ref: ProductReference,
                              query_embedding: Optional[List[float]] = None) -> ProductReference:
        """Aplica procesamiento ML a un ProductReference."""
        if not ref.product:
            return ref
        
        ml_data = {}
        
        # Extraer características ML según configuración
        if 'category' in self.config.ml_features:
            category = self._predict_category(ref.product)
            if category:
                ml_data['predicted_category'] = category
                ml_data['category_confidence'] = 0.8  # Valor por defecto
        
        if 'entities' in self.config.ml_features:
            entities = self._extract_entities(ref.product)
            if entities:
                ml_data['extracted_entities'] = entities
        
        if 'embedding' in self.config.ml_features and self.embedding_model:
            # Generar embedding si no existe
            if not ref.has_embedding:
                text = ref.product.to_text() if hasattr(ref.product, 'to_text') else ref.title
                embedding = self.embedding_model.encode(text)
                ml_data['embedding'] = embedding.tolist()
                ml_data['embedding_model'] = settings.ML_EMBEDDING_MODEL
            
            # Calcular similitud con query si hay embedding
            if query_embedding is not None and 'embedding' in ml_data:
                similarity = self._calculate_similarity(
                    query_embedding, 
                    ml_data['embedding']
                )
                ml_data['similarity_score'] = similarity
        
        if 'tags' in self.config.ml_features:
            tags = self._generate_tags(ref.product)
            if tags:
                ml_data['ml_tags'] = tags
        
        # 🔥 Crear referencia mejorada con ML
        if ml_data:
            ml_score = ml_data.get('similarity_score', 0.0) or ml_data.get('category_confidence', 0.0)
            
            # Usar la función de conveniencia de product_reference
            enhanced_ref = create_ml_enhanced_reference(
                product=ref.product,
                ml_score=ml_score,
                ml_data=ml_data
            )
            
            # Preservar score original
            enhanced_ref.score = ref.score
            
            return enhanced_ref
        
        return ref
    
    def _predict_category(self, product: Product) -> Optional[str]:
        """Predice categoría usando configuración del sistema."""
        if not product or not product.title:
            return None
        
        text = f"{product.title} {product.description or ''}".lower()
        
        # Buscar coincidencias con categorías del sistema
        for category in settings.ML_CATEGORIES:
            if category.lower() in text:
                return category
        
        # Si no encuentra, usar categoría principal si existe
        return product.main_category
    
    def _extract_entities(self, product: Product) -> Dict[str, List[str]]:
        """Extrae entidades del producto."""
        entities = {
            "PRODUCT": [],
            "BRAND": [],
            "CATEGORY": []
        }
        
        text = f"{product.title} {product.description or ''}"
        
        # Extracción simple de entidades
        import re
        # Patrón para marcas (palabras con mayúscula)
        brand_pattern = r'\b[A-Z][a-z]+\b'
        brands = re.findall(brand_pattern, text)
        entities["BRAND"] = list(set(brands))[:5]
        
        # Palabras clave de producto
        product_keywords = ['pro', 'max', 'plus', 'mini', 'ultra', 'lite']
        words = text.lower().split()
        for word in words:
            if len(word) > 3 and word not in ['this', 'that', 'with', 'from']:
                entities["PRODUCT"].append(word)
        
        entities["PRODUCT"] = list(set(entities["PRODUCT"]))[:10]
        
        return entities
    
    def _generate_tags(self, product: Product) -> List[str]:
        """Genera tags automáticos para el producto."""
        tags = []
        
        if product.title:
            # Extraer palabras clave del título
            import re
            words = re.findall(r'\b[a-z]{3,}\b', product.title.lower())
            tags.extend(words[:5])
        
        if product.main_category:
            tags.append(product.main_category.lower())
        
        if hasattr(product, 'ml_tags') and product.ml_tags:
            tags.extend(product.ml_tags[:3])
        
        return list(set(tags))[:8]
    
    def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Obtiene embedding de la query."""
        if query in self._query_cache:
            return self._query_cache[query]
        
        if not self.config.use_ml_embeddings or not self.embedding_model:
            return None
        
        try:
            embedding = self.embedding_model.encode(query)
            self._query_cache[query] = embedding.tolist()
            return embedding.tolist()
        except Exception as e:
            self.logger.warning(f"⚠️ Error generando embedding para query: {e}")
            return None
    
    def _calculate_similarity(self, 
                             embedding1: List[float], 
                             embedding2: List[float]) -> float:
        """Calcula similitud coseno entre embeddings."""
        try:
            import numpy as np
            
            v1 = np.array(embedding1)
            v2 = np.array(embedding2)
            
            # Normalizar vectores
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calcular similitud coseno
            similarity = np.dot(v1, v2) / (norm1 * norm2)
            
            # Asegurar valor entre 0 y 1
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculando similitud: {e}")
            return 0.0
    
    def _calculate_ml_score(self, 
                           ref: ProductReference, 
                           query_embedding: List[float]) -> float:
        """Calcula puntaje ML combinado para un producto."""
        if not ref.is_ml_processed:
            return ref.score
        
        base_score = ref.score
        ml_bonus = 0.0
        
        # Bonificación por similitud ML
        similarity = ref.ml_features.get('similarity_score')
        if similarity:
            ml_bonus += similarity * self.config.ml_embedding_weight
        
        # Bonificación por categoría predicha
        if 'predicted_category' in ref.ml_features:
            ml_bonus += 0.1 * self.config.ml_embedding_weight
        
        # Combinar scores
        return base_score * (1 - self.config.ml_embedding_weight) + ml_bonus
    
    def _rerank_results(self, 
                       results: List[ProductReference], 
                       query: str, 
                       user_id: Optional[str] = None) -> List[ProductReference]:
        """Aplica re-ranking a los resultados con RLHF y Collaborative Filter."""
        if not results:
            return []
        
        reranked = []
        
        for ref in results[:self.config.max_retrieved]:
            base_score = ref.score
            
            # 🔥 Aplicar RLHF scoring si disponible
            rlhf_score = 0.0
            if self.rlhf_model:
                text = ref.title if hasattr(ref, 'title') else ""
                rlhf_score = self._score_with_rlhf(query, text)
            
            # 🔥 Aplicar Collaborative Filter si hay usuario
            collab_score = 0.0
            if user_id and self._collaborative_filter:
                collab_scores = self._collaborative_filter.get_collaborative_scores(
                    user_id, 
                    [ref.id]
                )
                collab_score = collab_scores.get(ref.id, 0.0)
            
            # 🔥 Combinar scores (60% base, 20% RLHF, 20% Collaborative)
            final_score = (
                base_score * 0.6 +
                rlhf_score * 0.2 +
                collab_score * 0.2
            )
            
            # Crear copia con nuevo score
            new_ref = ProductReference(
                id=ref.id,
                product=ref.product,
                score=final_score,
                source=ref.source,
                confidence=ref.confidence,
                metadata=ref.metadata.copy(),
                ml_features=ref.ml_features.copy()
            )
            reranked.append(new_ref)
        
        # Ordenar
        reranked.sort(key=lambda x: x.score, reverse=True)
        final_results = reranked[:self.config.max_final]
        
        logger.info(f"🔄 Re-ranking aplicado: RLHF={self.rlhf_model is not None}, CF={collab_score>0}")
        return final_results
    
    # 🔥 NUEVO: Método para usar RLHF en scoring
    def _apply_rlhf_scoring(self, query: str, references: List[ProductReference]) -> Dict[str, float]:
        """Aplica scoring RLHF a las referencias"""
        if not self.rlhf_model or not references:
            return {}
        
        scores = {}
        try:
            for ref in references:
                if hasattr(ref, 'text'):
                    text = ref.text
                elif hasattr(ref, 'title'):
                    text = ref.title
                else:
                    continue
                
                # Puntuar con modelo RLHF
                score = self._score_with_rlhf(query, text)
                scores[ref.id] = score
            
            logger.debug(f"RLHF scoring aplicado a {len(scores)} productos")
            return scores
            
        except Exception as e:
            logger.warning(f"⚠️ Error en RLHF scoring: {e}")
            return {}
    
    def _score_with_rlhf(self, query: str, response: str) -> float:
        """Usa modelo RLHF para puntuar respuesta"""
        try:
            if not self.rlhf_model:
                return 0.5  # Score neutral
            
            # Tokenizar
            inputs = self.rlhf_model.tokenizer(
                f"Query: {query} Response: {response}",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(self.rlhf_model.device)
            
            # Predecir
            with torch.no_grad():
                outputs = self.rlhf_model.model(**inputs)
                score = torch.sigmoid(outputs.logits).item()
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5
    
    def _calculate_rerank_score(self, 
                               ref: ProductReference, 
                               query: str, 
                               user_id: Optional[str] = None) -> float:
        """Calcula score de re-ranking combinando múltiples factores."""
        base_score = ref.score
        
        # Factor de popularidad
        popularity_score = self._calculate_popularity_score(ref)
        
        # Factor de diversidad (evitar productos similares)
        diversity_score = self._calculate_diversity_score(ref, query)
        
        # Factor de novedad
        freshness_score = self._calculate_freshness_score(ref)
        
        # Factor personalizado si hay usuario
        personalization_score = 0.0
        if user_id:
            personalization_score = self._calculate_personalization_score(ref, user_id)
        
        # Combinar scores con ponderaciones
        final_score = (
            base_score * self.config.semantic_weight +
            popularity_score * self.config.popularity_weight +
            diversity_score * self.config.diversity_weight +
            freshness_score * self.config.freshness_weight +
            personalization_score * 0.2  # Peso fijo para personalización
        )
        
        return final_score
    
    def _calculate_popularity_score(self, ref: ProductReference) -> float:
        """Calcula score de popularidad basado en rating."""
        # Valor por defecto si no hay producto
        if not ref or not ref.product:
            return 0.5
        
        # Valores seguros
        rating = getattr(ref.product, 'average_rating', 0.0) or 0.0
        rating_count = getattr(ref.product, 'rating_count', 0) or 0
        
        # Convertir a números
        try:
            rating_num = float(rating)
            count_num = int(rating_count)
        except (ValueError, TypeError):
            return 0.5
        
        # Lógica de cálculo
        if count_num > 100:
            return min(1.0, rating_num / 5.0)
        elif count_num > 10:
            return (rating_num / 5.0) * 0.8
        else:
            return 0.5  # Valor neutral para pocas o ninguna review
    
    def _calculate_diversity_score(self, 
                                  ref: ProductReference, 
                                  query: str) -> float:
        """Calcula score de diversidad para evitar resultados similares."""
        # Por ahora, implementación simple
        # En una implementación real, se compararía con otros resultados
        return 0.7
    
    def _calculate_freshness_score(self, ref: ProductReference) -> float:
        """Calcula score de novedad/actualidad."""
        # Por ahora, implementación simple
        return 0.8
    
    def _calculate_personalization_score(self, 
                                        ref: ProductReference, 
                                        user_id: str) -> float:
        """Calcula score de personalización basado en historial del usuario."""
        # Por ahora, implementación simple
        # En una implementación real, se consultaría el historial del usuario
        return 0.6
    
    def _generate_answer(self, 
                        query: str, 
                        products: List[ProductReference]) -> str:
        """Genera respuesta usando LLM o plantilla simple."""
        # Si hay LLM disponible, usarlo
        if self.config.local_llm_enabled and self.llm_client and products:
            try:
                # Construir contexto con productos
                context = self._build_context_for_llm(products)
                
                prompt = f"""
                Eres un asistente de recomendaciones de Amazon.
                Usuario pregunta: "{query}"
                
                Productos disponibles para recomendar:
                {context}
                
                Genera una respuesta útil y natural que recomiende los productos más relevantes.
                Incluye detalles específicos de los productos como precio, características y por qué son relevantes.
                """
                
                response = self.llm_client.generate(prompt)
                return response.strip()
                
            except Exception as e:
                self.logger.warning(f"⚠️ Error generando respuesta con LLM: {e}")
        
        # Fallback a plantilla simple
        return self._generate_template_answer(query, products)
    
    def _build_context_for_llm(self, products: List[ProductReference]) -> str:
        """Construye contexto para el LLM."""
        context_lines = []
        
        for i, ref in enumerate(products[:3]):  # Limitar a 3 productos para contexto
            title = ref.title[:100]
            price = ref.price
            category = ref.ml_features.get('predicted_category') or ref.metadata.get('main_category', 'Unknown')
            
            line = f"{i+1}. {title} - ${price:.2f} - Categoría: {category}"
            context_lines.append(line)
        
        return "\n".join(context_lines)
    
    def _generate_template_answer(self, query: str, products: List[ProductReference]) -> str:
        """Genera respuesta usando plantilla simple con categorías mejoradas."""
        if not products:
            return f"Lo siento, no encontré productos para '{query}'."
        
        # Construir respuesta con plantilla
        answer_parts = [f"Encontré {len(products)} productos para '{query}':\n"]
        
        for i, ref in enumerate(products[:self.config.max_final]):
            title = ref.title[:80]
            price = ref.price
            
            # 🔥 CORRECCIÓN: Usar el método mejorado de extracción
            category = self._extract_category_for_display(ref, title)
            
            # Añadir emojis basados en categoría
            emoji = self._get_category_emoji(category)
            
            # 🔥 MOSTRAR CATEGORÍA en la respuesta
            answer_parts.append(
                f"{emoji} {i+1}. {title[:60]} "
                f"(💰 ${price:.2f} | 🏷️ {category})"  # ← ¡AHORA MUESTRA CATEGORÍA!
            )
        
        # Añadir recomendación final
        if len(products) > 1:
            best_product = products[0]
            best_title = best_product.title[:60]
            best_price = best_product.price
            
            best_category = self._extract_category_for_display(best_product, best_title)
            best_emoji = self._get_category_emoji(best_category)
            
            answer_parts.append(
                f"\n{best_emoji} **Recomendación principal**: {best_title} "
                f"(💰 ${best_price:.2f} | 🏷️ {best_category})"
            )
        
        return "\n".join(answer_parts)
    def _extract_category_for_display(self, ref: ProductReference, title: str) -> str:
        """Extrae la mejor categoría para mostrar de múltiples fuentes."""
        # 🔥 PRIMERO: Intentar extraer del título (más confiable para Nintendo)
        if 'nintendo' in title.lower() or 'wii' in title.lower() or 'gamecube' in title.lower():
            return 'Video Games'
        
        if 'playstation' in title.lower() or 'ps4' in title.lower() or 'ps5' in title.lower():
            return 'Video Games'
        
        if 'xbox' in title.lower():
            return 'Video Games'
        
        # Luego seguir con la lógica existente...
        category = 'General'
        
        # 1. Intentar de ml_features (predicción ML en tiempo real)
        if ref.ml_features and 'predicted_category' in ref.ml_features:
            category = ref.ml_features['predicted_category']
            self.logger.debug(f"[DEBUG] Usando ml_features: {category}")
        
        # 2. Intentar de metadata (guardado en índice Chroma)
        elif ref.metadata and 'main_category' in ref.metadata:
            category = ref.metadata['main_category']
            self.logger.debug(f"[DEBUG] Usando metadata['main_category']: {category}")
        
        # 3. Intentar de metadata con otro nombre de campo
        elif ref.metadata:
            # Buscar cualquier campo que contenga "categor" en el nombre
            for key in ref.metadata.keys():
                if 'categor' in key.lower():
                    category = ref.metadata[key]
                    self.logger.debug(f"[DEBUG] Usando metadata['{key}']: {category}")
                    break
        
        # 4. Si aún es "General", extraer del título
        if category == 'General':
            extracted = self._extract_category_from_title(title)
            if extracted != 'General':
                category = extracted
                self.logger.debug(f"[DEBUG] Usando extraída del título: {category}")
        
        self.logger.debug(f"[DEBUG] Categoría final: {category}")
        return category

    def _extract_category_from_title(self, title: str) -> str:
        """Extrae categoría del título usando palabras clave."""
        title_lower = title.lower()
        
        # Diccionario de palabras clave
        category_keywords = {
            'Video Games': ['nintendo', 'playstation', 'xbox', 'switch', 'wii', 'gamecube',
                        'ps4', 'ps5', 'xbox one', 'game', 'video game', 'videogame',
                        'switch', 'nes', 'snes', 'n64', 'gameboy', '3ds', 'ds'],
            'Electronics': ['iphone', 'samsung', 'android', 'smartphone', 'phone', 'tablet',
                        'laptop', 'computer', 'pc', 'macbook', 'electronic'],
            'Books': ['book', 'novel', 'author', 'edition', 'hardcover', 'paperback'],
            'Sports': ['wwe', 'fight', 'combat', 'sport', 'fitness', 'gym', 'ball'],
            'Toys': ['toy', 'lego', 'doll', 'action figure', 'puzzle', 'board game'],
            'Home': ['kitchen', 'home', 'furniture', 'appliance', 'cookware'],
            'Clothing': ['shirt', 't-shirt', 'pants', 'jeans', 'dress', 'jacket'],
            'Beauty': ['beauty', 'makeup', 'cosmetic', 'skincare', 'perfume'],
            'Automotive': ['car', 'auto', 'vehicle', 'tire', 'engine', 'motor'],
            'Office': ['office', 'stationery', 'pen', 'pencil', 'notebook']
        }
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return category
        
        return 'General'
    def _get_category_emoji(self, category: str) -> str:
        """Devuelve emoji apropiado para la categoría."""
        emoji_map = {
            'Electronics': '📱',
            'Books': '📚',
            'Clothing': '👕',
            'Home': '🏠',
            'Sports': '⚽',
            'Beauty': '💄',
            'Toys': '🧸',
            'Automotive': '🚗',
            'Office': '💼'
        }
        
        for key, emoji in emoji_map.items():
            if key.lower() in category.lower():
                return emoji
        
        return '📦'  # Emoji por defecto
    
    # --------------------------------------------------
    # Métodos de utilidad y configuración
    # --------------------------------------------------
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de configuración."""
        return {
            "rag_config": {
                "mode": self.config.mode.value,
                "ml_enabled": self.config.ml_enabled,
                "ml_features": self.config.ml_features,
                "local_llm_enabled": self.config.local_llm_enabled,
                "max_final_results": self.config.max_final,
                "enable_reranking": self.config.enable_reranking
            },
            "system_settings": {
                "ML_ENABLED": settings.ML_ENABLED,
                "ML_FEATURES": list(settings.ML_FEATURES),
                "LOCAL_LLM_ENABLED": settings.LOCAL_LLM_ENABLED,
                "LOCAL_LLM_MODEL": settings.LOCAL_LLM_MODEL
            },
            "components": {
                "retriever_loaded": self._retriever is not None,
                "llm_client_loaded": self._llm_client is not None,
                "embedding_model_loaded": self._embedding_model is not None,
                "rlhf_pipeline": self.rlhf_pipeline is not None,
                "collaborative_filter": self._collaborative_filter is not None
            }
        }
    
    def update_config(self, **kwargs) -> None:
        """Actualiza configuración dinámicamente."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"📡 Config actualizada: {key}={value}")
    
    def clear_cache(self) -> None:
        """Limpia caché interno."""
        self._query_cache.clear()
        self.logger.info("🗑️  Cache limpiado")
    
    def test_components(self) -> Dict[str, Any]:
        """Prueba todos los componentes del sistema."""
        results = {
            "retriever": False,
            "llm_client": False,
            "embedding_model": False,
            "rlhf_pipeline": self.rlhf_pipeline is not None,
            "collaborative_filter": self._collaborative_filter is not None,
            "errors": []
        }
        
        # Probar retriever
        try:
            _ = self.retriever
            results["retriever"] = True
        except Exception as e:
            results["errors"].append(f"Retriever: {e}")
        
        # Probar LLM client
        if self.config.local_llm_enabled:
            try:
                _ = self.llm_client
                results["llm_client"] = True
            except Exception as e:
                results["errors"].append(f"LLM Client: {e}")
        
        # Probar embedding model
        if self.config.use_ml_embeddings:
            try:
                _ = self.embedding_model
                results["embedding_model"] = True
            except Exception as e:
                results["errors"].append(f"Embedding Model: {e}")
        
        return results


# ----------------------------------------------------------
# Funciones de conveniencia
# ----------------------------------------------------------

def create_rag_agent(
    mode: str = "hybrid",
    ml_enabled: Optional[bool] = None,
    local_llm_enabled: Optional[bool] = None
) -> WorkingAdvancedRAGAgent:
    """
    Crea un agente RAG con configuración simplificada.
    
    Args:
        mode: Modo de operación (basic, hybrid, ml_enhanced, llm_enhanced)
        ml_enabled: Habilitar ML (usa settings si es None)
        local_llm_enabled: Habilitar LLM local (usa settings si es None)
        
    Returns:
        WorkingAdvancedRAGAgent configurado
    """
    # Usar configuración del sistema por defecto
    if ml_enabled is None:
        ml_enabled = settings.ML_ENABLED
    if local_llm_enabled is None:
        local_llm_enabled = settings.LOCAL_LLM_ENABLED
    
    # Crear configuración
    config = RAGConfig(
        mode=RAGMode(mode),
        ml_enabled=ml_enabled,
        local_llm_enabled=local_llm_enabled
    )
    
    # Crear agente
    agent = WorkingAdvancedRAGAgent(config=config)
    
    logger.info(f"🧠 RAG Agent creado en modo {mode}")
    logger.info(f"   • ML: {'✅' if ml_enabled else '❌'}")
    logger.info(f"   • LLM Local: {'✅' if local_llm_enabled else '❌'}")
    logger.info(f"   • RLHF: {'✅' if agent.rlhf_pipeline else '❌'}")
    logger.info(f"   • Collaborative Filter: {'✅' if agent._collaborative_filter else '❌'}")
    
    return agent


def test_rag_pipeline(query: str = "smartphone barato") -> Dict[str, Any]:
    """
    Prueba rápida del pipeline RAG.
    
    Args:
        query: Consulta de prueba
        
    Returns:
        Resultados de la prueba
    """
    logger.info(f"🧪 Probando pipeline RAG con query: '{query}'")
    
    try:
        # Crear agente
        agent = create_rag_agent(mode="hybrid")
        
        # Procesar consulta
        result = agent.process_query(query)
        
        # Preparar respuesta de prueba
        test_result = {
            "success": True,
            "query": query,
            "answer_length": len(result.get("answer", "")),
            "products_found": len(result.get("products", [])),
            "processing_time": result.get("stats", {}).get("processing_time", 0),
            "config_summary": agent.get_config_summary()
        }
        
        logger.info(f"✅ Test completado: {test_result['products_found']} productos encontrados")
        return test_result
        
    except Exception as e:
        logger.error(f"❌ Test falló: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


# ----------------------------------------------------------
# Ejecución directa para pruebas
# ----------------------------------------------------------

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧠 WorkingAdvancedRAGAgent - Prueba directa")
    print("="*50)
    
    # Probar configuración
    agent = create_rag_agent(mode="hybrid")
    
    # Mostrar configuración
    config_summary = agent.get_config_summary()
    print(f"\n📋 Configuración:")
    print(f"   • Modo: {config_summary['rag_config']['mode']}")
    print(f"   • ML: {'✅' if config_summary['rag_config']['ml_enabled'] else '❌'}")
    print(f"   • LLM Local: {'✅' if config_summary['rag_config']['local_llm_enabled'] else '❌'}")
    print(f"   • RLHF: {'✅' if config_summary['components']['rlhf_pipeline'] else '❌'}")
    print(f"   • Collaborative Filter: {'✅' if config_summary['components']['collaborative_filter'] else '❌'}")
    
    # Probar componentes
    test_results = agent.test_components()
    print(f"\n🔧 Componentes:")
    print(f"   • Retriever: {'✅' if test_results['retriever'] else '❌'}")
    print(f"   • LLM Client: {'✅' if test_results['llm_client'] else '❌'}")
    print(f"   • Embedding Model: {'✅' if test_results['embedding_model'] else '❌'}")
    print(f"   • RLHF Pipeline: {'✅' if test_results['rlhf_pipeline'] else '❌'}")
    print(f"   • Collaborative Filter: {'✅' if test_results['collaborative_filter'] else '❌'}")
    
    if test_results['errors']:
        print(f"\n⚠️ Errores encontrados:")
        for error in test_results['errors']:
            print(f"   • {error}")
    
    print("\n✅ RAG Agent listo para usar")