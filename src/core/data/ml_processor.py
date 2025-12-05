# src/core/data/ml_processor.py
"""
Módulo separado para el preprocesador ML 100% LOCAL.
Modelos pequeños que funcionan sin conexión a internet.
Con sistema de cache para modelos descargados.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import hashlib
import re
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Definir constantes locales para evitar importaciones circulares
DEFAULT_CATEGORIES = [
    "Electronics", "Home & Kitchen", "Clothing & Accessories", 
    "Sports & Outdoors", "Books", "Health & Beauty", 
    "Toys & Games", "Automotive", "Office Supplies", "Food & Beverages"
]

# Modelos locales disponibles
LOCAL_EMBEDDING_MODELS = [
    'all-MiniLM-L6-v2',           # 384 dimensiones, rápido, multilingüe
    'all-MiniLM-L12-v2',          # 384 dimensiones, más preciso
    'paraphrase-multilingual-MiniLM-L12-v2',  # Especialmente bueno para español
    'distiluse-base-multilingual-cased-v1',   # Multilingüe, 512 dimensiones
    'paraphrase-multilingual-mpnet-base-v2',  # 768 dimensiones, muy preciso
    'LaBSE',                                 # 768 dimensiones, 109 idiomas
]

LOCAL_ZERO_SHOT_MODELS = [
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "facebook/bart-large-mnli",
    "joeddav/xlm-roberta-large-xnli"  # Modelo multilingüe adicional
]

LOCAL_NER_MODELS = [
    "Davlan/bert-base-multilingual-cased-ner-hrl",
    "dslim/bert-base-NER",
    "Babelscape/wikineural-multilingual-ner"
]


class ProductDataPreprocessor:
    """
    Preprocesador de datos de productos con capacidades ML avanzadas 100% LOCAL.
    Enriquece productos con categorías, entidades, tags y embeddings.
    Con sistema de cache para modelos.
    
    NOTA: Este módulo NO importa Product para evitar dependencias circulares.
    Usa tipos genéricos (Dict[str, Any]) en su lugar.
    """
    
    def __init__(
        self, 
        categories: List[str] = None,
        use_gpu: bool = False,
        tfidf_max_features: int = 50,
        embedding_model: str = 'all-MiniLM-L6-v2',
        zero_shot_model: str = None,
        ner_model: str = None,
        verbose: bool = False,
        max_memory_mb: int = 2048,
        use_cache: bool = True
    ):
        """
        Inicializa el preprocesador con modelos ML 100% locales.
        
        Args:
            categories: Lista de categorías para clasificación zero-shot
            use_gpu: Si es True, intenta usar GPU para inferencia
            tfidf_max_features: Número máximo de features para TF-IDF
            embedding_model: Modelo de Sentence Transformers a usar
            zero_shot_model: Modelo específico para zero-shot
            ner_model: Modelo específico para NER
            verbose: Si es True, muestra logs detallados
            max_memory_mb: Límite máximo de memoria en MB
            use_cache: Si es True, usa sistema de cache para modelos
        """
        self.categories = categories or DEFAULT_CATEGORIES
        self.device = 0 if use_gpu else -1
        self.verbose = verbose
        self.use_cache = use_cache
        
        # Configurar cache de modelos
        self._setup_model_cache()
        
        # Configurar modelos
        self.embedding_model_name = self._validate_model_name(
            embedding_model, 
            LOCAL_EMBEDDING_MODELS, 
            'all-MiniLM-L6-v2'
        )
        
        self.zero_shot_model_name = zero_shot_model or LOCAL_ZERO_SHOT_MODELS[0]
        self.ner_model_name = ner_model or LOCAL_NER_MODELS[0]
        
        if verbose:
            logger.info(f"✅ Usando modelo local: {self.embedding_model_name}")
            logger.info(f"📊 Categorías configuradas: {len(self.categories)}")
            if self.model_cache and use_cache:
                logger.info("🔄 Sistema de cache de modelos activado")
        
        # 🔥 CORRECCIÓN: Importación condicional de librerías ML
        self._import_ml_libraries()
        
        # Inicializar modelos (lazy loading en métodos)
        self._zero_shot_classifier = None
        self._ner_pipeline = None
        self._embedding_model = None
        
        # Inicializar vectorizador TF-IDF solo si sklearn está disponible
        if self.sklearn_available:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                # Usar stopwords en español para TF-IDF
                self.tfidf_vectorizer = TfidfVectorizer(
                    stop_words=None,  # No usar stopwords predefinidas para español
                    max_features=tfidf_max_features,
                    ngram_range=(1, 2),
                    min_df=2
                )
                if verbose:
                    logger.info("✅ Vectorizador TF-IDF inicializado (con soporte español)")
            except ImportError:
                self.tfidf_vectorizer = None
                logger.warning("scikit-learn import falló, TF-IDF deshabilitado")
        else:
            self.tfidf_vectorizer = None
        
        # Cache para embeddings frecuentes
        self._embedding_cache = {}
        
        # Flag para TF-IDF entrenado
        self._tfidf_fitted = False
        
        # Cache de modelos para reutilización
        self._model_cache = {}
        
        # Configuración de memoria
        self.set_max_memory_usage(max_memory_mb)
        
        if verbose:
            logger.info("✅ ProductDataPreprocessor inicializado (100% LOCAL)")
            logger.info(f"📦 Dependencias: transformers={self.transformers_available}, "
                       f"sentence-transformers={self.sentence_transformers_available}, "
                       f"scikit-learn={self.sklearn_available}")

    def _validate_model_name(
        self, 
        model_name: str, 
        available_models: List[str], 
        default_model: str
    ) -> str:
        """Valida que el modelo esté en la lista de disponibles."""
        if model_name not in available_models:
            logger.warning(f"⚠️ Modelo {model_name} no reconocido, usando {default_model}")
            return default_model
        return model_name

    def _setup_model_cache(self):
        """Configura el sistema de cache para modelos."""
        if not self.use_cache:
            self.model_cache = None
            if self.verbose:
                logger.info("⚠️ Sistema de cache desactivado por configuración")
            return
            
        try:
            from src.core.utils.model_cache import ModelCache
            
            # Intentar inicializar el cache
            self.model_cache = ModelCache(verbose=self.verbose)
            
            # Pre-descargar modelos si es verbose
            if self.verbose:
                self.model_cache.pre_download_essential_models()
            
        except ImportError as e:
            self.model_cache = None
            if self.verbose:
                logger.warning(f"⚠️ Model cache no disponible: {e}")
                logger.warning("Usando descarga normal de modelos")
        except Exception as e:
            self.model_cache = None
            logger.error(f"❌ Error inicializando model cache: {e}")
            if self.verbose:
                logger.warning("Usando descarga normal de modelos")

    def _import_ml_libraries(self):
        """Importa librerías ML de forma condicional."""
        self.transformers_available = False
        self.sentence_transformers_available = False
        self.sklearn_available = False
        
        # Intentar importar transformers
        try:
            import transformers
            self.transformers_available = True
        except ImportError:
            logger.warning("transformers no disponible. Zero-shot y NER deshabilitados.")
        
        # Intentar importar sentence-transformers
        try:
            import sentence_transformers
            self.sentence_transformers_available = True
        except ImportError:
            logger.warning("sentence-transformers no disponible. Embeddings deshabilitados.")
        
        # Intentar importar scikit-learn
        try:
            import sklearn
            self.sklearn_available = True
        except ImportError:
            logger.warning("scikit-learn no disponible. TF-IDF deshabilitado.")

    # --------------------------------------------------
    # Propiedades para lazy loading de modelos con cache
    # --------------------------------------------------
    
    @property
    def zero_shot_classifier(self):
        """Obtiene el clasificador zero-shot (lazy loading) con cache"""
        if self._zero_shot_classifier is None and self.transformers_available:
            try:
                from transformers import pipeline
                
                model_name = self.zero_shot_model_name
                
                # Verificar si ya tenemos un modelo en cache
                if model_name in self._model_cache:
                    self._zero_shot_classifier = self._model_cache[model_name]
                    if self.verbose:
                        logger.info(f"🔄 Zero-shot cargado desde cache interno: {model_name}")
                    return self._zero_shot_classifier
                
                # Intentar cargar desde cache del sistema
                model_path = None
                if self.model_cache:
                    model_path = self.model_cache.get_model_path("zero_shot", model_name)
                
                if model_path and os.path.exists(model_path):
                    # Cargar desde cache local
                    if self.verbose:
                        logger.info(f"📂 Cargando zero-shot desde cache: {model_path}")
                    self._zero_shot_classifier = pipeline(
                        "zero-shot-classification", 
                        model=str(model_path),
                        device=self.device,
                        framework="pt"
                    )
                else:
                    # Descargar del hub
                    if self.verbose:
                        logger.info(f"⬇️ Descargando zero-shot: {model_name}")
                    self._zero_shot_classifier = pipeline(
                        "zero-shot-classification", 
                        model=model_name,
                        device=self.device,
                        framework="pt"
                    )
                    
                    # Guardar en cache si está disponible
                    if self.model_cache and model_path:
                        try:
                            self.model_cache.cache_model("zero_shot", model_name, model_path)
                        except Exception as e:
                            logger.warning(f"No se pudo guardar en cache: {e}")
                
                # Guardar en cache interno para reutilización
                self._model_cache[model_name] = self._zero_shot_classifier
                
                if self.verbose:
                    logger.info(f"✅ Zero-shot classifier cargado: {model_name}")
                    
            except Exception as e:
                logger.error(f"❌ Error cargando zero-shot classifier: {e}")
                # Fallback a modelo aún más pequeño
                try:
                    from transformers import pipeline
                    self._zero_shot_classifier = pipeline(
                        "zero-shot-classification", 
                        model="facebook/bart-large-mnli",
                        device=self.device
                    )
                    logger.info("✅ Zero-shot classifier fallback cargado")
                except Exception as e2:
                    logger.error(f"❌ Fallback también falló: {e2}")
                    self._zero_shot_classifier = None
        return self._zero_shot_classifier
    
    @property
    def ner_pipeline(self):
        """Obtiene el pipeline NER (lazy loading) con cache"""
        if self._ner_pipeline is None and self.transformers_available:
            try:
                from transformers import pipeline
                
                model_name = self.ner_model_name
                
                # Verificar cache interno
                if model_name in self._model_cache:
                    self._ner_pipeline = self._model_cache[model_name]
                    if self.verbose:
                        logger.info(f"🔄 NER cargado desde cache interno: {model_name}")
                    return self._ner_pipeline
                
                # Intentar cargar desde cache del sistema
                model_path = None
                if self.model_cache:
                    model_path = self.model_cache.get_model_path("ner", model_name)
                
                if model_path and os.path.exists(model_path):
                    # Cargar desde cache local
                    if self.verbose:
                        logger.info(f"📂 Cargando NER desde cache: {model_path}")
                    self._ner_pipeline = pipeline(
                        "ner", 
                        model=str(model_path),
                        aggregation_strategy="simple",
                        device=self.device
                    )
                else:
                    # Descargar del hub
                    if self.verbose:
                        logger.info(f"⬇️ Descargando NER: {model_name}")
                    self._ner_pipeline = pipeline(
                        "ner", 
                        model=model_name,
                        aggregation_strategy="simple",
                        device=self.device
                    )
                    
                    # Guardar en cache si está disponible
                    if self.model_cache and model_path:
                        try:
                            self.model_cache.cache_model("ner", model_name, model_path)
                        except Exception as e:
                            logger.warning(f"No se pudo guardar en cache: {e}")
                
                # Guardar en cache interno
                self._model_cache[model_name] = self._ner_pipeline
                
                if self.verbose:
                    logger.info(f"✅ NER pipeline cargado: {model_name}")
                    
            except Exception as e:
                logger.error(f"❌ Error cargando NER pipeline: {e}")
                # Fallback a modelo más pequeño
                try:
                    from transformers import pipeline
                    self._ner_pipeline = pipeline(
                        "ner", 
                        model="dslim/bert-base-NER",
                        aggregation_strategy="simple",
                        device=self.device
                    )
                    logger.info("✅ NER pipeline fallback cargado")
                except Exception as e2:
                    logger.error(f"❌ Fallback también falló: {e2}")
                    self._ner_pipeline = None
        return self._ner_pipeline
    
    @property
    def embedding_model(self):
        """Obtiene el modelo de embeddings (lazy loading) con cache"""
        if self._embedding_model is None and self.sentence_transformers_available:
            try:
                from sentence_transformers import SentenceTransformer
                
                model_name = self.embedding_model_name
                
                # Verificar cache interno
                if model_name in self._model_cache:
                    self._embedding_model = self._model_cache[model_name]
                    if self.verbose:
                        logger.info(f"🔄 Embedding model cargado desde cache interno: {model_name}")
                    return self._embedding_model
                
                # Intentar cargar desde cache del sistema
                model_path = None
                if self.model_cache:
                    model_path = self.model_cache.get_model_path("embedding", model_name)
                
                if model_path and os.path.exists(model_path):
                    # Cargar desde cache local
                    if self.verbose:
                        logger.info(f"📂 Cargando embedding model desde cache: {model_path}")
                    self._embedding_model = SentenceTransformer(str(model_path))
                else:
                    # Descargar del hub
                    if self.verbose:
                        logger.info(f"⬇️ Descargando embedding model: {model_name}")
                    self._embedding_model = SentenceTransformer(model_name)
                    
                    # Guardar en cache si está disponible
                    if self.model_cache and model_path:
                        try:
                            self.model_cache.cache_model("embedding", model_name, model_path)
                        except Exception as e:
                            logger.warning(f"No se pudo guardar en cache: {e}")
                
                # Mover a GPU si está configurado
                if self.device == 0:
                    try:
                        self._embedding_model = self._embedding_model.to('cuda')
                    except:
                        logger.warning("⚠️ GPU no disponible, usando CPU")
                
                # Guardar en cache interno
                self._model_cache[model_name] = self._embedding_model
                
                if self.verbose:
                    model_info = self._embedding_model
                    logger.info(f"✅ Embedding model cargado: {model_name}")
                    if hasattr(model_info, 'get_sentence_embedding_dimension'):
                        logger.info(f"📐 Dimensiones: {model_info.get_sentence_embedding_dimension()}")
                    
            except Exception as e:
                logger.error(f"❌ Error cargando embedding model {model_name}: {e}")
                # Fallback a modelo más pequeño
                try:
                    from sentence_transformers import SentenceTransformer
                    self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("✅ Embedding model fallback cargado: all-MiniLM-L6-v2")
                except Exception as e2:
                    logger.error(f"❌ Fallback también falló: {e2}")
                    self._embedding_model = None
        return self._embedding_model
    
    # --------------------------------------------------
    # Métodos de utilidad para cache
    # --------------------------------------------------
    
    def clear_internal_cache(self):
        """Limpia el cache interno de modelos."""
        self._model_cache.clear()
        self._embedding_cache.clear()
        self._zero_shot_classifier = None
        self._ner_pipeline = None
        self._embedding_model = None
        if self.verbose:
            logger.info("🧹 Cache interno limpiado")
    
    def warm_up_models(self):
        """Pre-calienta los modelos para evitar latencia en primera inferencia."""
        if self.verbose:
            logger.info("🔥 Pre-calentando modelos...")
        
        # Pre-cargar todos los modelos
        models_to_warm = []
        
        if self.transformers_available:
            models_to_warm.extend(['zero_shot', 'ner'])
        
        if self.sentence_transformers_available:
            models_to_warm.append('embedding')
        
        for model_type in models_to_warm:
            try:
                if model_type == 'zero_shot':
                    _ = self.zero_shot_classifier
                elif model_type == 'ner':
                    _ = self.ner_pipeline
                elif model_type == 'embedding':
                    _ = self.embedding_model
            except Exception as e:
                logger.warning(f"⚠️ No se pudo pre-cargar {model_type}: {e}")
        
        if self.verbose:
            logger.info("✅ Modelos pre-calentados")
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del cache."""
        stats = {
            "internal_cache_size": len(self._model_cache),
            "embedding_cache_size": len(self._embedding_cache),
            "use_system_cache": self.model_cache is not None,
            "models_loaded": list(self._model_cache.keys())
        }
        
        if self.model_cache:
            stats.update(self.model_cache.get_stats())
        
        return stats
    def preprocess_product(
        self, 
        product_data: Dict[str, Any],
        enable_ml: bool = True
    ) -> Dict[str, Any]:
        """
        Procesa un producto con todas las capacidades ML 100% local.
        
        Args:
            product_data: Diccionario con datos del producto
            enable_ml: Si es False, solo realiza limpieza básica sin ML
            
        Returns:
            Diccionario con producto enriquecido
        """
        try:
            # Limpieza básica siempre
            cleaned_data = self._clean_product_data(product_data)
            
            if not enable_ml:
                cleaned_data['ml_processed'] = False
                return cleaned_data
            
            # Extraer texto para procesamiento
            title = cleaned_data.get('title', '')
            description = cleaned_data.get('description', '')
            
            # Limpiar texto (eliminar caracteres extraños)
            title = self._clean_text(title)
            description = self._clean_text(description)
            
            full_text = f"{title}. {description}".strip()
            
            if not full_text or len(full_text) < 10:
                if self.verbose:
                    logger.warning("⚠️ Producto sin texto suficiente para procesamiento ML")
                cleaned_data['ml_processed'] = False
                return cleaned_data
            
            enriched_data = cleaned_data.copy()
            
            # 1. Clasificación Zero-Shot para categorías
            if self.transformers_available:
                predicted_category = self._predict_category_zero_shot(full_text)
                if predicted_category:
                    enriched_data['predicted_category'] = predicted_category
                    # Añadir a categorías si no existe
                    enriched_data.setdefault('categories', [])
                    if predicted_category not in enriched_data['categories']:
                        enriched_data['categories'].append(predicted_category)
            
            # 2. Extracción de entidades con NER
            if self.transformers_available:
                entities = self._extract_entities_ner(full_text)
                if entities:
                    enriched_data['extracted_entities'] = entities
            
            # 3. Generación de tags con TF-IDF
            if self.sklearn_available and self._tfidf_fitted:
                tags = self._generate_tags_tfidf(full_text)
                if tags:
                    existing_tags = enriched_data.setdefault('tags', [])
                    for tag in tags[:5]:
                        if tag not in existing_tags:
                            existing_tags.append(tag)
            
            # 4. Generación de embedding semántico
            if self.sentence_transformers_available:
                embedding = self._generate_embedding(full_text)
                if embedding is not None:
                    enriched_data['embedding'] = embedding
                    enriched_data['embedding_model'] = self.embedding_model_name
            
            enriched_data['ml_processed'] = True
            enriched_data['ml_timestamp'] = datetime.now().isoformat()
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"❌ Error procesando producto: {e}")
            # Devolver datos limpios pero sin ML
            cleaned_data = self._clean_product_data(product_data)
            cleaned_data['ml_processed'] = False
            cleaned_data['ml_error'] = str(e)
            return cleaned_data
    def process_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa un producto completo con todas las capacidades ML."""
        enriched = product_data.copy()
        
        # Extraer texto para procesamiento
        text_fields = []
        for field in ['name', 'description', 'brand', 'category']:
            if field in product_data and product_data[field]:
                text_fields.append(str(product_data[field]))
        
        if not text_fields:
            return enriched
        
        combined_text = " ".join(text_fields)
        
        # 1. Clasificación de categoría
        classification = self.classify_product(combined_text)
        if classification["labels"]:
            enriched["predicted_category"] = classification["labels"][0]
            enriched["category_confidence"] = float(classification["scores"][0])
        
        # 2. Extracción de entidades
        entities = self.extract_entities(combined_text)
        if entities:
            enriched["entities"] = [
                {"entity": e["word"], "type": e["entity_group"], "score": float(e["score"])}
                for e in entities
            ]
        
        # 3. Generar embedding
        try:
            embedding = self.generate_embeddings(combined_text)
            if embedding.size > 0:
                enriched["embedding"] = embedding[0].tolist()
        except Exception as e:
            logger.warning(f"No se pudo generar embedding: {e}")
        
        # 4. Extraer keywords usando TF-IDF si está disponible
        if self.tfidf_vectorizer and not self._tfidf_fitted:
            # Entrenar TF-IDF con datos disponibles
            try:
                self.tfidf_vectorizer.fit([combined_text])
                self._tfidf_fitted = True
            except:
                pass
        
        if self._tfidf_fitted:
            try:
                tfidf_result = self.tfidf_vectorizer.transform([combined_text])
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                scores = tfidf_result.toarray()[0]
                
                # Obtener top keywords
                top_indices = scores.argsort()[-5:][::-1]
                keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
                
                if keywords:
                    enriched["keywords"] = keywords
            except:
                pass
        
        return enriched
    def batch_process(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Procesa múltiples productos en lote de forma eficiente."""
        results = []
        
        for i, product in enumerate(products):
            if self.verbose and i % 10 == 0:
                logger.info(f"Procesando producto {i+1}/{len(products)}")
            
            try:
                enriched = self.process_product(product)
                results.append(enriched)
            except Exception as e:
                logger.error(f"Error procesando producto {i}: {e}")
                results.append(product.copy())  # Mantener datos originales
        
        return results
    def classify_product(self, product_text: str) -> Dict[str, Any]:
        """Clasifica un producto en categorías usando zero-shot learning."""
        if not self.zero_shot_classifier or not product_text:
            return {"labels": [], "scores": []}
        
        try:
            result = self.zero_shot_classifier(
                product_text, 
                self.categories,
                multi_label=False
            )
            return {
                "labels": result["labels"],
                "scores": result["scores"]
            }
        except Exception as e:
            logger.error(f"Error en clasificación zero-shot: {e}")
            return {"labels": [], "scores": []}
    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Genera embeddings para uno o más textos."""
        if not self.embedding_model:
            raise ValueError("Embedding model no disponible")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Verificar cache
        cached_embeddings = []
        texts_to_process = []
        
        for text in texts:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self._embedding_cache:
                cached_embeddings.append(self._embedding_cache[text_hash])
            else:
                texts_to_process.append(text)
        
        # Procesar textos no cacheados
        if texts_to_process:
            new_embeddings = self.embedding_model.encode(texts_to_process)
            
            # Almacenar en cache
            for text, embedding in zip(texts_to_process, new_embeddings):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                self._embedding_cache[text_hash] = embedding
        else:
            new_embeddings = []
        
        # Combinar resultados
        if cached_embeddings and new_embeddings.size > 0:
            all_embeddings = np.vstack([np.array(cached_embeddings), new_embeddings])
        elif cached_embeddings:
            all_embeddings = np.array(cached_embeddings)
        else:
            all_embeddings = new_embeddings
        
        return all_embeddings
    def _clean_product_data(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Limpia los datos básicos del producto."""
        cleaned = product_data.copy()
        
        # Limpiar campos de texto
        text_fields = ['title', 'description', 'brand', 'model']
        for field in text_fields:
            if field in cleaned and isinstance(cleaned[field], str):
                cleaned[field] = self._clean_text(cleaned[field]).strip()
        
        # Asegurar tipos correctos para listas
        for list_field in ['categories', 'tags', 'images']:
            if list_field in cleaned and not isinstance(cleaned[list_field], list):
                if isinstance(cleaned[list_field], str):
                    # Intentar convertir string separado por comas a lista
                    cleaned[list_field] = [
                        item.strip() 
                        for item in cleaned[list_field].split(',') 
                        if item.strip()
                    ]
                else:
                    cleaned[list_field] = []
        
        return cleaned

    def _clean_text(self, text: str) -> str:
        """Limpia texto eliminando caracteres extraños."""
        if not isinstance(text, str):
            return ""
        
        # Remover múltiples espacios
        cleaned = re.sub(r'\s+', ' ', text)
        # Mantener solo caracteres alfanuméricos y puntuación básica
        cleaned = re.sub(r'[^\w\s.,;:!?¿¡()-]', ' ', cleaned)
        # Remover espacios al inicio y final
        return cleaned.strip()
    
    def _predict_category_zero_shot(self, text: str) -> Optional[str]:
        """Predice categoría usando zero-shot classification local."""
        if not self.zero_shot_classifier or not text.strip():
            return None
        
        try:
            # Limitar texto para evitar problemas de memoria
            max_length = 500
            truncated_text = text[:max_length] if len(text) > max_length else text
            
            result = self.zero_shot_classifier(
                truncated_text, 
                candidate_labels=self.categories, 
                multi_label=False
            )
            
            if result["labels"] and result["scores"]:
                best_idx = np.argmax(result["scores"])
                best_score = result["scores"][best_idx]
                best_label = result["labels"][best_idx]
                
                # Solo devolver si la confianza es razonable
                if best_score > 0.3:
                    return best_label
                else:
                    if self.verbose:
                        logger.debug(f"🔍 Confianza baja para categoría: {best_label} ({best_score:.2f})")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error en zero-shot classification: {e}")
            return None
    
    def _extract_entities_ner(self, text: str) -> Dict[str, List[str]]:
        """Extrae entidades nombradas usando NER local."""
        if not self.ner_pipeline or not text.strip():
            return {}
        
        try:
            # Limitar texto para NER
            max_length = 1000
            truncated_text = text[:max_length] if len(text) > max_length else text
            
            entities = self.ner_pipeline(truncated_text)
            entity_dict = {}
            
            for entity in entities:
                entity_group = entity['entity_group']
                word = entity['word']
                
                # Limpiar palabra
                cleaned_word = word.strip()
                if not cleaned_word or len(cleaned_word) < 2:
                    continue
                
                if entity_group not in entity_dict:
                    entity_dict[entity_group] = []
                
                if cleaned_word not in entity_dict[entity_group]:
                    entity_dict[entity_group].append(cleaned_word)
            
            # Solo mantener grupos relevantes para productos
            relevant_groups = ['ORG', 'PRODUCT', 'MISC', 'PER', 'LOC']
            filtered_dict = {
                group: entity_dict[group] 
                for group in relevant_groups 
                if group in entity_dict
            }
            
            return filtered_dict
            
        except Exception as e:
            logger.error(f"❌ Error en NER extraction: {e}")
            return {}
    
    def fit_tfidf(self, descriptions: List[str]) -> None:
        """Entrena el vectorizador TF-IDF con descripciones de productos."""
        if not self.sklearn_available or self.tfidf_vectorizer is None:
            logger.warning("⚠️ TF-IDF no disponible (scikit-learn no instalado)")
            return
        
        try:
            valid_descriptions = [
                desc for desc in descriptions 
                if desc and isinstance(desc, str) and len(desc.strip()) > 10
            ]
            
            if not valid_descriptions:
                logger.warning("⚠️ No hay descripciones válidas para entrenar TF-IDF")
                return
            
            # Limitar tamaño para entrenamiento eficiente
            if len(valid_descriptions) > 10000:
                valid_descriptions = valid_descriptions[:10000]
                logger.info(f"📊 Limitando TF-IDF a 10,000 descripciones")
            
            self.tfidf_vectorizer.fit(valid_descriptions)
            self._tfidf_fitted = True
            
            if self.verbose:
                vocab_size = len(self.tfidf_vectorizer.get_feature_names_out())
                logger.info(f"✅ TF-IDF entrenado con {len(valid_descriptions)} descripciones")
                logger.info(f"📚 Vocabulario: {vocab_size} palabras")
            
        except Exception as e:
            logger.error(f"❌ Error entrenando TF-IDF: {e}")
            self._tfidf_fitted = False
    
    def _generate_tags_tfidf(self, text: str) -> List[str]:
        """Genera tags usando TF-IDF."""
        if (not self.sklearn_available or 
            self.tfidf_vectorizer is None or 
            not self._tfidf_fitted):
            return []
        
        try:
            vector = self.tfidf_vectorizer.transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = vector.toarray()[0]
            
            # Obtener top 10 scores
            top_indices = tfidf_scores.argsort()[-10:][::-1]
            
            tags = []
            for idx in top_indices:
                score = tfidf_scores[idx]
                if score > 0.01:  # Umbral mínimo
                    tags.append(feature_names[idx])
            
            return tags[:5]  # Limitar a 5 tags
            
        except Exception as e:
            logger.error(f"❌ Error generando tags TF-IDF: {e}")
            return []
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Genera embedding semántico para el texto."""
        if not self.embedding_model or not text.strip():
            return None
        
        # Usar hash como clave de cache
        cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        if cache_key in self._embedding_cache:
            if self.verbose:
                logger.debug(f"🔍 Embedding encontrado en cache para texto de {len(text)} chars")
            return self._embedding_cache[cache_key].copy()
        
        try:
            # Limitar texto para embeddings
            max_length = 512
            truncated_text = text[:max_length] if len(text) > max_length else text
            
            embedding = self.embedding_model.encode([truncated_text])[0]
            
            # Normalizar embedding
            embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
            
            # Guardar en cache (con límite de tamaño)
            embedding_list = embedding_norm.tolist()
            self._embedding_cache[cache_key] = embedding_list
            
            # Limitar tamaño de cache según configuración de memoria
            if len(self._embedding_cache) > self.max_cache_size:
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]
            
            return embedding_list
            
        except Exception as e:
            logger.error(f"❌ Error generando embedding: {e}")
            return None
    
    def preprocess_batch(
        self, 
        products: List[Dict[str, Any]], 
        batch_size: int = None,
        enable_ml: bool = True
    ) -> List[Dict[str, Any]]:
        """Procesa un lote de productos de manera eficiente."""
        if not products:
            logger.warning("⚠️ Lista de productos vacía")
            return []
        
        # Usar batch size configurado o calcular basado en memoria
        if batch_size is None:
            batch_size = self.max_batch_size
        
        total_products = len(products)
        if self.verbose:
            logger.info(f"🚀 Procesando lote de {total_products} productos")
        
        # Entrenar TF-IDF si es necesario
        if enable_ml and self.sklearn_available and not self._tfidf_fitted:
            self._prepare_tfidf(products)
        
        # Procesar en batches para manejar memoria
        processed_products = []
        for i in range(0, total_products, batch_size):
            batch = products[i:i + batch_size]
            
            try:
                for product in batch:
                    processed = self.preprocess_product(product, enable_ml=enable_ml)
                    processed_products.append(processed)
                    
                    if self.verbose and len(processed_products) % 100 == 0:
                        logger.info(
                            f"📦 Procesados {len(processed_products)}/{total_products} "
                            f"({(len(processed_products)/total_products)*100:.1f}%)"
                        )
                        
            except Exception as e:
                logger.error(f"❌ Error procesando batch {i//batch_size}: {e}")
                # Procesar individualmente en caso de error
                for product in batch:
                    try:
                        processed = self.preprocess_product(product, enable_ml=False)
                        processed_products.append(processed)
                    except:
                        processed_products.append(product)
        
        return self._generate_statistics(processed_products)

    def _prepare_tfidf(self, products: List[Dict[str, Any]]) -> None:
        """Prepara descripciones para TF-IDF."""
        descriptions = []
        for product in products:
            title = product.get('title', '')
            desc = product.get('description', '')
            full_text = f"{title}. {desc}".strip()
            if full_text and len(full_text) > 10:
                descriptions.append(full_text)
        
        if descriptions:
            if self.verbose:
                logger.info(f"📊 Entrenando TF-IDF con {len(descriptions)} descripciones")
            self.fit_tfidf(descriptions)

    def _generate_statistics(self, processed_products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Genera estadísticas del procesamiento y retorna productos procesados."""
        if self.verbose and processed_products:
            # Estadísticas del procesamiento
            ml_processed = sum(1 for p in processed_products if p.get('ml_processed', False))
            embeddings = sum(1 for p in processed_products if 'embedding' in p)
            categories = sum(1 for p in processed_products if 'predicted_category' in p)
            
            logger.info(f"✅ Procesamiento completado:")
            logger.info(f"   • ML procesados: {ml_processed}/{len(processed_products)}")
            logger.info(f"   • Con embeddings: {embeddings}")
            logger.info(f"   • Con categorías: {categories}")
        
        return processed_products
    
    # --------------------------------------------------
    # Métodos de utilidad y configuración
    # --------------------------------------------------
    
    def set_max_memory_usage(self, max_mb: int = 2048) -> None:
        """Configura límites de memoria para procesamiento."""
        self.max_cache_size = max_mb // 4  # 25% para cache (en número de embeddings)
        self.max_batch_size = max_mb // 20  # Tamaño de batch basado en memoria
        
        if self.verbose:
            logger.info(f"💾 Configurado límite de memoria: {max_mb}MB")
            logger.info(f"   • Max cache: {self.max_cache_size} embeddings")
            logger.info(f"   • Max batch size: {self.max_batch_size} productos")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtiene información sobre los modelos cargados."""
        info = {
            'transformers_available': self.transformers_available,
            'sentence_transformers_available': self.sentence_transformers_available,
            'sklearn_available': self.sklearn_available,
            'zero_shot_classifier_loaded': self._zero_shot_classifier is not None,
            'ner_pipeline_loaded': self._ner_pipeline is not None,
            'embedding_model_loaded': self._embedding_model is not None,
            'tfidf_fitted': self._tfidf_fitted,
            'embedding_model_name': self.embedding_model_name,
            'zero_shot_model_name': self.zero_shot_model_name,
            'ner_model_name': self.ner_model_name,
            'categories_count': len(self.categories),
            'embedding_cache_size': len(self._embedding_cache),
            'verbose_mode': self.verbose,
            'device': 'GPU' if self.device == 0 else 'CPU',
            'max_cache_size': self.max_cache_size,
            'max_batch_size': self.max_batch_size
        }
        
        # Añadir información específica del embedding model si está cargado
        if self._embedding_model is not None:
            try:
                info['embedding_dimensions'] = self._embedding_model.get_sentence_embedding_dimension()
            except:
                info['embedding_dimensions'] = 'unknown'
        
        return info
    
    def clear_cache(self) -> None:
        """Limpia la cache de embeddings."""
        self._embedding_cache.clear()
        logger.info("✅ Cache de embeddings limpiado")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """
        Verifica si todas las dependencias ML están instaladas.
        
        Returns:
            Dict con estado de cada dependencia
        """
        dependencies = {
            'transformers': self.transformers_available,
            'sentence_transformers': self.sentence_transformers_available,
            'scikit_learn': self.sklearn_available,
            'numpy': True,  # Ya importado al inicio
            'models_loaded': self._embedding_model is not None
        }
        
        return dependencies
    
    def process_product_object(self, product: Any) -> Optional[Dict[str, Any]]:
        """
        Procesa un objeto genérico y devuelve diccionario enriquecido.
        
        Args:
            product: Objeto o diccionario con datos de producto
            
        Returns:
            Diccionario enriquecido o None si hay error
        """
        try:
            if isinstance(product, dict):
                product_dict = product
            elif hasattr(product, '__dict__'):
                # Convertir objeto a diccionario
                product_dict = product.__dict__
            elif hasattr(product, 'model_dump'):
                # Si es Pydantic model
                product_dict = product.model_dump()
            else:
                logger.error(f"❌ Tipo de producto no soportado: {type(product)}")
                return None
            
            return self.preprocess_product(product_dict)
            
        except Exception as e:
            logger.error(f"❌ Error procesando objeto producto: {e}")
            return None
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Devuelve diccionario con modelos locales disponibles por tipo."""
        return {
            'embedding_models': LOCAL_EMBEDDING_MODELS,
            'zero_shot_models': LOCAL_ZERO_SHOT_MODELS,
            'ner_models': LOCAL_NER_MODELS
        }
    
    def validate_model_availability(self) -> Dict[str, bool]:
        """
        Valida que los modelos locales estén disponibles para descargar/ejecutar.
        
        Returns:
            Dict con estado de disponibilidad de cada componente
        """
        status = {
            'embedding_model': False,
            'zero_shot_model': False,
            'ner_model': False
        }
        
        # Verificar embedding model
        if self.sentence_transformers_available:
            try:
                from sentence_transformers import SentenceTransformer
                test_model = SentenceTransformer('all-MiniLM-L6-v2')
                status['embedding_model'] = True
                del test_model
            except Exception as e:
                logger.warning(f"⚠️ Embedding model no disponible: {e}")
        
        # Verificar transformers models
        if self.transformers_available:
            try:
                from transformers import pipeline
                # Probar modelo pequeño
                test_pipeline = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
                status['zero_shot_model'] = True
                del test_pipeline
            except Exception as e:
                logger.warning(f"⚠️ Zero-shot model no disponible: {e}")
            
            try:
                from transformers import pipeline
                test_pipeline = pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner-hrl")
                status['ner_model'] = True
                del test_pipeline
            except Exception as e:
                logger.warning(f"⚠️ NER model no disponible: {e}")
        
        return status
    
    def export_configuration(self) -> Dict[str, Any]:
        """Exporta la configuración actual del preprocesador."""
        config = {
            'embedding_model': self.embedding_model_name,
            'zero_shot_model': self.zero_shot_model_name,
            'ner_model': self.ner_model_name,
            'categories': self.categories,
            'categories_count': len(self.categories),
            'device': 'GPU' if self.device == 0 else 'CPU',
            'tfidf_max_features': self.tfidf_vectorizer.max_features 
                if self.tfidf_vectorizer else None,
            'tfidf_fitted': self._tfidf_fitted,
            'cache_size': len(self._embedding_cache),
            'max_cache_size': self.max_cache_size,
            'max_batch_size': self.max_batch_size,
            'verbose': self.verbose,
            'dependencies': self.check_dependencies(),
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat()
        }
        
        return config
    
    
# Función de conveniencia para crear preprocesador
def create_ml_preprocessor(
    use_gpu: bool = False,
    verbose: bool = True,
    use_cache: bool = True
) -> ProductDataPreprocessor:
    """
    Crea un preprocesador ML con configuración por defecto.
    
    Args:
        use_gpu: Si es True, intenta usar GPU
        verbose: Si es True, muestra logs detallados
        use_cache: Si es True, usa sistema de cache para modelos
    
    Returns:
        ProductDataPreprocessor configurado
    """
    return ProductDataPreprocessor(
        use_gpu=use_gpu,
        verbose=verbose,
        use_cache=use_cache,
        embedding_model='all-MiniLM-L6-v2'
    )