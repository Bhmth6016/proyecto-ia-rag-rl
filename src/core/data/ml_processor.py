# src/core/data/ml_processor.py
"""
Módulo separado para el preprocesador ML 100% LOCAL.
Modelos pequeños que funcionan sin conexión a internet.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import hashlib

logger = logging.getLogger(__name__)


class ProductDataPreprocessor:
    """
    Preprocesador de datos de productos con capacidades ML avanzadas 100% LOCAL.
    Enriquece productos con categorías, entidades, tags y embeddings.
    
    NOTA: Este módulo NO importa Product para evitar dependencias circulares.
    Usa tipos genéricos (Dict[str, Any]) en su lugar.
    """
    
    def __init__(
        self, 
        categories: List[str] = None,
        use_gpu: bool = False,
        tfidf_max_features: int = 50,
        embedding_model: str = 'all-MiniLM-L6-v2',  # 🔥 Modelo local por defecto
        verbose: bool = False
    ):
        """
        Inicializa el preprocesador con modelos ML 100% locales.
        
        Args:
            categories: Lista de categorías para clasificación zero-shot
            use_gpu: Si es True, intenta usar GPU para inferencia
            tfidf_max_features: Número máximo de features para TF-IDF
            embedding_model: Modelo de Sentence Transformers a usar (debe ser local)
            verbose: Si es True, muestra logs detallados de inicialización
        """
        self.categories = categories or [
            "Electrónica", "Hogar y Cocina", "Ropa y Accesorios", 
            "Deportes y Aire Libre", "Libros y Medios", "Salud y Belleza", 
            "Juguetes y Juegos", "Automotriz", "Productos de Oficina", "Alimentos y Bebidas"
        ]
        
        self.device = 0 if use_gpu else -1
        self.verbose = verbose
        
        # 🔥 CORRECCIÓN: Solo modelos locales permitidos
        available_models = [
            'all-MiniLM-L6-v2',           # 384 dimensiones, rápido, multilingüe
            'all-MiniLM-L12-v2',          # 384 dimensiones, más preciso
            'paraphrase-multilingual-MiniLM-L12-v2',  # Especialmente bueno para español
            'distiluse-base-multilingual-cased-v1',   # Multilingüe, 512 dimensiones
            'paraphrase-multilingual-mpnet-base-v2',  # 768 dimensiones, muy preciso
            'LaBSE',                                 # 768 dimensiones, 109 idiomas
        ]
        
        if embedding_model not in available_models:
            logger.warning(f"⚠️ Modelo {embedding_model} no reconocido, usando all-MiniLM-L6-v2")
            self.embedding_model_name = 'all-MiniLM-L6-v2'
        else:
            self.embedding_model_name = embedding_model
        
        if verbose:
            logger.info(f"✅ Usando modelo local: {self.embedding_model_name}")
            logger.info(f"📊 Categorías configuradas: {len(self.categories)}")
        
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
        
        if verbose:
            logger.info("✅ ProductDataPreprocessor inicializado (100% LOCAL)")
            logger.info(f"📦 Dependencias: transformers={self.transformers_available}, "
                       f"sentence-transformers={self.sentence_transformers_available}, "
                       f"scikit-learn={self.sklearn_available}")

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
    # Propiedades para lazy loading de modelos
    # --------------------------------------------------
    
    @property
    def zero_shot_classifier(self):
        """Obtiene el clasificador zero-shot (lazy loading) - Modelo local"""
        if self._zero_shot_classifier is None and self.transformers_available:
            try:
                from transformers import pipeline
                
                # 🔥 Usar modelo multilingüe más pequeño y local
                model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
                
                self._zero_shot_classifier = pipeline(
                    "zero-shot-classification", 
                    model=model_name,
                    device=self.device,
                    framework="pt"
                )
                
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
        """Obtiene el pipeline NER (lazy loading) - Modelo local multilingüe"""
        if self._ner_pipeline is None and self.transformers_available:
            try:
                from transformers import pipeline
                
                # 🔥 Usar modelo NER multilingüe
                model_name = "Davlan/bert-base-multilingual-cased-ner-hrl"
                
                self._ner_pipeline = pipeline(
                    "ner", 
                    model=model_name,
                    aggregation_strategy="simple",
                    device=self.device
                )
                
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
        """Obtiene el modelo de embeddings (lazy loading) - Modelo local"""
        if self._embedding_model is None and self.sentence_transformers_available:
            try:
                from sentence_transformers import SentenceTransformer
                
                # 🔥 Usar el modelo especificado en configuración
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                
                if self.device == 0:
                    try:
                        self._embedding_model = self._embedding_model.to('cuda')
                    except:
                        logger.warning("⚠️ GPU no disponible, usando CPU")
                
                if self.verbose:
                    model_info = self._embedding_model
                    logger.info(f"✅ Embedding model cargado: {self.embedding_model_name}")
                    logger.info(f"📐 Dimensiones: {model_info.get_sentence_embedding_dimension()}")
                    
            except Exception as e:
                logger.error(f"❌ Error cargando embedding model {self.embedding_model_name}: {e}")
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
    # Métodos principales de procesamiento
    # --------------------------------------------------
    
    def preprocess_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa un producto con todas las capacidades ML 100% local.
        
        Args:
            product_data: Diccionario con datos del producto
            
        Returns:
            Diccionario con producto enriquecido
        """
        try:
            # Extraer texto para procesamiento
            title = product_data.get('title', '')
            description = product_data.get('description', '')
            
            # Limpiar texto (eliminar caracteres extraños)
            import re
            title = re.sub(r'[^\w\s.,;:!?-]', ' ', title)
            description = re.sub(r'[^\w\s.,;:!?-]', ' ', description)
            
            full_text = f"{title}. {description}".strip()
            
            if not full_text:
                if self.verbose:
                    logger.warning("⚠️ Producto sin texto para procesamiento ML")
                return product_data
            
            enriched_data = product_data.copy()
            
            # 1. Clasificación Zero-Shot para categorías (si está disponible)
            if self.transformers_available and self.zero_shot_classifier:
                predicted_category = self._predict_category_zero_shot(full_text)
                if predicted_category:
                    enriched_data['predicted_category'] = predicted_category
                    enriched_data['ml_confidence'] = 0.8  # Valor por defecto
                    
                    # Añadir a categorías si no existe
                    if 'categories' not in enriched_data or not enriched_data['categories']:
                        enriched_data['categories'] = [predicted_category]
                    elif predicted_category not in enriched_data['categories']:
                        enriched_data['categories'].append(predicted_category)
                    
                    if self.verbose:
                        logger.debug(f"📊 Categoría predicha: {predicted_category}")
            
            # 2. Extracción de entidades con NER (si está disponible)
            if self.transformers_available and self.ner_pipeline:
                entities = self._extract_entities_ner(full_text)
                if entities:
                    enriched_data['extracted_entities'] = entities
                    
                    # Extraer marca y modelo si están disponibles
                    if 'ORG' in entities and entities['ORG']:
                        enriched_data['brand'] = entities['ORG'][0]
                    if 'PRODUCT' in entities and entities['PRODUCT']:
                        enriched_data['model'] = entities['PRODUCT'][0]
                    
                    if self.verbose:
                        logger.debug(f"🏷️ Entidades extraídas: {list(entities.keys())}")
            
            # 3. Generación de tags con TF-IDF (si está disponible)
            if self.sklearn_available and self._tfidf_fitted and self.tfidf_vectorizer is not None:
                tags = self._generate_tags_tfidf(full_text)
                if tags:
                    existing_tags = enriched_data.setdefault('tags', [])
                    for tag in tags[:5]:  # Limitar a 5 tags
                        if tag not in existing_tags:
                            existing_tags.append(tag)
                    
                    if self.verbose:
                        logger.debug(f"🔖 Tags generados: {tags[:3]}")
            
            # 4. Generación de embedding semántico (si está disponible)
            if self.sentence_transformers_available and self.embedding_model:
                embedding = self._generate_embedding(full_text)
                if embedding is not None:
                    enriched_data['embedding'] = embedding
                    enriched_data['embedding_model'] = self.embedding_model_name
                    enriched_data['ml_processed'] = True
                    
                    if self.verbose:
                        logger.debug(f"🔤 Embedding generado: {len(embedding)} dimensiones")
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"❌ Error procesando producto: {e}")
            # Devolver datos originales en caso de error
            return product_data
    
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
            
            # Guardar en cache
            self._embedding_cache[cache_key] = embedding_norm.tolist()
            
            # Limitar tamaño de cache
            if len(self._embedding_cache) > 1000:
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]
            
            return embedding_norm.tolist()
            
        except Exception as e:
            logger.error(f"❌ Error generando embedding: {e}")
            return None
    
    def preprocess_batch(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Procesa un lote de productos de manera eficiente."""
        try:
            if not products:
                logger.warning("⚠️ Lista de productos vacía")
                return []
            
            total_products = len(products)
            if self.verbose:
                logger.info(f"🚀 Procesando lote de {total_products} productos")
            
            # Preparar descripciones para TF-IDF
            if self.sklearn_available and not self._tfidf_fitted:
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
            
            # Procesar productos individualmente
            processed_products = []
            for i, product in enumerate(products):
                try:
                    processed = self.preprocess_product(product)
                    processed_products.append(processed)
                    
                    if self.verbose and (i + 1) % 100 == 0:
                        logger.info(f"📦 Procesados {i + 1}/{total_products} productos ({((i+1)/total_products)*100:.1f}%)")
                        
                except Exception as e:
                    logger.error(f"❌ Error procesando producto {i}: {e}")
                    processed_products.append(product)  # Mantener producto original
            
            if self.verbose:
                # Estadísticas del procesamiento
                ml_processed = sum(1 for p in processed_products if p.get('ml_processed', False))
                embeddings = sum(1 for p in processed_products if 'embedding' in p)
                categories = sum(1 for p in processed_products if 'predicted_category' in p)
                
                logger.info(f"✅ Procesamiento completado:")
                logger.info(f"   • ML procesados: {ml_processed}/{total_products}")
                logger.info(f"   • Con embeddings: {embeddings}")
                logger.info(f"   • Con categorías: {categories}")
            
            return processed_products
            
        except Exception as e:
            logger.error(f"❌ Error en procesamiento por lotes: {e}")
            return products
    
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
            'categories_count': len(self.categories),
            'embedding_cache_size': len(self._embedding_cache),
            'verbose_mode': self.verbose,
            'device': 'GPU' if self.device == 0 else 'CPU'
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
    
    def get_available_models(self) -> List[str]:
        """Devuelve lista de modelos locales disponibles."""
        return [
            'all-MiniLM-L6-v2',
            'all-MiniLM-L12-v2',
            'paraphrase-multilingual-MiniLM-L12-v2',
            'distiluse-base-multilingual-cased-v1',
            'paraphrase-multilingual-mpnet-base-v2',
            'LaBSE'
        ]
    
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