# src/core/data/ml_processor.py
"""
Módulo separado para el preprocesador ML.
Puede instalarse opcionalmente: pip install transformers sentence-transformers scikit-learn
"""

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ProductDataPreprocessor:
    """
    Preprocesador de datos de productos con capacidades ML avanzadas.
    Enriquece productos con categorías, entidades, tags y embeddings.
    """
    
    def __init__(
        self, 
        categories: List[str] = None,
        use_gpu: bool = False,
        tfidf_max_features: int = 50,
        embedding_model: str = 'sentence-transformers/all-mpnet-base-v2'
    ):
        """
        Inicializa el preprocesador con modelos ML.
        
        Args:
            categories: Lista de categorías para clasificación zero-shot
            use_gpu: Si es True, intenta usar GPU para inferencia
            tfidf_max_features: Número máximo de features para TF-IDF
            embedding_model: Modelo de Sentence Transformers a usar
        """
        self.categories = categories or [
            "Electronics", "Home & Kitchen", "Clothing & Accessories", 
            "Sports & Outdoors", "Books", "Health & Beauty", 
            "Toys & Games", "Automotive", "Office Supplies", "Food & Beverages"
        ]
        
        self.device = 0 if use_gpu else -1
        
        # Inicializar modelos (lazy loading en métodos)
        self._zero_shot_classifier = None
        self._ner_pipeline = None
        self._embedding_model = None
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english", 
            max_features=tfidf_max_features,
            ngram_range=(1, 2)
        )
        
        self.embedding_model_name = embedding_model
        
        # Cache para embeddings frecuentes
        self._embedding_cache = {}
        
        logger.info(f"ProductDataPreprocessor inicializado con {len(self.categories)} categorías")
    
    # --------------------------------------------------
    # Propiedades para lazy loading de modelos
    # --------------------------------------------------
    
    @property
    def zero_shot_classifier(self):
        """Obtiene el clasificador zero-shot (lazy loading)"""
        if self._zero_shot_classifier is None:
            try:
                self._zero_shot_classifier = pipeline(
                    "zero-shot-classification", 
                    model="facebook/bart-large-mnli",
                    device=self.device
                )
                logger.info("Zero-shot classifier cargado")
            except Exception as e:
                logger.error(f"Error cargando zero-shot classifier: {e}")
                self._zero_shot_classifier = None
        return self._zero_shot_classifier
    
    @property
    def ner_pipeline(self):
        """Obtiene el pipeline NER (lazy loading)"""
        if self._ner_pipeline is None:
            try:
                self._ner_pipeline = pipeline(
                    "ner", 
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple",
                    device=self.device
                )
                logger.info("NER pipeline cargado")
            except Exception as e:
                logger.error(f"Error cargando NER pipeline: {e}")
                self._ner_pipeline = None
        return self._ner_pipeline
    
    @property
    def embedding_model(self):
        """Obtiene el modelo de embeddings (lazy loading)"""
        if self._embedding_model is None:
            try:
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                if self.device == 0:
                    self._embedding_model = self._embedding_model.to('cuda')
                logger.info(f"Embedding model cargado: {self.embedding_model_name}")
            except Exception as e:
                logger.error(f"Error cargando embedding model: {e}")
                self._embedding_model = None
        return self._embedding_model
    
    # --------------------------------------------------
    # Métodos principales de procesamiento
    # --------------------------------------------------
    
    def preprocess_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa un producto con todas las capacidades ML.
        
        Args:
            product_data: Diccionario con datos del producto
            
        Returns:
            Diccionario con producto enriquecido
        """
        try:
            # Extraer texto para procesamiento
            title = product_data.get('title', '')
            description = product_data.get('description', '')
            full_text = f"{title}. {description}".strip()
            
            if not full_text:
                logger.warning("Producto sin texto para procesamiento ML")
                return product_data
            
            enriched_data = product_data.copy()
            
            # 1. Clasificación Zero-Shot para categorías
            predicted_category = self._predict_category_zero_shot(full_text)
            if predicted_category:
                enriched_data['predicted_category'] = predicted_category
                if 'categories' not in enriched_data or not enriched_data['categories']:
                    enriched_data['categories'] = [predicted_category]
                elif predicted_category not in enriched_data['categories']:
                    enriched_data['categories'].append(predicted_category)
            
            # 2. Extracción de entidades con NER
            entities = self._extract_entities_ner(full_text)
            if entities:
                enriched_data['extracted_entities'] = entities
                if 'ORG' in entities and entities['ORG']:
                    enriched_data['brand'] = entities['ORG'][0]
                if 'PRODUCT' in entities and entities['PRODUCT']:
                    enriched_data['model'] = entities['PRODUCT'][0]
            
            # 3. Generación de tags con TF-IDF
            if hasattr(self, '_tfidf_fitted') and self._tfidf_fitted:
                tags = self._generate_tags_tfidf(full_text)
                if tags:
                    enriched_data.setdefault('tags', []).extend(tags[:5])
            
            # 4. Generación de embedding semántico
            embedding = self._generate_embedding(full_text)
            if embedding is not None:
                enriched_data['embedding'] = embedding
                enriched_data['embedding_model'] = self.embedding_model_name
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"Error procesando producto: {e}")
            return product_data
    
    def _predict_category_zero_shot(self, text: str) -> Optional[str]:
        """Predice categoría usando zero-shot classification."""
        if not self.zero_shot_classifier or not text.strip():
            return None
        
        try:
            result = self.zero_shot_classifier(
                text, 
                candidate_labels=self.categories, 
                multi_label=False
            )
            
            if result["labels"] and result["scores"]:
                best_idx = np.argmax(result["scores"])
                return result["labels"][best_idx]
            
            return None
            
        except Exception as e:
            logger.error(f"Error en zero-shot classification: {e}")
            return None
    
    def _extract_entities_ner(self, text: str) -> Dict[str, List[str]]:
        """Extrae entidades nombradas usando NER."""
        if not self.ner_pipeline or not text.strip():
            return {}
        
        try:
            entities = self.ner_pipeline(text)
            entity_dict = {}
            
            for entity in entities:
                entity_group = entity['entity_group']
                word = entity['word']
                
                cleaned_word = word.strip()
                if not cleaned_word:
                    continue
                
                if entity_group not in entity_dict:
                    entity_dict[entity_group] = []
                
                if cleaned_word not in entity_dict[entity_group]:
                    entity_dict[entity_group].append(cleaned_word)
            
            relevant_groups = ['ORG', 'PRODUCT', 'MISC', 'PER', 'LOC']
            filtered_dict = {
                group: entity_dict[group] 
                for group in relevant_groups 
                if group in entity_dict
            }
            
            return filtered_dict
            
        except Exception as e:
            logger.error(f"Error en NER extraction: {e}")
            return {}
    
    def fit_tfidf(self, descriptions: List[str]) -> None:
        """Entrena el vectorizador TF-IDF con descripciones de productos."""
        try:
            valid_descriptions = [desc for desc in descriptions if desc and isinstance(desc, str)]
            
            if not valid_descriptions:
                logger.warning("No hay descripciones válidas para entrenar TF-IDF")
                return
            
            self.tfidf_vectorizer.fit(valid_descriptions)
            self._tfidf_fitted = True
            
            logger.info(f"TF-IDF entrenado con {len(valid_descriptions)} descripciones")
            
        except Exception as e:
            logger.error(f"Error entrenando TF-IDF: {e}")
            self._tfidf_fitted = False
    
    def _generate_tags_tfidf(self, text: str) -> List[str]:
        """Genera tags usando TF-IDF."""
        if not hasattr(self, '_tfidf_fitted') or not self._tfidf_fitted:
            return []
        
        try:
            vector = self.tfidf_vectorizer.transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = vector.toarray()[0]
            top_indices = tfidf_scores.argsort()[-5:][::-1]
            
            tags = []
            for idx in top_indices:
                if tfidf_scores[idx] > 0:
                    tags.append(feature_names[idx])
            
            return tags
            
        except Exception as e:
            logger.error(f"Error generando tags TF-IDF: {e}")
            return []
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Genera embedding semántico para el texto."""
        if not self.embedding_model or not text.strip():
            return None
        
        cache_key = hash(text)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        try:
            embedding = self.embedding_model.encode([text])[0]
            embedding_norm = embedding / np.linalg.norm(embedding)
            
            self._embedding_cache[cache_key] = embedding_norm.tolist()
            
            if len(self._embedding_cache) > 1000:
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]
            
            return embedding_norm.tolist()
            
        except Exception as e:
            logger.error(f"Error generando embedding: {e}")
            return None
    
    def preprocess_batch(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Procesa un lote de productos de manera eficiente."""
        try:
            descriptions = []
            for product in products:
                title = product.get('title', '')
                desc = product.get('description', '')
                full_text = f"{title}. {desc}".strip()
                if full_text:
                    descriptions.append(full_text)
            
            if descriptions:
                self.fit_tfidf(descriptions)
            
            processed_products = []
            for i, product in enumerate(products):
                try:
                    processed = self.preprocess_product(product)
                    processed_products.append(processed)
                except Exception as e:
                    logger.error(f"Error procesando producto {i}: {e}")
                    processed_products.append(product)
            
            return processed_products
            
        except Exception as e:
            logger.error(f"Error en procesamiento por lotes: {e}")
            return products
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtiene información sobre los modelos cargados."""
        info = {
            'zero_shot_classifier_loaded': self._zero_shot_classifier is not None,
            'ner_pipeline_loaded': self._ner_pipeline is not None,
            'embedding_model_loaded': self._embedding_model is not None,
            'tfidf_fitted': getattr(self, '_tfidf_fitted', False),
            'embedding_model_name': self.embedding_model_name,
            'categories_count': len(self.categories),
            'embedding_cache_size': len(self._embedding_cache)
        }
        
        return info
    
    def clear_cache(self) -> None:
        """Limpia la cache de embeddings."""
        self._embedding_cache.clear()
        logger.info("Cache de embeddings limpiado")