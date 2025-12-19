# src/core/nlp/enrichment.py
import logging
from typing import Dict, List, Optional, Any, Generator
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)

class NLPEnricher:
    """Sistema NLP para enriquecer productos con NER y Zero-Shot"""
    
    def __init__(self, 
            ner_model: Optional[str] = None,
            zero_shot_model: Optional[str] = None,
            device: str = "cuda",
            use_small_models: bool = True):
        
        from src.core.config import settings
        
        self.device = device
        self.use_small_models = use_small_models
        
        # 🔥 USAR MODELOS MÁS PEQUEÑOS POR DEFECTO
        if ner_model is None:
            if use_small_models:
                self.ner_model_name = "dslim/distilbert-NER"  # 66MB
            else:
                self.ner_model_name = getattr(settings, 'NER_MODEL', "dslim/bert-base-NER")
        else:
            self.ner_model_name = ner_model
        
        if zero_shot_model is None:
            if use_small_models:
                self.zero_shot_model_name = "typeform/distilbert-base-uncased-mnli"  # 66MB
            else:
                self.zero_shot_model_name = getattr(settings, 'ZERO_SHOT_MODEL', "facebook/bart-large-mnli")
        else:
            self.zero_shot_model_name = zero_shot_model
        
        # Carga diferida de modelos
        self._ner_pipeline: Optional[Any] = None
        self._zero_shot_pipeline: Optional[Any] = None
        self._initialized = False
        
        logger.info(f"🔧 NLPEnricher configurado (modelos pequeños: {use_small_models})")
        
    def initialize(self, force: bool = False) -> None:
        """Inicializa modelos NLP (lazy loading) - VERSIÓN OPTIMIZADA."""
        if self._initialized and not force:
            return
            
        try:
            # 🔥 SILENCIAR WARNINGS DE HUGGINGFACE
            import os
            import warnings
            warnings.filterwarnings("ignore", message=".*huggingface_hub.*")
            warnings.filterwarnings("ignore", message=".*symlinks.*")
            
            logger.info("🔧 Inicializando sistema NLP...")
            
            # Cargar modelo NER con opciones optimizadas
            self._ner_pipeline = pipeline(
                "ner",
                model=self.ner_model_name,
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1,
                aggregation_strategy="simple",
                batch_size=1  # 🔥 Evitar problemas de memoria
            )
            logger.debug(f"✅ NER model cargado: {self.ner_model_name}")
            
            # Cargar modelo Zero-Shot optimizado
            self._zero_shot_pipeline = pipeline(
                "zero-shot-classification",
                model=self.zero_shot_model_name,
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1
            )
            logger.debug(f"✅ Zero-Shot model cargado: {self.zero_shot_model_name}")
            
            self._initialized = True
            logger.info("✅ Sistema NLP inicializado")
            
            # Restaurar warnings
            warnings.filterwarnings("default")
                
        except Exception as e:
            logger.error(f"❌ Error inicializando NLP: {e}")
            self._initialized = False
            # Restaurar warnings en caso de error
            import warnings
            warnings.filterwarnings("default")
            
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        if not self._initialized:
            self.initialize()
            
        if not self._ner_pipeline or not text:
            return {}
            
        try:
            entities = self._ner_pipeline(text)
            
            if entities is None:  # ← AÑADIR ESTA VALIDACIÓN
                return {}
            
            # Organizar por tipo de entidad
            organized: Dict[str, List[Dict[str, Any]]] = {
                "PRODUCT": [],
                "BRAND": [],
                "CATEGORY": [],
                "ATTRIBUTE": []
            }
            
            for entity in entities:
                # Validar que entity no sea None y tenga los campos necesarios
                if not entity:
                    continue
                    
                if not isinstance(entity, dict):
                    continue
                    
                entity_type = entity.get('entity_group')
                entity_text = entity.get('word')
                confidence = entity.get('score')
                
                # Validar que tengamos los datos mínimos
                if not entity_type or not entity_text or confidence is None:
                    continue
                
                # Mapear tipos de entidad
                if entity_type in ["ORG", "MISC"]:
                    organized["BRAND"].append({
                        "name": entity_text,
                        "confidence": float(confidence)
                    })
                elif entity_type in ["PRODUCT", "OBJ"]:
                    organized["PRODUCT"].append({
                        "name": entity_text,
                        "confidence": float(confidence)
                    })
                else:
                    organized["ATTRIBUTE"].append({
                        "name": entity_text,
                        "type": entity_type,
                        "confidence": float(confidence)
                    })
                    
            return organized
            
        except Exception as e:
            logger.warning(f"Error en NER: {e}")
            return {}
    
    def zero_shot_classify(self, 
                          text: str, 
                          candidate_labels: List[str]) -> Dict[str, Any]:
        """Clasificación Zero-Shot del texto"""
        if not self._initialized:
            self.initialize()
            
        if not self._zero_shot_pipeline or not text or not candidate_labels:
            return {}
            
        try:
            result = self._zero_shot_pipeline(
                text,
                candidate_labels=candidate_labels,
                multi_label=True
            )
            
            # Validar que result sea un diccionario
            if not isinstance(result, dict):
                logger.warning(f"Resultado inesperado de zero_shot_pipeline: {type(result)}")
                return {}
            
            # Organizar resultados
            labels = result.get('labels', [])
            scores = result.get('scores', [])
            
            if not labels or not scores:
                return {}
            
            classification: Dict[str, float] = {}
            for label, score in zip(labels, scores):
                if score > 0.3:  # Umbral mínimo
                    classification[label] = float(score)
                    
            # Encontrar mejor categoría
            if classification:
                best_label = max(classification.items(), key=lambda x: x[1])
                return {
                    "classification": classification,
                    "best_category": best_label[0],
                    "confidence": best_label[1]
                }
            
            return {}
            
        except Exception as e:
            logger.warning(f"Error en Zero-Shot: {e}")
            return {}
    
    def enrich_product(self, 
                      product_data: Dict[str, Any],
                      categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Enriquece un producto con NLP"""
        if not self._initialized:
            self.initialize()
            
        enriched = product_data.copy()
        
        # Crear texto para análisis
        text_parts = [
            product_data.get('title', ''),
            product_data.get('description', ''),
            product_data.get('brand', '')
        ]
        text = " ".join(filter(None, text_parts))
        
        if not text:
            return enriched
            
        # 1. Extraer entidades
        entities = self.extract_entities(text)
        if entities:
            enriched['ner_entities'] = entities
            enriched['has_ner'] = True
            
        # 2. Clasificación Zero-Shot
        if categories:
            zero_shot_result = self.zero_shot_classify(text, categories)
            if zero_shot_result:
                enriched['zero_shot_classification'] = zero_shot_result.get('classification', {})
                enriched['predicted_category'] = zero_shot_result.get('best_category')
                enriched['classification_confidence'] = zero_shot_result.get('confidence')
                enriched['has_zero_shot'] = True
                
        enriched['nlp_processed'] = True
        return enriched
    
    def batch_enrich(self,
                    products_data: List[Dict[str, Any]],
                    categories: Optional[List[str]] = None,
                    batch_size: int = 10) -> List[Dict[str, Any]]:
        """Enriquece un batch de productos"""
        enriched_products = []
        
        for i in range(0, len(products_data), batch_size):
            batch = products_data[i:i + batch_size]
            
            for product in batch:
                try:
                    enriched = self.enrich_product(product, categories)
                    enriched_products.append(enriched)
                except Exception as e:
                    logger.warning(f"Error enriqueciendo producto: {e}")
                    enriched_products.append(product)  # Mantener original
                    
            logger.debug(f"Procesados {min(i + batch_size, len(products_data))}/{len(products_data)} productos")
            
        return enriched_products
    
    def cleanup_memory(self) -> None:
        """Libera memoria de modelos NLP - VERSIÓN MEJORADA."""
        try:
            # 🔥 IMPORTACIÓN SEGURA DE TORCH
            torch_available = 'torch' in globals() or 'torch' in locals()
            
            # Liberar pipelines
            if self._ner_pipeline:
                del self._ner_pipeline
                self._ner_pipeline = None
            
            if self._zero_shot_pipeline:
                del self._zero_shot_pipeline
                self._zero_shot_pipeline = None
            
            # Forzar garbage collection
            import gc
            gc.collect()
            
            # Limpieza de cuda solo si torch está disponible
            if torch_available and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("🧹 Memoria cuda liberada")
            
            self._initialized = False
            logger.info("✅ Memoria NLP liberada")
            
        except Exception as e:
            logger.debug(f"⚠️ Error menor en limpieza NLP: {e}")
    def extract_product_title(self, text: str) -> Dict[str, Any]:
        """
        Extrae componentes para construir un título de producto.
        
        Args:
            text: Texto del producto (descripción, características, etc.)
            
        Returns:
            Dict con componentes para construir título
        """
        if not text:
            return {}
        
        if not self._initialized:
            self.initialize()
        
        components = {
            "brand": None,
            "product_type": None,
            "key_features": [],
            "attributes": []
        }
        
        try:
            # Extraer entidades
            entities = self.extract_entities(text)
            
            # Identificar marca (usar entidad ORG)
            if entities.get("BRAND"):
                brands = entities["BRAND"]
                if brands:
                    # Seleccionar la marca con mayor confianza
                    sorted_brands = sorted(brands, key=lambda x: x.get('confidence', 0), reverse=True)
                    components["brand"] = sorted_brands[0]["name"]
            
            # Identificar tipo de producto
            # Usar Zero-Shot para clasificar tipo de producto
            product_types = [
                "smartphone", "laptop", "tablet", "headphones", "television",
                "book", "novel", "textbook", "magazine",
                "shirt", "pants", "dress", "shoes", "jacket",
                "toy", "game", "puzzle", "board game",
                "kitchen appliance", "furniture", "home decor",
                "sports equipment", "fitness gear",
                "cosmetic", "skincare", "perfume",
                "car accessory", "tool", "electronic device"
            ]
            
            zero_shot_result = self.zero_shot_classify(text, product_types)
            if zero_shot_result:
                classification = zero_shot_result.get('classification', {})
                if classification:
                    # Obtener tipo con mayor score
                    best_type = max(classification.items(), key=lambda x: x[1])[0]
                    components["product_type"] = best_type
            
            # Extraer características clave
            if entities.get("ATTRIBUTE"):
                attributes = entities["ATTRIBUTE"]
                for attr in attributes[:3]:  # Primeros 3 atributos
                    name = attr.get("name", "")
                    if name and len(name) > 2:
                        components["attributes"].append(name)
            
            # Extraer colores y tamaños
            if entities.get("COLOR"):
                colors = [c["name"] for c in entities["COLOR"][:2]]
                components["key_features"].extend(colors)
            
            if entities.get("SIZE"):
                sizes = [s["name"] for s in entities["SIZE"][:2]]
                components["key_features"].extend(sizes)
            
            # Extraer palabras clave del texto
            import re
            words = re.findall(r'\b[A-Z][a-z]+\b', text)
            for word in words[:5]:
                if word not in ["The", "And", "For", "With", "This", "That"]:
                    components["key_features"].append(word.lower())
            
            return components
            
        except Exception as e:
            logger.debug(f"Error extrayendo título: {e}")
            return {}
    def __del__(self) -> None:
        """Destructor para liberar memoria automáticamente."""
        try:
            self.cleanup_memory()
        except:
            pass