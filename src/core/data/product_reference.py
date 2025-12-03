"""
ProductReference - Para estandarizar el manejo de productos en todo el sistema.
Versión corregida con manejo consistente de ML.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict, Literal, List, Union
import json
import logging
import warnings

logger = logging.getLogger(__name__)

# Importar configuración
from src.core.config import settings

@dataclass
class ProductReference:
    """Referencia estandarizada a un producto en el sistema."""
    id: str
    product: Optional[Any] = None  # El objeto Product original
    score: float = 0.0
    source: Literal["rag", "collaborative", "hybrid", "ml", "ml_enhanced", "mixed"] = "hybrid"
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    ml_features: Dict[str, Any] = None  # Features específicas ML
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.ml_features is None:
            self.ml_features = {}
        
        # 🔥 CORRECCIÓN: Normalizar source basado en ML features
        self._normalize_source()
    
    def _normalize_source(self):
        """Normaliza el source basado en features ML."""
        if not self.is_ml_processed:
            return
        
        # Solo modificar si no está ya en un estado ML
        if self.source not in ["ml", "ml_enhanced", "mixed"]:
            if self.ml_features.get('embedding') or self.ml_features.get('similarity_score'):
                self.source = "ml_enhanced"
            elif self.ml_features.get('predicted_category') or self.ml_features.get('extracted_entities'):
                self.source = "ml"
    
    @property
    def title(self) -> str:
        """Obtiene el título del producto si está disponible."""
        if self.product and hasattr(self.product, 'title'):
            return self.product.title
        return self.metadata.get('title', '')
    
    @property
    def price(self) -> float:
        """Obtiene el precio si está disponible."""
        if self.product and hasattr(self.product, 'price'):
            try:
                return float(self.product.price)
            except:
                pass
        return self.metadata.get('price', 0.0)
    
    @property
    def is_ml_processed(self) -> bool:
        """Indica si el producto ha sido procesado por ML."""
        # 🔥 CORRECCIÓN: Verificar múltiples fuentes
        return (
            bool(self.ml_features) or 
            self.source in ["ml", "ml_enhanced", "mixed"] or
            self.metadata.get('ml_processed', False) or
            self.metadata.get('has_embedding', False)
        )
    
    @property
    def ml_confidence(self) -> float:
        """Obtiene la confianza ML si está disponible."""
        return self.ml_features.get('confidence', self.metadata.get('ml_confidence', 0.0))
    
    @property
    def embedding(self) -> Optional[List[float]]:
        """Obtiene el embedding del producto si está disponible."""
        # 🔥 CORRECCIÓN: Buscar en múltiples lugares
        if self.product and hasattr(self.product, 'embedding'):
            return self.product.embedding
        
        # Buscar en ml_features
        if 'embedding' in self.ml_features:
            return self.ml_features['embedding']
        
        # Buscar en metadata
        if 'embedding' in self.metadata:
            embedding = self.metadata['embedding']
            if isinstance(embedding, list):
                return embedding
        
        return None
    
    @property
    def has_embedding(self) -> bool:
        """Verifica si tiene embedding."""
        return self.embedding is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        result = {
            "id": self.id,
            "title": self.title,
            "score": self.score,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata.copy(),
            "ml_features": self.ml_features.copy(),
            "is_ml_processed": self.is_ml_processed,
            "has_embedding": self.has_embedding,
            "ml_confidence": self.ml_confidence
        }
        
        # Añadir información del producto si es serializable
        if self.product:
            product_data = self._extract_product_data(self.product)
            if product_data:
                result["product_data"] = product_data
        
        return result
    
    def to_product(self) -> Optional[Any]:
        """
        Convierte ProductReference de vuelta a Product si es posible.
        
        Returns:
            Objeto Product original o None si no está disponible
        """
        # 🔥 CORRECCIÓN: Priorizar producto original
        if self.product is not None:
            return self.product
        
        # Intentar recrear Product desde metadata
        try:
            from src.core.data.product import Product
            
            # Extraer datos básicos del producto
            product_data = {
                'id': self.id,
                'title': self.title,
                'price': self.price
            }
            
            # Añadir metadata común
            common_fields = ['description', 'brand', 'main_category', 
                           'categories', 'product_type', 'average_rating']
            for field in common_fields:
                if field in self.metadata:
                    product_data[field] = self.metadata[field]
            
            # Crear producto básico
            product = Product(**product_data)
            
            # 🔥 CORRECCIÓN: Transferir información ML si existe
            if self.is_ml_processed:
                product.ml_processed = True
                
                # Transferir features ML
                if 'predicted_category' in self.ml_features:
                    product.predicted_category = self.ml_features['predicted_category']
                if 'extracted_entities' in self.ml_features:
                    product.extracted_entities = self.ml_features['extracted_entities']
                if 'ml_tags' in self.ml_features:
                    product.ml_tags = self.ml_features['ml_tags']
                if 'embedding' in self.ml_features:
                    product.embedding = self.ml_features['embedding']
                    product.embedding_model = self.ml_features.get('embedding_model', 'unknown')
            
            return product
            
        except ImportError:
            warnings.warn("Product module not available, returning None")
            return None
        except Exception as e:
            logger.error(f"Error convirtiendo ProductReference a Product: {e}")
            return None
    
    def _extract_product_data(self, product: Any) -> Dict[str, Any]:
        """Extrae datos del producto de forma segura."""
        try:
            if hasattr(product, 'to_dict'):
                return product.to_dict()
            
            # Extraer atributos básicos
            product_dict = {}
            
            # Atributos básicos
            basic_attrs = ['title', 'description', 'price', 'main_category', 
                          'brand', 'average_rating', 'rating_count']
            for attr in basic_attrs:
                if hasattr(product, attr):
                    value = getattr(product, attr)
                    if value is not None:
                        product_dict[attr] = value
            
            # 🔥 CORRECCIÓN: Extraer metadata ML usando método específico
            ml_data = self._extract_ml_product_data(product)
            if ml_data:
                product_dict.update(ml_data)
            
            return product_dict
            
        except Exception as e:
            logger.error(f"Error extrayendo datos del producto: {e}")
            return {}
    
    def _extract_ml_product_data(self, product: Any) -> Dict[str, Any]:
        """Extrae datos ML específicos del producto."""
        ml_data = {}
        
        try:
            # Verificar si tiene método específico para ML
            if hasattr(product, 'get_ml_metrics'):
                ml_metrics = product.get_ml_metrics()
                if ml_metrics:
                    ml_data['ml_metrics'] = ml_metrics
            
            # Extraer atributos ML individuales
            ml_attrs = ['ml_processed', 'predicted_category', 'extracted_entities',
                       'ml_tags', 'embedding', 'embedding_model', 'similarity_score']
            
            for attr in ml_attrs:
                if hasattr(product, attr):
                    value = getattr(product, attr)
                    if value is not None:
                        ml_data[attr] = value
            
            # Extraer metadata adicional
            if hasattr(product, 'to_metadata'):
                metadata = product.to_metadata()
                ml_fields = ['has_embedding', 'ml_tags', 'predicted_category', 'extracted_entities']
                for field in ml_fields:
                    if field in metadata and metadata[field]:
                        ml_data[field] = metadata[field]
            
        except Exception as e:
            logger.debug(f"Error extrayendo datos ML: {e}")
        
        return ml_data
    
    def to_json(self) -> str:
        """Convierte a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_product(cls, product: Any, 
                    score: float = 0.0, 
                    source: str = "rag",
                    ml_confidence: float = 0.0) -> 'ProductReference':
        """Crea ProductReference desde un objeto Product."""
        if not hasattr(product, 'id'):
            raise ValueError("Product must have 'id' attribute")
        
        metadata = {}
        ml_features = {}
        
        # Extraer metadata básica
        if hasattr(product, 'to_metadata'):
            metadata = product.to_metadata()
        elif hasattr(product, '__dict__'):
            # Filtrar atributos privados y métodos
            metadata = {
                k: v for k, v in product.__dict__.items() 
                if not k.startswith('_') and not callable(v) and not isinstance(v, type)
            }
        
        # 🔥 CORRECCIÓN CRÍTICA: Manejo consistente de ML
        ml_processed = getattr(product, 'ml_processed', False)
        has_embedding = hasattr(product, 'embedding') and product.embedding is not None
        
        # Determinar source basado en características REALES
        if ml_processed or has_embedding:
            # 🔥 NO CAMBIAR source dinámicamente - mantener lo que el usuario especificó
            # Solo marcar como ml_enhanced si el usuario no especificó algo diferente
            if source == "rag" and (has_embedding or getattr(product, 'predicted_category', None)):
                source = "ml_enhanced"
            elif source == "collaborative" and ml_processed:
                source = "mixed"
        
        # Extraer features ML
        ml_features = cls._extract_ml_features(product)
        
        # Actualizar metadata con info ML
        metadata['ml_processed'] = ml_processed
        metadata['has_embedding'] = has_embedding
        
        if hasattr(product, 'predicted_category'):
            metadata['predicted_category'] = product.predicted_category
        
        if has_embedding:
            metadata['embedding_dim'] = len(product.embedding)
        
        # 🔥 CORRECCIÓN: Calcular confianza de forma más inteligente
        confidence = cls._calculate_confidence(product, ml_confidence)
        
        return cls(
            id=product.id,
            product=product,
            score=score,
            source=source,
            confidence=confidence,
            metadata=metadata,
            ml_features=ml_features
        )
    
    @classmethod
    def _calculate_confidence(cls, product: Any, ml_confidence: float) -> float:
        """Calcula confianza combinada de producto y ML."""
        base_confidence = 0.8
        
        # Aumentar confianza si tiene ML
        if getattr(product, 'ml_processed', False):
            base_confidence += 0.1
        
        # Aumentar confianza si tiene embedding
        if hasattr(product, 'embedding') and product.embedding:
            base_confidence += 0.05
        
        # Combinar con confianza ML si está disponible
        if ml_confidence > 0:
            base_confidence = max(base_confidence, ml_confidence)
        
        # Limitar a 0-1
        return min(1.0, max(0.0, base_confidence))
    
    @classmethod
    def _extract_ml_features(cls, product: Any) -> Dict[str, Any]:
        """Extrae features ML del producto."""
        ml_features = {}
        
        try:
            # Extraer atributos ML disponibles
            ml_attributes = [
                'embedding', 'predicted_category', 'extracted_entities',
                'ml_tags', 'similarity_score', 'ml_confidence',
                'embedding_model', 'ner_entities', 'tfidf_tags',
                'category_confidence', 'entity_confidence'
            ]
            
            for attr in ml_attributes:
                if hasattr(product, attr):
                    value = getattr(product, attr)
                    if value is not None:
                        # 🔥 CORRECCIÓN: Limitar tamaño de embeddings en features
                        if attr == 'embedding' and isinstance(value, list) and len(value) > 10:
                            ml_features[attr] = value[:10]  # Solo primeros 10 para features
                        else:
                            ml_features[attr] = value
            
            # Añadir flags
            ml_features['ml_processed'] = getattr(product, 'ml_processed', False)
            ml_features['has_embedding'] = hasattr(product, 'embedding') and bool(product.embedding)
            
        except Exception as e:
            logger.debug(f"Error extrayendo features ML: {e}")
        
        return ml_features
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProductReference':
        """Crea ProductReference desde diccionario."""
        return cls(
            id=data['id'],
            product=data.get('product'),
            score=data.get('score', 0.0),
            source=data.get('source', 'hybrid'),
            confidence=data.get('confidence', 0.0),
            metadata=data.get('metadata', {}),
            ml_features=data.get('ml_features', {})
        )
    
    def update_ml_features(self, new_features: Dict[str, Any]) -> None:
        """Actualiza las features ML con nuevos valores."""
        self.ml_features.update(new_features)
        
        # 🔥 CORRECCIÓN: Solo actualizar source si realmente hay nueva info ML significativa
        significant_features = {'embedding', 'predicted_category', 'similarity_score', 'extracted_entities'}
        has_significant = any(feat in new_features for feat in significant_features)
        
        if has_significant and self.source not in ["ml", "ml_enhanced", "mixed"]:
            self.source = "ml_enhanced"
        
        # Actualizar metadata si es necesario
        if 'embedding' in new_features:
            self.metadata['has_embedding'] = True
        
        logger.debug(f"ProductReference {self.id}: ML features actualizadas")
    
    def get_ml_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de las features ML disponibles."""
        summary = {
            "has_ml_features": bool(self.ml_features),
            "ml_source": self.source if self.source in ["ml", "ml_enhanced", "mixed"] else "none",
            "has_embedding": self.has_embedding,
            "ml_features_count": len(self.ml_features),
            "confidence": self.ml_confidence
        }
        
        # Añadir tipos de features disponibles
        feature_types = set()
        for key in self.ml_features:
            if 'category' in key:
                feature_types.add('category')
            elif 'entity' in key:
                feature_types.add('entities')
            elif 'embedding' in key:
                feature_types.add('embedding')
            elif 'tag' in key:
                feature_types.add('tags')
            elif 'similarity' in key:
                feature_types.add('similarity')
        
        summary["feature_types"] = list(feature_types)
        
        return summary
    
    def __str__(self) -> str:
        ml_indicator = " 🔥" if self.is_ml_processed else ""
        return f"ProductReference(id={self.id}, title={self.title[:30]}..., score={self.score:.2f}, source={self.source}{ml_indicator})"
    
    def __repr__(self) -> str:
        return f"ProductReference(id={self.id}, source={self.source}, ml_processed={self.is_ml_processed})"


# 🔥 CORREGIDO: Funciones utilitarias para manejo de ProductReferences con ML
def filter_by_ml_confidence(references: List[ProductReference], 
                          min_confidence: float = None) -> List[ProductReference]:
    """Filtra ProductReferences por confianza ML mínima."""
    if min_confidence is None:
        # Usar configuración del sistema
        min_confidence = settings.ML_CONFIDENCE_THRESHOLD
    
    return [ref for ref in references if ref.ml_confidence >= min_confidence]


def sort_by_ml_score(references: List[ProductReference], 
                    ml_weight: float = None) -> List[ProductReference]:
    """Ordena ProductReferences combinando score normal y ML."""
    if ml_weight is None:
        ml_weight = settings.ML_WEIGHT
    
    def combined_score(ref: ProductReference) -> float:
        base_score = ref.score
        ml_score = ref.ml_confidence
        
        # Combinar scores con peso ajustable
        return base_score * (1 - ml_weight) + ml_score * ml_weight
    
    return sorted(references, key=combined_score, reverse=True)


def group_by_ml_features(references: List[ProductReference]) -> Dict[str, List[ProductReference]]:
    """Agrupa ProductReferences por tipo de features ML."""
    groups = {
        "with_embedding": [],
        "with_category": [],
        "with_entities": [],
        "with_tags": [],
        "no_ml": []
    }
    
    for ref in references:
        if not ref.is_ml_processed:
            groups["no_ml"].append(ref)
            continue
        
        # Verificar qué features tiene
        has_embedding = ref.has_embedding
        has_category = 'predicted_category' in ref.ml_features
        has_entities = 'extracted_entities' in ref.ml_features
        has_tags = 'ml_tags' in ref.ml_features
        
        if has_embedding:
            groups["with_embedding"].append(ref)
        elif has_category:
            groups["with_category"].append(ref)
        elif has_entities:
            groups["with_entities"].append(ref)
        elif has_tags:
            groups["with_tags"].append(ref)
        else:
            groups["no_ml"].append(ref)
    
    return groups


def create_ml_enhanced_reference(product: Any, 
                                ml_score: float = 0.0, 
                                ml_data: Dict[str, Any] = None) -> ProductReference:
    """
    Crea un ProductReference específicamente para productos procesados por ML.
    """
    if ml_data is None:
        ml_data = {}
    
    # 🔥 CORRECCIÓN: Usar from_product con source ml
    ref = ProductReference.from_product(product, score=ml_score, source="ml")
    
    # Actualizar con datos ML específicos
    if ml_data:
        ref.update_ml_features(ml_data)
        ref.confidence = max(ref.confidence, ml_data.get('confidence', 0.0))
    
    return ref


def merge_ml_references(references: List[ProductReference]) -> ProductReference:
    """
    Combina múltiples ProductReferences (por ejemplo, de diferentes fuentes ML).
    Útil para ensamblar resultados de diferentes modelos ML.
    """
    if not references:
        raise ValueError("No references to merge")
    
    # Tomar el primero como base
    base = references[0]
    
    # Combinar ML features de todos
    all_ml_features = {}
    for ref in references:
        if ref.ml_features:
            all_ml_features.update(ref.ml_features)
    
    # Crear nueva referencia combinada
    merged = ProductReference(
        id=base.id,
        product=base.product,
        score=max(r.score for r in references),
        source="mixed",  # Siempre mixed cuando se combinan
        confidence=max(r.confidence for r in references),
        metadata=base.metadata.copy(),
        ml_features=all_ml_features
    )
    
    return merged