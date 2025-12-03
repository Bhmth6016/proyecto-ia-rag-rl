"""
ProductReference - Para estandarizar el manejo de productos en todo el sistema.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict, Literal, List, Union
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProductReference:
    """Referencia estandarizada a un producto en el sistema."""
    id: str
    product: Optional[Any] = None  # El objeto Product original
    score: float = 0.0
    source: Literal["rag", "collaborative", "hybrid", "ml", "ml_enhanced", "mixed"] = "hybrid"
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    ml_features: Dict[str, Any] = None  # 🔥 NUEVO: Features específicas ML
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.ml_features is None:  # 🔥 NUEVO: Inicializar dict ML
            self.ml_features = {}
    
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
        return bool(self.ml_features) or self.source in ["ml", "ml_enhanced", "mixed"]
    
    @property
    def ml_confidence(self) -> float:
        """Obtiene la confianza ML si está disponible."""
        return self.ml_features.get('confidence', 0.0)
    
    @property
    def embedding(self) -> Optional[List[float]]:
        """Obtiene el embedding del producto si está disponible."""
        if self.product and hasattr(self.product, 'embedding'):
            return self.product.embedding
        return self.metadata.get('embedding') or self.ml_features.get('embedding')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        result = {
            "id": self.id,
            "title": self.title,
            "score": self.score,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata.copy(),
            "ml_features": self.ml_features.copy(),  # 🔥 NUEVO: Incluir features ML
            "is_ml_processed": self.is_ml_processed,
            "has_embedding": self.embedding is not None
        }
        
        # Añadir información del producto si es serializable
        if self.product:
            product_data = self._extract_product_data(self.product)
            if product_data:
                result["product_data"] = product_data
        
        return result
    
    def _extract_product_data(self, product: Any) -> Dict[str, Any]:
        """Extrae datos del producto de forma segura."""
        try:
            if hasattr(product, 'to_dict'):
                return product.to_dict()
            
            # Extraer atributos básicos y ML
            product_dict = {}
            
            # Atributos básicos
            basic_attrs = ['title', 'description', 'price', 'category', 
                          'brand', 'main_category', 'average_rating']
            for attr in basic_attrs:
                if hasattr(product, attr):
                    value = getattr(product, attr)
                    if value is not None:
                        product_dict[attr] = value
            
            # 🔥 NUEVO: Atributos ML
            ml_attrs = ['embedding', 'predicted_category', 'extracted_entities', 
                       'ml_tags', 'similarity_score', 'ml_confidence']
            for attr in ml_attrs:
                if hasattr(product, attr):
                    value = getattr(product, attr)
                    if value is not None:
                        product_dict[attr] = value
            
            # Si el producto tiene metadata específica
            if hasattr(product, 'metadata') and product.metadata:
                product_dict['metadata'] = product.metadata
            
            return product_dict
            
        except Exception as e:
            logger.error(f"Error extrayendo datos del producto: {e}")
            return {}
    
    def to_json(self) -> str:
        """Convierte a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_product(cls, product, score: float = 0.0, source: str = "rag", 
                     ml_confidence: float = 0.0) -> 'ProductReference':
        """Crea ProductReference desde un objeto Product."""
        if not hasattr(product, 'id'):
            raise ValueError("Product must have 'id' attribute")
        
        metadata = {}
        ml_features = {}  # 🔥 NUEVO: Dict para features ML
        
        # Extraer metadata básica
        if hasattr(product, 'to_metadata'):
            metadata = product.to_metadata()
        elif hasattr(product, '__dict__'):
            metadata = {k: v for k, v in product.__dict__.items() 
                       if not k.startswith('_') and not callable(v)}
        
        # 🔥 NUEVO: Incluir información ML
        ml_processed = getattr(product, 'ml_processed', False)
        has_embedding = hasattr(product, 'embedding') and product.embedding
        
        if ml_processed or has_embedding:
            if source == "rag":
                source = "ml_enhanced"
            elif source == "collaborative":
                source = "mixed"
            elif source not in ["ml", "ml_enhanced", "mixed"]:
                source = "ml" if ml_processed else source
            
            # 🔥 NUEVO: Extraer features ML
            ml_features = cls._extract_ml_features(product)
            
            # Añadir metadata ML
            metadata['ml_processed'] = ml_processed
            metadata['has_embedding'] = has_embedding
            
            if hasattr(product, 'predicted_category'):
                metadata['predicted_category'] = product.predicted_category
            
            if has_embedding:
                metadata['embedding_dim'] = len(product.embedding)
                # 🔥 NUEVO: Añadir embedding preview (primeros 5 elementos)
                try:
                    embedding_preview = product.embedding[:5] if isinstance(product.embedding, list) else None
                    if embedding_preview:
                        metadata['embedding_preview'] = embedding_preview
                except:
                    pass
        
        # 🔥 NUEVO: Calcular confianza combinada
        confidence = 0.8  # confianza por defecto
        if ml_confidence > 0:
            confidence = max(confidence, ml_confidence)
        
        # 🔥 NUEVO: Si tiene embedding, aumentar confianza
        if has_embedding:
            confidence = min(1.0, confidence + 0.1)
        
        return cls(
            id=product.id,
            product=product,
            score=score,
            source=source,
            confidence=confidence,
            metadata=metadata,
            ml_features=ml_features  # 🔥 NUEVO: Añadir features ML
        )
    
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
                        ml_features[attr] = value
            
            # 🔥 NUEVO: Calcular métricas ML
            if hasattr(product, 'embedding') and product.embedding:
                ml_features['embedding_dim'] = len(product.embedding)
                ml_features['has_embedding'] = True
            
            if hasattr(product, 'ml_processed'):
                ml_features['ml_processed'] = product.ml_processed
            
            # 🔥 NUEVO: Extraer información de procesamiento ML
            if hasattr(product, 'ml_processing_info'):
                ml_features['processing_info'] = product.ml_processing_info
            
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
            ml_features=data.get('ml_features', {})  # 🔥 NUEVO: Incluir ml_features
        )
    
    def update_ml_features(self, new_features: Dict[str, Any]) -> None:
        """Actualiza las features ML con nuevos valores."""
        self.ml_features.update(new_features)
        
        # 🔥 NUEVO: Actualizar source si hay nuevas features ML
        if new_features and self.source not in ["ml", "ml_enhanced", "mixed"]:
            self.source = "ml_enhanced"
        
        logger.debug(f"ProductReference {self.id}: ML features actualizadas")
    
    def get_ml_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de las features ML disponibles."""
        summary = {
            "has_ml_features": bool(self.ml_features),
            "ml_source": self.source if self.source in ["ml", "ml_enhanced", "mixed"] else "none",
            "has_embedding": self.embedding is not None,
            "ml_features_count": len(self.ml_features),
            "confidence": self.ml_confidence
        }
        
        # Añadir tipos de features disponibles
        feature_types = {}
        for key in self.ml_features:
            if key.endswith('_category'):
                feature_types['category'] = True
            elif key.endswith('_entities'):
                feature_types['entities'] = True
            elif key.endswith('_embedding'):
                feature_types['embedding'] = True
            elif key.endswith('_tags'):
                feature_types['tags'] = True
        
        summary["feature_types"] = list(feature_types.keys())
        
        return summary
    
    def __str__(self) -> str:
        ml_indicator = " 🔥" if self.is_ml_processed else ""
        return f"ProductReference(id={self.id}, title={self.title[:30]}..., score={self.score:.2f}, source={self.source}{ml_indicator})"
    
    def __repr__(self) -> str:
        return f"ProductReference(id={self.id}, source={self.source}, ml_processed={self.is_ml_processed})"


# 🔥 NUEVO: Funciones utilitarias para manejo de ProductReferences con ML
def filter_by_ml_confidence(references: List[ProductReference], 
                          min_confidence: float = 0.5) -> List[ProductReference]:
    """Filtra ProductReferences por confianza ML mínima."""
    return [ref for ref in references if ref.ml_confidence >= min_confidence]


def sort_by_ml_score(references: List[ProductReference], 
                    ml_weight: float = 0.3) -> List[ProductReference]:
    """Ordena ProductReferences combinando score normal y ML."""
    
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
        "no_ml": []
    }
    
    for ref in references:
        if not ref.is_ml_processed:
            groups["no_ml"].append(ref)
            continue
        
        # Verificar qué features tiene
        has_embedding = ref.embedding is not None
        has_category = 'predicted_category' in ref.ml_features
        has_entities = 'extracted_entities' in ref.ml_features
        
        if has_embedding:
            groups["with_embedding"].append(ref)
        elif has_category:
            groups["with_category"].append(ref)
        elif has_entities:
            groups["with_entities"].append(ref)
        else:
            groups["no_ml"].append(ref)
    
    return groups


def create_ml_enhanced_reference(product: Any, ml_score: float = 0.0, 
                                ml_data: Dict[str, Any] = None) -> ProductReference:
    """
    Crea un ProductReference específicamente para productos procesados por ML.
    """
    if ml_data is None:
        ml_data = {}
    
    # Crear referencia base
    ref = ProductReference.from_product(product, score=ml_score, source="ml")
    
    # Actualizar con datos ML específicos
    if ml_data:
        ref.update_ml_features(ml_data)
        ref.confidence = max(ref.confidence, ml_data.get('confidence', 0.0))
    
    return ref