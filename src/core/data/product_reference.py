"""
ProductReference - Para estandarizar el manejo de productos en todo el sistema.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict, Literal
import json

@dataclass
class ProductReference:
    """Referencia estandarizada a un producto en el sistema."""
    id: str
    product: Optional[Any] = None  # El objeto Product original
    score: float = 0.0
    source: Literal["rag", "collaborative", "hybrid", "ml"] = "hybrid"
    confidence: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para serialización."""
        result = {
            "id": self.id,
            "title": self.title,
            "score": self.score,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata.copy()
        }
        
        # Añadir información del producto si es serializable
        if self.product and hasattr(self.product, 'to_dict'):
            result["product_data"] = self.product.to_dict()
        elif self.product:
            # Intentar extraer atributos básicos
            product_dict = {}
            for attr in ['title', 'description', 'price', 'category', 'brand']:
                if hasattr(self.product, attr):
                    value = getattr(self.product, attr)
                    if value is not None:
                        product_dict[attr] = value
            if product_dict:
                result["product_data"] = product_dict
        
        return result
    
    def to_json(self) -> str:
        """Convierte a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_product(cls, product, score: float = 0.0, source: str = "rag") -> 'ProductReference':
        """Crea ProductReference desde un objeto Product."""
        if not hasattr(product, 'id'):
            raise ValueError("Product must have 'id' attribute")
        
        metadata = {}
        if hasattr(product, 'to_metadata'):
            metadata = product.to_metadata()
        elif hasattr(product, '__dict__'):
            metadata = {k: v for k, v in product.__dict__.items() 
                       if not k.startswith('_') and not callable(v)}
        
        return cls(
            id=product.id,
            product=product,
            score=score,
            source=source,
            confidence=0.8,  # confianza por defecto
            metadata=metadata
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProductReference':
        """Crea ProductReference desde diccionario."""
        return cls(
            id=data['id'],
            product=data.get('product'),
            score=data.get('score', 0.0),
            source=data.get('source', 'hybrid'),
            confidence=data.get('confidence', 0.0),
            metadata=data.get('metadata', {})
        )
    
    def __str__(self) -> str:
        return f"ProductReference(id={self.id}, title={self.title}, score={self.score:.2f}, source={self.source})"