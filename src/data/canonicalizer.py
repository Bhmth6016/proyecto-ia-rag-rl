# src/data/canonicalizer.py
"""
Canonización de productos - ÚNICO lugar donde se generan embeddings
"""
import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

@dataclass
class CanonicalProduct:
    """Producto canónico con embedding generado una sola vez"""
    id: str
    title: str
    description: str
    price: Optional[float]
    category: str
    rating: Optional[float]
    rating_count: Optional[int]
    
    # Campos calculados
    title_embedding: np.ndarray = field(repr=False)
    content_embedding: np.ndarray = field(repr=False)
    content_hash: str = field(init=False)
    
    def __post_init__(self):
        """Calcula hash del contenido para deduplicación"""
        content_str = f"{self.title}|{self.description}|{self.price}|{self.category}"
        self.content_hash = hashlib.md5(content_str.encode()).hexdigest()
    
    @property
    def features_dict(self) -> Dict[str, Any]:
        """Extrae características para ranking"""
        return {
            "price_available": 1.0 if self.price is not None else 0.0,
            "has_rating": 1.0 if self.rating is not None else 0.0,
            "rating_value": self.rating if self.rating else 0.0,
            "has_rating_count": 1.0 if self.rating_count else 0.0,
            "category": self.category
        }


class ProductCanonicalizer:
    """Canoniza productos y genera embeddings una sola vez"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(f"✅ Canonicalizer inicializado con modelo: {embedding_model}")
    
    def canonicalize(self, raw_product: Dict[str, Any]) -> Optional[CanonicalProduct]:
        """Convierte producto crudo a formato canónico"""
        try:
            # Validar campos mínimos
            title = raw_product.get('title', '').strip()
            if not title or len(title) < 3:
                return None
            
            description = raw_product.get('description', '').strip()
            if not description:
                description = title
            
            # Extraer y limpiar precio
            price = self._extract_price(raw_product.get('price'))
            
            # Categoría
            category = raw_product.get('main_category', 'General')
            if not category:
                category = 'General'
            
            # Rating
            rating = self._extract_rating(raw_product.get('average_rating'))
            rating_count = raw_product.get('rating_count')
            
            # Generar embeddings
            title_embedding = self.embedding_model.encode(title, normalize_embeddings=True)
            content = f"{title} {description}"
            content_embedding = self.embedding_model.encode(content, normalize_embeddings=True)
            
            # ID único
            product_id = raw_product.get('id', f"prod_{hashlib.md5(title.encode()).hexdigest()[:8]}")
            
            return CanonicalProduct(
                id=product_id,
                title=title[:200],
                description=description[:500],
                price=price,
                category=category,
                rating=rating,
                rating_count=rating_count,
                title_embedding=title_embedding,
                content_embedding=content_embedding
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Error canonizando producto: {e}")
            return None
    
    def _extract_price(self, price_data: Any) -> Optional[float]:
        """Extrae precio de forma robusta"""
        if price_data is None:
            return None
        
        try:
            if isinstance(price_data, (int, float)):
                return float(price_data)
            elif isinstance(price_data, str):
                import re
                match = re.search(r'(\d+(?:\.\d+)?)', price_data)
                if match:
                    return float(match.group(1))
        except:
            pass
        
        return None
    
    def _extract_rating(self, rating_data: Any) -> Optional[float]:
        """Extrae rating de forma robusta"""
        if rating_data is None:
            return None
        
        try:
            if isinstance(rating_data, (int, float)):
                rating = float(rating_data)
                return max(0.0, min(5.0, rating))
        except:
            pass
        
        return None
    
    def batch_canonicalize(self, raw_products: List[Dict]) -> List[CanonicalProduct]:
        """Canoniza múltiples productos"""
        canonical = []
        
        for i, raw in enumerate(raw_products):
            if i % 100 == 0:
                logger.info(f"📦 Canonizando producto {i}/{len(raw_products)}")
            
            product = self.canonicalize(raw)
            if product:
                canonical.append(product)
        
        logger.info(f"✅ Canonizados {len(canonical)}/{len(raw_products)} productos")
        return canonical
    
    def save_canonical(self, products: List[CanonicalProduct], path: str):
        """Guarda productos canónicos (sin embeddings)"""
        save_data = []
        
        for product in products:
            save_data.append({
                "id": product.id,
                "title": product.title,
                "description": product.description,
                "price": product.price,
                "category": product.category,
                "rating": product.rating,
                "rating_count": product.rating_count,
                "content_hash": product.content_hash
            })
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Guardados {len(save_data)} productos canónicos en {path}")