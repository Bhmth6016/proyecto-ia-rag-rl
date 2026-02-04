# src/data/canonicalizer.py - ARREGLADO
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import re

logger = logging.getLogger(__name__)

@dataclass
class CanonicalProduct:
    # Campos REQUERIDOS (sin valor por defecto) - PRIMERO
    id: str
    title: str
    description: str
    category: str
    title_embedding: np.ndarray = field(repr=False)
    content_embedding: np.ndarray = field(repr=False)

    # Campos OPCIONALES (con valor por defecto) - DESPUÉS
    price: Optional[float] = None
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    image_url: Optional[str] = None
    
    # ✅ AÑADIR ESTOS CAMPOS
    ner_attributes: Dict[str, List[str]] = field(default_factory=dict)
    enriched_text: str = ""

    # Campo calculado
    content_hash: str = field(init=False)
    
    def __post_init__(self):
        # Incluir ner_attributes en hash
        content_str = f"{self.title}|{self.description}|{self.price}|{self.category}|{self.image_url}|{bool(self.ner_attributes)}"
        self.content_hash = hashlib.md5(content_str.encode()).hexdigest()
    
    @property
    def features_dict(self) -> Dict[str, Any]:
        return {
            "price_available": 1.0 if self.price is not None else 0.0,
            "has_rating": 1.0 if self.rating is not None else 0.0,
            "rating_value": self.rating if self.rating else 0.0,
            "has_rating_count": 1.0 if self.rating_count else 0.0,
            "has_image": 1.0 if self.image_url else 0.0,  # NUEVO
            "category": self.category
        }

class ProductCanonicalizer:
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(f" Canonicalizer inicializado con modelo: {embedding_model}")
    
    def canonicalize(self, raw_product: Dict[str, Any]) -> Optional[CanonicalProduct]:
        try:
            # OBTENER PARENT_ASIN COMO ID PRINCIPAL
            product_id = self._extract_parent_asin(raw_product)
            if not product_id:
                return None
            
            title = self._extract_title(raw_product)
            if not title or len(title.strip()) < 3:
                return None
            
            description = self._extract_description(raw_product)
            if not description:
                description = title  # Usar título como fallback
            
            category = self._extract_category(raw_product)
            
            # Generar embeddings PRIMERO (son requeridos)
            title_embedding = self.embedding_model.encode(
                title,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            content = f"{title} {description}"
            content_embedding = self.embedding_model.encode(
                content,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            
            # Ahora extraer los campos opcionales
            price = self._extract_price(raw_product)
            rating = self._extract_rating(raw_product)
            rating_count = self._extract_rating_count(raw_product)
            image_url = self._extract_image_url(raw_product)
            
            return CanonicalProduct(
                id=product_id,
                title=title[:200],
                description=description[:500],
                category=category,
                title_embedding=title_embedding,
                content_embedding=content_embedding,
                price=price,
                rating=rating,
                rating_count=rating_count,
                image_url=image_url
            )
            
        except Exception as e:
            logger.debug(f" Error canonizando producto: {e}")
            return None
    
    def _extract_parent_asin(self, raw_product: Dict[str, Any]) -> Optional[str]:
        """Extraer parent_asin como ID principal"""
        # Primero intentar con parent_asin
        if 'parent_asin' in raw_product and raw_product['parent_asin']:
            return str(raw_product['parent_asin']).strip()
        
        # Si no hay parent_asin, usar asin
        if 'asin' in raw_product and raw_product['asin']:
            return str(raw_product['asin']).strip()
        
        # Fallback: otros campos de ID
        id_fields = ['id', 'product_id', 'product_asin']
        for field_name in id_fields:
            if field_name in raw_product and raw_product[field_name]:
                return str(raw_product[field_name])
        
        # Último recurso: generar hash del título
        title = self._extract_title(raw_product)
        if title:
            return f"prod_{hashlib.md5(title.encode()).hexdigest()[:12]}"
        
        return None
    
    def _extract_image_url(self, raw_product: Dict[str, Any]) -> Optional[str]:
        """Extraer URL de imagen miniatura"""
        image_fields = [
            'imageURL',
            'image_url',
            'main_image',
            'thumbnail',
            'imUrl',
            'imURL',
            'image'
        ]
        
        for field_name in image_fields:
            if field_name in raw_product and raw_product[field_name]:
                image_url = raw_product[field_name]
                if isinstance(image_url, str) and image_url.strip():
                    return image_url.strip()
                elif isinstance(image_url, list) and image_url:
                    first_image = image_url[0]
                    if isinstance(first_image, str) and first_image.strip():
                        return first_image.strip()
        
        # También buscar en nested structures
        if 'images' in raw_product and raw_product['images']:
            images = raw_product['images']
            if isinstance(images, list) and images:
                for img in images:
                    if isinstance(img, str) and img.strip():
                        return img.strip()
        
        return None
    
    def _extract_title(self, raw_product: Dict[str, Any]) -> str:
        if 'title' in raw_product and raw_product['title']:
            title = raw_product['title']
            if isinstance(title, str):
                return title.strip()
            elif isinstance(title, list):
                return ' '.join([str(t) for t in title if t]).strip()
        
        alt_fields = ['name', 'product_title', 'product_name']
        for field_name in alt_fields:
            if field_name in raw_product and raw_product[field_name]:
                title = raw_product[field_name]
                if isinstance(title, str):
                    return title.strip()
        
        return ""
    
    def _extract_description(self, raw_product: Dict[str, Any]) -> str:
        description_parts = []
        
        if 'description' in raw_product and raw_product['description']:
            desc = raw_product['description']
            if isinstance(desc, str):
                description_parts.append(desc.strip())
            elif isinstance(desc, list):
                description_parts.extend([str(d).strip() for d in desc if d])
        
        if 'features' in raw_product and raw_product['features']:
            features = raw_product['features']
            if isinstance(features, list):
                description_parts.extend([str(f).strip() for f in features if f])
        
        if 'bullet_points' in raw_product and raw_product['bullet_points']:
            bullets = raw_product['bullet_points']
            if isinstance(bullets, list):
                description_parts.extend([str(b).strip() for b in bullets if b])
        
        if description_parts:
            return ' '.join(description_parts)
        
        return ""
    
    def _extract_price(self, raw_product: Dict[str, Any]) -> Optional[float]:
        price_data = raw_product.get('price')
        
        if price_data is None:
            return None
        
        try:
            if isinstance(price_data, (int, float)):
                return float(price_data)
            
            if isinstance(price_data, str):
                match = re.search(r'(\d+[\.,]?\d*)', price_data.replace(',', ''))
                if match:
                    price_str = match.group(1).replace(',', '.')
                    return float(price_str)
            
            if isinstance(price_data, dict):
                for value in price_data.values():
                    if isinstance(value, (int, float)):
                        return float(value)
                    elif isinstance(value, str):
                        match = re.search(r'(\d+[\.,]?\d*)', value.replace(',', ''))
                        if match:
                            price_str = match.group(1).replace(',', '.')
                            return float(price_str)
        
        except (ValueError, TypeError, AttributeError):
            pass
        
        return None
    
    def _extract_category(self, raw_product: Dict[str, Any]) -> str:
        if 'main_category' in raw_product and raw_product['main_category']:
            cat = raw_product['main_category']
            if isinstance(cat, str):
                return cat.strip()
            elif isinstance(cat, list) and cat:
                return str(cat[0]).strip()
        
        alt_fields = ['category', 'categories', 'primary_category']
        for field_name in alt_fields:
            if field_name in raw_product and raw_product[field_name]:
                cat = raw_product[field_name]
                if isinstance(cat, str):
                    return cat.strip()
                elif isinstance(cat, list) and cat:
                    return str(cat[0]).strip()
        
        return "General"
    
    def _extract_rating(self, raw_product: Dict[str, Any]) -> Optional[float]:
        rating_fields = ['average_rating', 'rating', 'overall_rating', 'stars', 'review_score']
        
        for field_name in rating_fields:
            if field_name in raw_product and raw_product[field_name] is not None:
                try:
                    rating = float(raw_product[field_name])
                    if rating > 5 and rating <= 100:  # Escala 0-100
                        rating = rating / 20
                    elif rating > 10:  # Escala 0-10
                        rating = rating / 2
                    
                    return max(0.0, min(5.0, rating))
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _extract_rating_count(self, raw_product: Dict[str, Any]) -> Optional[int]:
        count_fields = ['rating_number', 'rating_count', 'total_ratings', 'num_ratings']
        
        for field_name in count_fields:
            if field_name in raw_product and raw_product[field_name] is not None:
                try:
                    return int(raw_product[field_name])
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def batch_canonicalize(self, raw_products: List[Dict]) -> List[CanonicalProduct]:
        canonical = []
        errors = 0
        
        for i, raw in enumerate(raw_products):
            if i % 100 == 0 and i > 0:
                logger.info(f" Procesados {i}/{len(raw_products)} productos...")
            
            product = self.canonicalize(raw)
            if product:
                canonical.append(product)
            else:
                errors += 1
                if errors <= 5:  # Log solo primeros errores
                    logger.debug(f"  Producto {i} no pudo ser canonicalizado")
        
        logger.info(f" Canonizados {len(canonical)}/{len(raw_products)} productos")
        if errors > 0:
            logger.info(f"  {errors} productos no pudieron ser procesados")
        
        return canonical