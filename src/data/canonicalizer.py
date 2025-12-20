# src/data/canonicalizer.py
"""
Canonización de productos para datos Amazon JSONL
"""
import json
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
    """Canoniza productos específicamente para Amazon JSONL"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(f"✅ Canonicalizer inicializado con modelo: {embedding_model}")
    
    def canonicalize(self, raw_product: Dict[str, Any]) -> Optional[CanonicalProduct]:
        """Convierte producto crudo a formato canónico - Optimizado para Amazon JSONL"""
        try:
            # Extraer ID único (generar uno si no existe)
            product_id = self._extract_id(raw_product)
            if not product_id:
                return None
            
            # Extraer título
            title = self._extract_title(raw_product)
            if not title or len(title.strip()) < 3:
                return None
            
            # Extraer descripción
            description = self._extract_description(raw_product)
            if not description:
                description = title  # Usar título como fallback
            
            # Extraer precio
            price = self._extract_price(raw_product)
            
            # Extraer categoría
            category = self._extract_category(raw_product)
            
            # Extraer rating
            rating = self._extract_rating(raw_product)
            rating_count = self._extract_rating_count(raw_product)
            
            # Generar embeddings
            title_embedding = self.embedding_model.encode(title, normalize_embeddings=True)
            content = f"{title} {description}"
            content_embedding = self.embedding_model.encode(content, normalize_embeddings=True)
            
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
            logger.debug(f"⚠️ Error canonizando producto: {e}")
            return None
    
    def _extract_id(self, raw_product: Dict[str, Any]) -> str:
        """Extrae ID único del producto"""
        # Intentar diferentes campos para ID
        id_fields = ['asin', 'id', 'product_id']
        
        for field in id_fields:
            if field in raw_product and raw_product[field]:
                return str(raw_product[field])
        
        # Si no hay ID, generar uno basado en título hash
        title = self._extract_title(raw_product)
        if title:
            return f"prod_{hashlib.md5(title.encode()).hexdigest()[:12]}"
        
        # Último recurso: ID aleatorio
        import uuid
        return f"prod_{uuid.uuid4().hex[:12]}"
    
    def _extract_title(self, raw_product: Dict[str, Any]) -> str:
        """Extrae título del producto"""
        # Campo principal
        if 'title' in raw_product and raw_product['title']:
            title = raw_product['title']
            if isinstance(title, str):
                return title.strip()
            elif isinstance(title, list):
                return ' '.join([str(t) for t in title if t]).strip()
        
        # Campos alternativos
        alt_fields = ['name', 'product_title', 'product_name']
        for field in alt_fields:
            if field in raw_product and raw_product[field]:
                title = raw_product[field]
                if isinstance(title, str):
                    return title.strip()
        
        return ""
    
    def _extract_description(self, raw_product: Dict[str, Any]) -> str:
        """Extrae descripción del producto"""
        description_parts = []
        
        # Descripción principal
        if 'description' in raw_product and raw_product['description']:
            desc = raw_product['description']
            if isinstance(desc, str):
                description_parts.append(desc.strip())
            elif isinstance(desc, list):
                description_parts.extend([str(d).strip() for d in desc if d])
        
        # Features/bullet points
        if 'features' in raw_product and raw_product['features']:
            features = raw_product['features']
            if isinstance(features, list):
                description_parts.extend([str(f).strip() for f in features if f])
        
        # Bullet points (campo alternativo)
        if 'bullet_points' in raw_product and raw_product['bullet_points']:
            bullets = raw_product['bullet_points']
            if isinstance(bullets, list):
                description_parts.extend([str(b).strip() for b in bullets if b])
        
        # Unir todas las partes
        if description_parts:
            return ' '.join(description_parts)
        
        return ""
    
    def _extract_price(self, raw_product: Dict[str, Any]) -> Optional[float]:
        """Extrae precio del producto"""
        price_data = raw_product.get('price')
        
        if price_data is None:
            return None
        
        try:
            # Si es número directamente
            if isinstance(price_data, (int, float)):
                return float(price_data)
            
            # Si es string, extraer número
            if isinstance(price_data, str):
                # Buscar primer número (entero o decimal)
                match = re.search(r'(\d+[\.,]?\d*)', price_data.replace(',', ''))
                if match:
                    price_str = match.group(1).replace(',', '.')
                    return float(price_str)
            
            # Si es diccionario, buscar valor
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
        """Extrae categoría del producto"""
        # Campo principal
        if 'main_category' in raw_product and raw_product['main_category']:
            cat = raw_product['main_category']
            if isinstance(cat, str):
                return cat.strip()
            elif isinstance(cat, list) and cat:
                return str(cat[0]).strip()
        
        # Campos alternativos
        alt_fields = ['category', 'categories', 'primary_category']
        for field in alt_fields:
            if field in raw_product and raw_product[field]:
                cat = raw_product[field]
                if isinstance(cat, str):
                    return cat.strip()
                elif isinstance(cat, list) and cat:
                    return str(cat[0]).strip()
        
        return "General"
    
    def _extract_rating(self, raw_product: Dict[str, Any]) -> Optional[float]:
        """Extrae rating del producto"""
        rating_fields = ['average_rating', 'rating', 'overall_rating', 'stars', 'review_score']
        
        for field in rating_fields:
            if field in raw_product and raw_product[field] is not None:
                try:
                    rating = float(raw_product[field])
                    # Asegurar que esté en rango 0-5
                    if rating > 5 and rating <= 100:  # Escala 0-100
                        rating = rating / 20
                    elif rating > 10:  # Escala 0-10
                        rating = rating / 2
                    
                    return max(0.0, min(5.0, rating))
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _extract_rating_count(self, raw_product: Dict[str, Any]) -> Optional[int]:
        """Extrae número de ratings"""
        count_fields = ['rating_number', 'rating_count', 'total_ratings', 'num_ratings']
        
        for field in count_fields:
            if field in raw_product and raw_product[field] is not None:
                try:
                    return int(raw_product[field])
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def batch_canonicalize(self, raw_products: List[Dict]) -> List[CanonicalProduct]:
        """Canoniza múltiples productos con logging detallado"""
        canonical = []
        errors = 0
        
        for i, raw in enumerate(raw_products):
            if i % 100 == 0 and i > 0:
                logger.info(f"📦 Procesados {i}/{len(raw_products)} productos...")
            
            product = self.canonicalize(raw)
            if product:
                canonical.append(product)
            else:
                errors += 1
                if errors <= 5:  # Log solo primeros errores
                    logger.debug(f"  Producto {i} no pudo ser canonicalizado")
        
        logger.info(f"✅ Canonizados {len(canonical)}/{len(raw_products)} productos")
        if errors > 0:
            logger.info(f"⚠️  {errors} productos no pudieron ser procesados")
        
        return canonical