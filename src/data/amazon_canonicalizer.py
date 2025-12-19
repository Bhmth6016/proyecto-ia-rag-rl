# src/data/amazon_canonicalizer.py
"""
Canonicalizer especializado para datos de Amazon
"""
import json
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from dataclasses import dataclass, field
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class AmazonProduct:
    """Producto de Amazon en formato canónico"""
    id: str
    title: str
    description: str
    price: Optional[float]
    category: str
    brand: Optional[str]
    rating: Optional[float]
    rating_count: Optional[int]
    features: List[str]
    image_url: Optional[str]
    
    # Campos calculados
    title_embedding: np.ndarray = field(repr=False)
    content_embedding: np.ndarray = field(repr=False)
    content_hash: str = field(init=False)
    
    def __post_init__(self):
        """Calcula hash del contenido para deduplicación"""
        content_str = f"{self.title}|{self.description}|{self.category}|{self.brand}"
        self.content_hash = hashlib.md5(content_str.encode()).hexdigest()
    
    @property
    def features_dict(self) -> Dict[str, Any]:
        """Extrae características para ranking"""
        return {
            "price_available": 1.0 if self.price is not None else 0.0,
            "has_rating": 1.0 if self.rating is not None else 0.0,
            "rating_value": self.rating if self.rating else 0.0,
            "has_rating_count": 1.0 if self.rating_count and self.rating_count > 0 else 0.0,
            "has_brand": 1.0 if self.brand else 0.0,
            "has_features": 1.0 if self.features else 0.0,
            "num_features": min(1.0, len(self.features) / 10),
            "category": self.category
        }


class AmazonCanonicalizer:
    """Canoniza productos de Amazon específicamente"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(f"✅ AmazonCanonicalizer inicializado con modelo: {embedding_model}")
    
    def canonicalize_from_jsonl(self, jsonl_path: str, max_products: int = 1000) -> List[AmazonProduct]:
        """Carga y canoniza productos desde archivo JSONL"""
        products = []
        skipped = 0
        
        logger.info(f"📥 Cargando productos desde: {jsonl_path}")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_products:
                    break
                
                try:
                    raw_data = json.loads(line.strip())
                    product = self._parse_amazon_product(raw_data)
                    
                    if product:
                        products.append(product)
                    else:
                        skipped += 1
                
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                
                if (i + 1) % 100 == 0:
                    logger.info(f"  Procesados: {i + 1} productos")
        
        logger.info(f"✅ Cargados {len(products)} productos, omitidos {skipped}")
        return products
    
    def _parse_amazon_product(self, raw_data: Dict[str, Any]) -> Optional[AmazonProduct]:
        """Parsea un producto crudo de Amazon"""
        try:
            # Extraer campos principales (adaptar según estructura real)
            title = self._extract_field(raw_data, ['title', 'name', 'Title'])
            if not title or len(title) < 3:
                return None
            
            # Descripción
            description = self._extract_field(raw_data, 
                ['description', 'Description', 'desc', 'bullet_points', 'features'])
            
            if not description:
                description = title
            
            # Precio
            price = self._extract_price(raw_data)
            
            # Categoría
            category = self._extract_category(raw_data)
            
            # Brand
            brand = self._extract_field(raw_data, ['brand', 'Brand', 'manufacturer'])
            
            # Rating
            rating = self._extract_rating(raw_data)
            rating_count = self._extract_rating_count(raw_data)
            
            # Features
            features = self._extract_features(raw_data)
            
            # Image URL
            image_url = self._extract_field(raw_data, ['image', 'imageURL', 'main_image'])
            
            # ID
            product_id = self._extract_field(raw_data, ['asin', 'id', 'product_id'])
            if not product_id:
                product_id = f"amzn_{hashlib.md5(title.encode()).hexdigest()[:10]}"
            
            # Generar embeddings
            title_embedding = self.embedding_model.encode(title, normalize_embeddings=True)
            content = f"{title} {description} {' '.join(features)}"
            content_embedding = self.embedding_model.encode(content[:1000], normalize_embeddings=True)
            
            return AmazonProduct(
                id=product_id,
                title=title[:200],
                description=description[:500],
                price=price,
                category=category,
                brand=brand,
                rating=rating,
                rating_count=rating_count,
                features=features[:10],  # Limitar a 10 features
                image_url=image_url,
                title_embedding=title_embedding,
                content_embedding=content_embedding
            )
            
        except Exception as e:
            logger.debug(f"Error parseando producto: {e}")
            return None
    
    def _extract_field(self, data: Dict, keys: List[str]) -> Optional[str]:
        """Extrae campo usando múltiples posibles nombres de clave"""
        for key in keys:
            if key in data and data[key]:
                value = data[key]
                if isinstance(value, str):
                    return value.strip()
                elif isinstance(value, list):
                    return ' '.join(str(x) for x in value[:3])
                else:
                    return str(value)
        return None
    
    def _extract_price(self, data: Dict) -> Optional[float]:
        """Extrae precio"""
        price_keys = ['price', 'Price', 'list_price', 'actual_price']
        
        for key in price_keys:
            if key in data:
                try:
                    price_str = str(data[key])
                    # Limpiar: quitar símbolos de moneda, comas, etc.
                    import re
                    match = re.search(r'(\d+\.?\d*)', price_str.replace(',', ''))
                    if match:
                        return float(match.group(1))
                except:
                    continue
        
        return None
    
    def _extract_category(self, data: Dict) -> str:
        """Extrae categoría"""
        category_keys = ['main_category', 'category', 'categories', 'primary_category']
        
        for key in category_keys:
            if key in data:
                cat = data[key]
                if isinstance(cat, str):
                    # Tomar solo la categoría principal
                    categories = cat.split('|')
                    return categories[0].strip() if categories else "General"
                elif isinstance(cat, list) and cat:
                    return str(cat[0])
        
        return "General"
    
    def _extract_rating(self, data: Dict) -> Optional[float]:
        """Extrae rating"""
        rating_keys = ['rating', 'average_rating', 'overall_rating', 'stars']
        
        for key in rating_keys:
            if key in data:
                try:
                    rating = float(data[key])
                    # Normalizar a 0-5
                    if rating > 5:
                        rating = rating / 20  # Si está en escala 100
                    return max(0.0, min(5.0, rating))
                except:
                    continue
        
        return None
    
    def _extract_rating_count(self, data: Dict) -> Optional[int]:
        """Extrae conteo de ratings"""
        count_keys = ['rating_count', 'review_count', 'num_ratings', 'total_reviews']
        
        for key in count_keys:
            if key in data:
                try:
                    count = int(data[key])
                    return max(0, count)
                except:
                    continue
        
        return None
    
    def _extract_features(self, data: Dict) -> List[str]:
        """Extrae features"""
        features_keys = ['features', 'feature', 'bullet_points', 'key_features']
        
        for key in features_keys:
            if key in data:
                features = data[key]
                if isinstance(features, list):
                    return [str(f) for f in features[:10]]  # Limitar a 10
                elif isinstance(features, str):
                    return [f.strip() for f in features.split(',')[:10]]
        
        return []