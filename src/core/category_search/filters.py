# src/core/category_search/filters.py
from dataclasses import dataclass
from collections import defaultdict
import logging
from functools import singledispatchmethod
from typing import Dict, List, Optional, Set, Union, Any, Tuple
from collections.abc import Sequence 

logger = logging.getLogger(__name__)

@dataclass
class FilterRange:
    min: float
    max: float
    
    def contains(self, value: Optional[Union[float, int]]) -> bool:
        if value is None:
            return False
        return self.min <= value <= self.max

@dataclass
class FilterOption:
    name: str
    values: Set[str]
    
    def matches(self, product_value: Optional[str]) -> bool:
        if not product_value:
            return False
        return str(product_value) in self.values

class ProductFilter:
    def __init__(self):
        self.price_range: Optional[FilterRange] = None
        self.rating_range: Optional[FilterRange] = None
        self.brands: Set[str] = set()
        self.features: Dict[str, Set[str]] = defaultdict(set)
    
    @singledispatchmethod
    def apply(self, product: Dict[str, Any]) -> bool:
        """Base implementation for single product filtering"""
        if not isinstance(product, dict):
            return False
            
        # Your existing filter logic here
        if self.price_range and not self.price_range.contains(product.get('price')):
            return False
            
        # Filtro por rating
        if self.rating_range and not self.rating_range.contains(product.get('average_rating')):
            return False
            
        # Filtro por marca
        if self.brands:
            product_brand = product.get('details', {}).get('Brand')
            if not product_brand or str(product_brand) not in self.brands:
                return False
                
        # Filtros por características
        for feature, allowed_values in self.features.items():
            product_value = product.get('details', {}).get(feature)
            if not product_value or str(product_value) not in allowed_values:
                return False
                
        return True

    @apply.register
    def _(self, products: list) -> list:
        """Handle both List[Dict] and plain list"""
        return [p for p in products if isinstance(p, dict) and self.apply(p)]

    def add_price_filter(self, min_price: float, max_price: float):
        """Añade filtro por rango de precios"""
        self.price_range = FilterRange(min_price, max_price)
        logger.info(f"Añadido filtro de precio: {min_price}-{max_price}")

    def add_rating_filter(self, min_rating: float, max_rating: float = 5.0):
        """Añade filtro por rango de ratings"""
        self.rating_range = FilterRange(min_rating, max_rating)
        logger.info(f"Añadido filtro de rating: {min_rating}-{max_rating}")

    def add_brand_filter(self, brands: Union[str, List[str]]):
        """Añade filtro por marcas"""
        if isinstance(brands, str):
            brands = [brands]
        self.brands.update(str(b).strip() for b in brands)
        logger.info(f"Añadido filtro de marcas: {', '.join(brands)}")

    def add_feature_filter(self, feature: str, values: Union[str, List[str]]):
        """Añade filtro por características específicas"""
        if isinstance(values, str):
            values = [values]
        self.features[feature].update(str(v).strip() for v in values)
        logger.info(f"Añadido filtro para {feature}: {', '.join(values)}")

    def clear_filters(self):
        """Resetea todos los filtros"""
        self.price_range = None
        self.rating_range = None
        self.brands.clear()
        self.features.clear()
        logger.info("Todos los filtros han sido reseteados")

    def to_dict(self) -> Dict[str, Any]:
        """Serializa los filtros a diccionario"""
        return {
            "price_range": [self.price_range.min, self.price_range.max] if self.price_range else None,
            "rating_range": [self.rating_range.min, self.rating_range.max] if self.rating_range else None,
            "brands": sorted(self.brands),
            "features": {k: sorted(v) for k, v in self.features.items()}
        }

    @classmethod
    def from_dict(cls, filter_dict: Dict[str, Any]) -> 'ProductFilter':
        """Crea un ProductFilter desde un diccionario"""
        pf = cls()
        
        if filter_dict.get('price_range'):
            pf.add_price_filter(*filter_dict['price_range'])
            
        if filter_dict.get('rating_range'):
            pf.add_rating_filter(*filter_dict['rating_range'])
            
        if filter_dict.get('brands'):
            pf.add_brand_filter(filter_dict['brands'])
            
        if filter_dict.get('features'):
            for feature, values in filter_dict['features'].items():
                pf.add_feature_filter(feature, values)
                
        return pf

def extract_global_filters(products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extrae los rangos de filtros disponibles de todos los productos"""
    if not products:
        return {}
    
    filters = {
        "price_range": [float('inf'), 0],
        "ratings": set(),
        "brands": set(),
        "features": defaultdict(set)
    }
    
    for product in products:
        # Precio
        if isinstance(product.get('price'), (int, float)):
            filters["price_range"][0] = min(filters["price_range"][0], product['price'])
            filters["price_range"][1] = max(filters["price_range"][1], product['price'])
        
        # Rating
        if isinstance(product.get('average_rating'), (int, float)):
            rounded = round(product['average_rating'])
            if 0 <= rounded <= 5:
                filters["ratings"].add(rounded)
        
        # Marca
        if isinstance(product.get('details', {}).get('Brand'), str):
            filters["brands"].add(product['details']['Brand'].strip())
        
        # Características
        if isinstance(product.get('details'), dict):
            for k, v in product['details'].items():
                if k != 'Brand' and v is not None:
                    filters["features"][k].add(str(v).strip())
    
    return {
        "price_range": [
            filters["price_range"][0] if filters["price_range"][0] != float('inf') else 0,
            max(filters["price_range"][1], 0)
        ],
        "ratings": sorted(filters["ratings"]),
        "brands": sorted(filters["brands"]),
        "features": {k: sorted(v) for k, v in filters["features"].items()}
    }

def apply_filters(
    products: List[Dict[str, Any]],
    price_range: Optional[Tuple[float, float]] = None,
    ratings: Optional[List[int]] = None,
    brands: Optional[List[str]] = None,
    features: Optional[Dict[str, List[str]]] = None
) -> List[Dict[str, Any]]:
    """
    Función de conveniencia para aplicar filtros en un solo paso
    
    Args:
        products: Lista de productos a filtrar
        price_range: Tupla (min, max) para precio
        ratings: Lista de ratings a incluir
        brands: Lista de marcas a incluir
        features: Diccionario {feature: [valores]}
    
    Returns:
        Lista filtrada de productos
    """
    pf = ProductFilter()
    
    if price_range:
        pf.add_price_filter(*price_range)
    if ratings:
        pf.add_rating_filter(min(ratings), max(ratings))
    if brands:
        pf.add_brand_filter(brands)
    if features:
        for feature, values in features.items():
            pf.add_feature_filter(feature, values)
    
    return pf.apply(products)