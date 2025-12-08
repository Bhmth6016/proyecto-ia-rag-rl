# src/core/data/product_service.py
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from src.core.data.product import Product
from src.core.config import settings

class ProductService:
    """Servicio para acceso a productos con cache"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.PROC_DIR
        self.products_cache: Dict[str, Product] = {}
        self._load_products()
    
    def _load_products(self) -> None:
        """Carga todos los productos a cache"""
        try:
            products_file = self.data_dir / "products.json"
            if products_file.exists():
                with open(products_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in data:
                    product = Product(**item)
                    self.products_cache[product.id] = product
                
                print(f"📦 {len(self.products_cache)} productos cargados en cache")
        except Exception as e:
            print(f"❌ Error cargando productos: {e}")
    
    def get_product(self, product_id: str) -> Optional[Product]:
        """Obtiene un producto por ID"""
        return self.products_cache.get(product_id)
    
    def get_popular_products(self, limit: int = 100) -> List[Product]:
        """Obtiene productos populares ordenados por rating"""
        products = list(self.products_cache.values())
        
        # Ordenar por rating_count y rating
        sorted_products = sorted(
            products,
            key=lambda p: (
                getattr(p, 'rating_count', 0),
                getattr(p, 'average_rating', 0)
            ),
            reverse=True
        )
        
        return sorted_products[:limit]
    
    def search_products(self, query: str, limit: int = 20) -> List[Product]:
        """Búsqueda simple por texto"""
        results = []
        query_lower = query.lower()
        
        for product in self.products_cache.values():
            title = getattr(product, 'title', '').lower()
            description = getattr(product, 'description', '').lower()
            
            if (query_lower in title) or (query_lower in description):
                results.append(product)
                if len(results) >= limit:
                    break
        
        return results