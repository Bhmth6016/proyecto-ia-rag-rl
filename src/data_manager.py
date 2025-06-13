import logging
from typing import List, Dict, Optional, Union
from pathlib import Path
from .data_loader import DataLoader

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.loader = DataLoader(raw_dir, processed_dir)
        self._data_cache = None
        self._category_index = {}
        self._brand_index = {}

    def load_and_cache_data(self, force_reload: bool = False) -> None:
        if self._data_cache is None or force_reload:
            self._data_cache = self.loader.load_data()
            self._build_indices()
            logger.info("Datos cargados e índices construidos")

    def _build_indices(self) -> None:
        self._category_index = {}
        self._brand_index = {}
        
        for product in self._data_cache:
            main_cat = product['main_category']
            if main_cat not in self._category_index:
                self._category_index[main_cat] = []
            self._category_index[main_cat].append(product)
            
            brand = product['brand']
            if brand:
                if brand not in self._brand_index:
                    self._brand_index[brand] = []
                self._brand_index[brand].append(product)

    def get_all_products(self) -> List[Dict]:
        if self._data_cache is None:
            self.load_and_cache_data()
        return self._data_cache.copy()

    def get_products_by_category(self, category: str) -> List[Dict]:
        if self._data_cache is None:
            self.load_and_cache_data()
        return self._category_index.get(category, []).copy()

    def get_products_by_brand(self, brand: str) -> List[Dict]:
        if self._data_cache is None:
            self.load_and_cache_data()
        return self._brand_index.get(brand, []).copy()

    def search_products(
        self,
        query: str,
        min_rating: float = 0.0,
        max_price: Optional[float] = None
    ) -> List[Dict]:
        if self._data_cache is None:
            self.load_and_cache_data()
            
        query = query.lower()
        results = []
        
        for product in self._data_cache:
            matches_query = (
                query in product['title'].lower() or 
                query in product['description'].lower()
            )
            matches_rating = product['review_stats']['rating'] >= min_rating
            matches_price = (
                max_price is None or 
                product['price'] <= max_price
            )
            
            if matches_query and matches_rating and matches_price:
                results.append(product)
        
        return sorted(
            results,
            key=lambda x: (-x['review_stats']['rating'], x['price'])
        )

    def get_product_by_asin(self, asin: str) -> Optional[Dict]:
        if self._data_cache is None:
            self.load_and_cache_data()
            
        for product in self._data_cache:
            if product['asin'] == asin:
                return product
        return None

    def get_categories(self) -> List[str]:
        if self._data_cache is None:
            self.load_and_cache_data()
        return list(self._category_index.keys())

    def get_brands(self) -> List[str]:
        if self._data_cache is None:
            self.load_and_cache_data()
        return list(self._brand_index.keys())