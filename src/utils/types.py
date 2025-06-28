# types.py (versión simplificada)
from typing import TypedDict, List, Dict

class Filters(TypedDict):
    price_range: List[float]
    ratings: List[int]
    details: Dict[str, List[str]]
    categories: List[str]

class CategoryFilters(TypedDict):
    global_filters: Filters
    by_category: Dict[str, Filters]