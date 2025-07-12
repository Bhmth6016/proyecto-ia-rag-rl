# src/core/category_search/category_tree.py
"""
Unified filtering & category-tree module for the Amazon product catalog.
Combines:
  - Hierarchical category tree (CategoryTree)
  - Global and per-category filter extraction
  - Runtime ProductFilter with single-dispatch apply()
  - JSON export utilities
  - Legacy helpers (sanitize, safe-filename, etc.)
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from functools import singledispatchmethod

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# Runtime product filter
# ------------------------------------------------------------------
class ProductFilter:
    def __init__(self):
        self.price_range: Optional[FilterRange] = None
        self.rating_range: Optional[FilterRange] = None
        self.brands: Set[str] = set()
        self.features: Dict[str, Set[str]] = defaultdict(set)

    # Single-dispatch apply -------------------------------------------------
    @singledispatchmethod
    def apply(self, product: Dict[str, Any]) -> bool:
        if not isinstance(product, dict):
            return False

        if self.price_range and not self.price_range.contains(product.get("price")):
            return False

        if self.rating_range and not self.rating_range.contains(product.get("average_rating")):
            return False

        if self.brands:
            product_brand = product.get("details", {}).get("Brand")
            if not product_brand or str(product_brand) not in self.brands:
                return False

        for feature, allowed in self.features.items():
            product_val = product.get("details", {}).get(feature)
            if not product_val or str(product_val) not in allowed:
                return False
        return True

    @apply.register(list)
    def _(self, products: list) -> list:
        return [p for p in products if isinstance(p, dict) and self.apply(p)]

    # Builder helpers -------------------------------------------------------
    def add_price_filter(self, min_price: float, max_price: float) -> None:
        self.price_range = FilterRange(min_price, max_price)

    def add_rating_filter(self, min_rating: float, max_rating: float = 5.0) -> None:
        self.rating_range = FilterRange(min_rating, max_rating)

    def add_brand_filter(self, brands: Union[str, List[str]]) -> None:
        if isinstance(brands, str):
            brands = [brands]
        self.brands.update(str(b).strip() for b in brands)

    def add_feature_filter(self, feature: str, values: Union[str, List[str]]) -> None:
        if isinstance(values, str):
            values = [values]
        self.features[feature].update(str(v).strip() for v in values)

    def clear_filters(self) -> None:
        self.price_range = None
        self.rating_range = None
        self.brands.clear()
        self.features.clear()

    # Serialization ---------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "price_range": [self.price_range.min, self.price_range.max] if self.price_range else None,
            "rating_range": [self.rating_range.min, self.rating_range.max] if self.rating_range else None,
            "brands": sorted(self.brands),
            "features": {k: sorted(v) for k, v in self.features.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductFilter":
        pf = cls()
        if data.get("price_range"):
            pf.add_price_filter(*data["price_range"])
        if data.get("rating_range"):
            pf.add_rating_filter(*data["rating_range"])
        if data.get("brands"):
            pf.add_brand_filter(data["brands"])
        if data.get("features"):
            for feat, vals in data["features"].items():
                pf.add_feature_filter(feat, vals)
        return pf


# ------------------------------------------------------------------
# Global / per-category filter extractor
# ------------------------------------------------------------------
def extract_global_filters(products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract available filter ranges from a product list."""
    if not products:
        return {}

    filters = {
        "price_range": [float("inf"), 0],
        "ratings": set(),
        "brands": set(),
        "features": defaultdict(set),
    }

    for p in products:
        # Price
        if isinstance(p.get("price"), (int, float)):
            filters["price_range"][0] = min(filters["price_range"][0], p["price"])
            filters["price_range"][1] = max(filters["price_range"][1], p["price"])

        # Rating
        if isinstance(p.get("average_rating"), (int, float)):
            rounded = round(p["average_rating"])
            if 0 <= rounded <= 5:
                filters["ratings"].add(rounded)

        # Brand
        brand = p.get("details", {}).get("Brand")
        if isinstance(brand, str) and brand.strip():
            filters["brands"].add(brand.strip())

        # Generic features
        details = p.get("details")
        if isinstance(details, dict):
            for k, v in details.items():
                if k != "Brand" and v is not None:
                    filters["features"][k].add(str(v).strip())

    return {
        "price_range": [
            filters["price_range"][0] if filters["price_range"][0] != float("inf") else 0,
            max(filters["price_range"][1], 0),
        ],
        "ratings": sorted(filters["ratings"]),
        "brands": sorted(filters["brands"]),
        "features": {k: sorted(v) for k, v in filters["features"].items()},
    }


# ------------------------------------------------------------------
# Category tree (hierarchical)
# ------------------------------------------------------------------
@dataclass
class CategoryNode:
    name: str
    products: List[Dict]
    filters: Dict[str, Any]
    parent: Optional["CategoryNode"] = None
    children: List["CategoryNode"] = None

    def __post_init__(self):
        self.children = self.children or []


class CategoryTree:
    def __init__(self, products: List[Dict], min_products_per_category: int = 10):
        self.products = products
        self.min_products = min_products_per_category
        self.root = CategoryNode(name="root", products=[], filters={})
        self._category_map: Dict[str, CategoryNode] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build_tree(self) -> CategoryNode:
        categories = self._group_by_main_category()
        for path, prods in categories.items():
            self._add_category_path(path, prods)
        self._generate_filters()
        return self.root

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def save_to_json(self, file_path: Union[str, Path]) -> None:
        def _node_to_dict(node: CategoryNode) -> Dict:
            return {
                "name": node.name,
                "product_count": len(node.products),
                "filters": node.filters,
                "children": [_node_to_dict(child) for child in node.children],
            }

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(_node_to_dict(self.root), f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _group_by_main_category(self) -> Dict[str, List[Dict]]:
        grouped = defaultdict(list)
        for prod in self.products:
            if not self._is_valid_product(prod):
                continue
            cat = self._normalize_category(prod.get("main_category"))
            if cat:
                grouped[cat].append(prod)
        return {k: v for k, v in grouped.items() if len(v) >= self.min_products}

    def _add_category_path(self, category_path: str, products: List[Dict]) -> None:
        parts = self._split_category_path(category_path)
        current = self.root
        for part in parts:
            child = next((c for c in current.children if c.name == part), None)
            if child is None:
                child = CategoryNode(name=part, products=[], filters={}, parent=current)
                current.children.append(child)
                self._category_map[part] = child
            current = child
        current.products = products

    def _generate_filters(self) -> None:
        def _dfs(node: CategoryNode):
            if node.products:
                node.filters = extract_global_filters(node.products)
            for child in node.children:
                _dfs(child)

        _dfs(self.root)

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _is_valid_product(product: Dict) -> bool:
        return (
            isinstance(product, dict)
            and isinstance(product.get("title"), str)
            and product["title"].strip()
            and isinstance(product.get("main_category"), str)
        )

    @staticmethod
    def _normalize_category(category: Optional[str]) -> Optional[str]:
        if not category:
            return None
        cleaned = re.sub(r"^meta_", "", str(category))
        cleaned = re.sub(r"_processed\.pkl$", "", cleaned)
        cleaned = re.sub(r"[^\w\s-]", "", cleaned).replace("_", " ").strip()
        return cleaned.title() if cleaned else None

    @staticmethod
    def _split_category_path(category_path: str) -> List[str]:
        return [part.strip() for part in category_path.split(">") if part.strip()]


# ------------------------------------------------------------------
# Convenience wrappers (drop-in for legacy code)
# ------------------------------------------------------------------
def sanitize_category_name(name: Optional[str]) -> str:
    return CategoryTree._normalize_category(name) or "Other Categories"


def get_safe_filename(raw_category: Optional[str]) -> str:
    if not raw_category:
        return "meta_other_processed.pkl"
    safe = re.sub(r"\W+", "_", str(raw_category).lower()).strip("_")
    return f"meta_{safe}_processed.pkl" if safe else "meta_other_processed.pkl"


def apply_filters(
    products: List[Dict[str, Any]],
    price_range: Optional[Tuple[float, float]] = None,
    ratings: Optional[List[int]] = None,
    brands: Optional[List[str]] = None,
    features: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """One-liner helper."""
    pf = ProductFilter()
    if price_range:
        pf.add_price_filter(*price_range)
    if ratings:
        pf.add_rating_filter(min(ratings), max(ratings))
    if brands:
        pf.add_brand_filter(brands)
    if features:
        for feat, vals in features.items():
            pf.add_feature_filter(feat, vals)
    return pf.apply(products)


def generate_categories_and_filters(
    products: List[Dict[str, Any]],
    output_file: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Legacy entry point that returns the flat JSON structure expected by the UI."""
    try:
        tree = CategoryTree(products)
        tree.build_tree()

        result = {
            "global": extract_global_filters(products),
            "by_category": {},
        }

        def _collect(node: CategoryNode):
            if node.products and node.name != "root":
                result["by_category"][node.name] = {
                    **node.filters,
                    "products": node.products,
                }
            for child in node.children:
                _collect(child)

        _collect(tree.root)

        if output_file is None:
            output_file = Path("data") / "processed" / "category_filters.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        tree.save_to_json(output_file.with_suffix(".tree.json"))  # hierarchical tree
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return result

    except Exception as e:
        logger.error(f"Error generating categories/filters: {e}", exc_info=True)
        return None