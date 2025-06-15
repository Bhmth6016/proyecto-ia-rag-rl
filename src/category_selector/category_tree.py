import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict
from src.data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sanitize_category_name(name: str) -> str:
    clean = (
        os.path.basename(name)
        .replace("meta_", "")
        .replace("_processed.pkl", "")
        .replace("_", " ")
    )
    clean = re.sub(r"[^\w\s-]", "", clean.strip())
    return clean.title()

def extract_filters_from_products(products: List[Dict[str, Any]]) -> Dict[str, Any]:
    filters = {
        "price_range": [],
        "average_rating": [],
        "category_tags": [],
    }
    prices: List[float] = []
    ratings: Set[int] = set()
    categories: Set[str] = set()

    for product in products:
        price = product.get("price")
        if isinstance(price, (int, float)):
            prices.append(float(price))

        rating = product.get("average_rating")
        if isinstance(rating, (int, float)):
            ratings.add(round(float(rating)))

        product_categories = product.get("categories", [])
        if isinstance(product_categories, list):
            for cat in product_categories:
                if isinstance(cat, (str, int, float)):
                    categories.add(str(cat).strip())

    if prices:
        filters["price_range"] = [min(prices), max(prices)]

    filters["average_rating"] = sorted(ratings)
    filters["category_tags"] = sorted(c for c in categories if c)

    return filters

def get_safe_filename(raw_category: str) -> str:
    return f"meta_{raw_category.lower().replace(' ', '_')}_processed.pkl"

def load_category_tree() -> Dict[str, Dict[str, Any]]:
    loader = DataLoader()
    category_tree: Dict[str, Dict[str, Any]] = {}

    try:
        categorized_data = loader.load_by_main_category(use_cache=True)

        for raw_category, products in categorized_data.items():
            if not products:
                continue

            safe_name = sanitize_category_name(raw_category)
            safe_filename = get_safe_filename(raw_category)
            file_path = loader.processed_dir / safe_filename

            if not file_path.exists():
                continue

            category_tree[safe_name] = {
                "file_path": str(file_path),
                "filters": extract_filters_from_products(products)
            }

    except Exception as e:
        logger.error(f"Error cargando categorías: {e}")

    return category_tree

def generar_categorias_y_filtros(productos: List[Dict[str, Any]]):
    output_path = Path("data") / "processed" / "category_filters.json"
    if output_path.exists():
        logger.info("Archivo de filtros ya existe")
        return

    categorias = set()
    filtros = defaultdict(set)

    for item in productos:
        if item is None:  # Añadir verificación para items nulos
            continue
            
        categoria = item.get("category", "Otros")
        categorias.add(categoria)
        
        details = item.get("details")
        if not isinstance(details, dict):  # Verificar que details es un diccionario
            continue
            
        for k, v in details.items():
            if isinstance(v, str) and len(v) < 30:
                filtros[k].add(v)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "categorias": list(categorias),
            "filtros": {k: list(v) for k, v in filtros.items()}
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Categorías y filtros guardados en {output_path}")