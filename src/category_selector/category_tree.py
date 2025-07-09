import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from src.data_loader import DataLoader

logger = logging.getLogger(__name__)

def sanitize_category_name(name: Optional[str]) -> str:
    if name is None:
        return "Otras Categorías"

    try:
        base_name = os.path.basename(str(name))
        clean = (
            base_name.replace("meta_", "")
                    .replace("_processed.pkl", "")
                    .replace("_", " ")
                    .strip()
        )
        clean = re.sub(r"[^\w\s-]", "", clean)
        return clean.title() if clean else "Otras Categorías"
    except Exception:
        return "Otras Categorías"

def extract_filters_from_products(products: List[Dict[str, Any]]) -> Dict[str, Any]:
    filters = {
        "price_range": [float('inf'), 0],
        "average_rating": set(),
        "details": defaultdict(set)
    }

    for product in products:
        if not isinstance(product, dict):
            continue

        # Procesar precio
        if isinstance(product.get("price"), (int, float)):
            filters["price_range"][0] = min(filters["price_range"][0], product["price"])
            filters["price_range"][1] = max(filters["price_range"][1], product["price"])

        # Procesar rating
        if isinstance(product.get("average_rating"), (int, float)):
            rounded = round(product["average_rating"])
            if 0 <= rounded <= 5:
                filters["average_rating"].add(rounded)

        # Procesar detalles
        if isinstance(product.get("details"), dict):
            for k, v in product["details"].items():
                if k and v is not None:
                    try:
                        filters["details"][str(k).strip()].add(str(v).strip())
                    except (AttributeError, TypeError):
                        continue

    return {
        "price_range": [
            filters["price_range"][0] if filters["price_range"][0] != float('inf') else 0,
            max(filters["price_range"][1], 0)
        ],
        "average_rating": sorted(filters["average_rating"]),
        "details": {
            k: sorted(v)
            for k, v in filters["details"].items()
            if k and v
        }
    }

def get_safe_filename(raw_category: Optional[str]) -> str:
    default = "meta_otros_processed.pkl"

    if raw_category is None:
        return default

    try:
        clean = (
            str(raw_category)
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .strip("_")
        )
        return f"meta_{clean}_processed.pkl" if clean else default
    except Exception:
        return default



def generar_categorias_y_filtros(productos: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(productos, list):
        logger.error("Los productos no son una lista válida")
        return None

    output_dir = Path("data") / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "category_filters.json"

    try:
        # Estructura compatible con ProductInterface
        data = {
            "global": {
                "price_range": [0, 0],
                "ratings": [],
                "details": {},
                "categories": []
            },
            "by_category": {}
        }

        # Procesar productos globales
        global_filters = extract_filters_from_products(productos)
        data["global"].update({
            "price_range": global_filters["price_range"],
            "ratings": global_filters["average_rating"],
            "details": global_filters["details"]
        })

        # Procesar por categoría
        categorized = defaultdict(list)
        for product in productos:
            if category := product.get("main_category"):
                categorized[category].append(product)

        for category, products in categorized.items():
            cat_filters = extract_filters_from_products(products)
            data["by_category"][category] = {
                "price_range": cat_filters["price_range"],
                "ratings": cat_filters["average_rating"],
                "details": cat_filters["details"],
                "products": products
            }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Categorías y filtros guardados en {output_path}")
        return data

    except Exception as e:
        logger.error(f"Error generando categorías: {str(e)}", exc_info=True)
        return None