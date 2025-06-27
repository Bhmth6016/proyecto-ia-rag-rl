import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Any, Optional
from collections import defaultdict
from src.data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sanitize_category_name(name: Optional[str]) -> str:
    """Normaliza nombres de categoría para uso seguro en UI"""
    if name is None:
        return "Otras Categorías"
    
    try:
        # Extraer solo el nombre base si es una ruta
        base_name = os.path.basename(str(name))
        # Limpieza de caracteres especiales
        clean = (
            base_name.replace("meta_", "")
                    .replace("_processed.pkl", "")
                    .replace("_", " ")
                    .strip()
        )
        # Eliminar caracteres no permitidos y capitalizar
        clean = re.sub(r"[^\w\s-]", "", clean)
        return clean.title() if clean else "Otras Categorías"
    except Exception:
        return "Otras Categorías"

def extract_filters_from_products(products: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extrae y valida filtros de una lista de productos"""
    filters: Dict[str, Any] = {
        "price_range": [0, 0],
        "average_rating": [],
        "category_tags": [],
        "details": defaultdict(set)
    }
    
    if not products or not isinstance(products, list):
        return filters

    prices: List[float] = []
    ratings: Set[int] = set()
    categories: Set[str] = set()

    for product in products:
        if not isinstance(product, dict):
            continue

        # Procesamiento de precios
        price = product.get("price")
        if isinstance(price, (int, float)) and price >= 0:
            prices.append(float(price))

        # Procesamiento de ratings
        rating = product.get("average_rating")
        if isinstance(rating, (int, float)) and 0 <= rating <= 5:
            ratings.add(round(float(rating)))

        # Procesamiento de categorías
        product_categories = product.get("categories", [])
        if isinstance(product_categories, list):
            for cat in product_categories:
                if cat is not None:
                    try:
                        cleaned = str(cat).strip()
                        if cleaned:
                            categories.add(cleaned)
                    except (AttributeError, TypeError):
                        continue

        # Procesamiento de detalles
        details = product.get("details", {})
        if isinstance(details, dict):
            for k, v in details.items():
                if k is not None and v is not None:
                    try:
                        key = str(k).strip()
                        value = str(v).strip()
                        if key and value:
                            filters["details"][key].add(value)
                    except (AttributeError, TypeError):
                        continue

    # Resultados finales con validación
    filters["price_range"] = [
        min(prices) if prices else 0,
        max(prices) if prices else 0
    ]
    filters["average_rating"] = sorted(r for r in ratings if 0 <= r <= 5)
    filters["category_tags"] = sorted(c for c in categories if c)
    filters["details"] = {
        k: sorted(v) 
        for k, v in filters["details"].items() 
        if k and v
    }

    return filters

def get_safe_filename(raw_category: Optional[str]) -> str:
    """Genera nombres de archivo seguros a partir de categorías"""
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

def load_category_tree() -> Dict[str, Dict[str, Any]]:
    """Carga y valida el árbol completo de categorías"""
    loader = DataLoader()
    category_tree: Dict[str, Dict[str, Any]] = {}

    try:
        categorized_data = loader.load_by_main_category(use_cache=True)
        
        if not isinstance(categorized_data, dict):
            logger.error("Datos categorizados no son un diccionario")
            return category_tree

        for raw_category, products in categorized_data.items():
            if not products or not isinstance(products, list):
                continue

            safe_name = sanitize_category_name(raw_category)
            safe_filename = get_safe_filename(raw_category)
            file_path = loader.processed_dir / safe_filename

            if not file_path.exists():
                logger.debug(f"Archivo no encontrado: {file_path}")
                continue

            category_tree[safe_name] = {
                "file_path": str(file_path),
                "product_count": len(products),
                "filters": extract_filters_from_products(products)
            }

    except Exception as e:
        logger.error(f"Error crítico cargando categorías: {str(e)}", exc_info=True)

    return category_tree

def generar_categorias_y_filtros(productos: List[Dict[str, Any]]):
    """Genera el archivo JSON de categorías y filtros con validación robusta"""
    if not isinstance(productos, list):
        logger.error("Los productos no son una lista válida")
        return

    output_dir = Path("data") / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "category_filters.json"

    try:
        categorias: Set[str] = set()
        filtros: Dict[str, Set[str]] = defaultdict(set)

        for item in productos:
            if not isinstance(item, dict):
                continue

            # Procesamiento de categoría principal
            categoria = item.get("main_category")
            if categoria is not None:
                try:
                    cleaned = str(categoria).strip()
                    if cleaned:
                        categorias.add(cleaned)
                except (AttributeError, TypeError):
                    pass

            # Procesamiento de detalles
            details = item.get("details", {})
            if isinstance(details, dict):
                for k, v in details.items():
                    if k is not None and v is not None:
                        try:
                            key = str(k).strip()
                            value = str(v).strip()
                            if key and value and len(value) < 100:  # Limitar tamaño
                                filtros[key].add(value)
                        except (AttributeError, TypeError):
                            continue

        # Validación final antes de guardar
        if not categorias and not filtros:
            logger.warning("No se encontraron categorías o filtros válidos")
            return

        data = {
            "categorias": sorted(c for c in categorias if c),
            "filtros": {
                k: sorted(v for v in values if v)
                for k, values in filtros.items()
                if k and values
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Categorías y filtros guardados en {output_path}")

    except Exception as e:
        logger.error(f"Error generando categorías: {str(e)}", exc_info=True)