# category_tree.py
import os
import pickle
from collections import defaultdict

DATA_DIR = r"C:\Users\evill\OneDrive\Documentos\Github\proyecto-ia-rag-rl\data\processed"

def readable_category_name(filename: str) -> str:
    """Convierte el nombre de archivo a una categoría legible"""
    base = os.path.basename(filename)
    name = base.replace("meta_", "").replace("_processed.pkl", "").replace("_", " ")
    return name.strip().title()

def extract_filters_from_products(products: list) -> dict:
    filters = {
        "price_range": [],
        "average_rating": [],
        "category_tags": [],
    }

    prices = []
    ratings = set()
    categories = set()

    for product in products:
        # Precio (puede ser None)
        price = product.get("price")
        if isinstance(price, (int, float)):
            prices.append(price)

        # Rating (puede ser None)
        rating = product.get("average_rating")
        if isinstance(rating, (int, float)):
            ratings.add(round(rating))

        # Categorías (ahora puede ser None o lista vacía)
        product_categories = product.get("categories", []) or []  # Convierte None en lista vacía
        if isinstance(product_categories, list):
            categories.update(
                cat for cat in product_categories 
                if cat and isinstance(cat, str)
            )

    if prices:
        filters["price_range"] = [min(prices), max(prices)]
    
    filters["average_rating"] = sorted(ratings)
    filters["category_tags"] = sorted(categories)

    return filters

def load_category_tree():
    """Carga los archivos .pkl y construye árbol de categorías y filtros"""
    category_tree = {}

    for filename in os.listdir(DATA_DIR):
        if filename.endswith("_processed.pkl") and filename.startswith("meta_"):
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, "rb") as f:
                products = pickle.load(f)

            readable_name = readable_category_name(filename)
            filters = extract_filters_from_products(products)

            category_tree[readable_name] = {
                "file_path": filepath,
                "filters": filters,
            }

    return category_tree
