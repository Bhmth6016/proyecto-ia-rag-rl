# src/core/data/loader.py
"""
DataLoader con rutas dinámicas y política de caché desde settings.py
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from threading import Lock

from tqdm import tqdm

from src.core.data.product import Product
from src.core.config import settings
from src.core.utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Mapeo de palabras clave
# ------------------------------------------------------------------
_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "beauty": [
        "makeup", "skincare", "cosmetics", "perfume", "serum", "beauty", "lipstick", "cream", "mascara", "nail polish"
    ],
    "gift_card": [
        "gift card", "voucher", "tarjeta regalo", "balance", "store credit", "code", "redeem"
    ],
    "magazine": [
        "magazine", "revista", "subscription", "editorial", "print issue", "digital magazine"
    ],
    "movie": [
        "movie", "film", "blu-ray", "dvd", "streaming", "tv series", "netflix", "disney", "show"
    ],
    "instrument": [
        "guitar", "piano", "drum", "instrument", "violin", "keyboard", "microphone", "amplifier"
    ],
    "office": [
        "printer", "stationery", "office", "desk", "chair", "pen", "notebook", "paper", "laminator", "folder"
    ],
    "garden": [
        "patio", "garden", "grill", "barbecue", "mower", "outdoor", "plant", "hose", "fertilizer", "greenhouse"
    ],
    "software": [
        "software", "app", "application", "program", "license", "subscription", "antivirus", "editor", "IDE"
    ],
    "sports": [
        "sport", "fitness", "bike", "ball", "exercise", "yoga", "mat", "dumbbell", "treadmill", "skateboard"
    ],
    "subscription": [
        "subscription", "monthly box", "plan", "auto-renew", "delivery", "kit", "bundle"
    ],
    "tools": [
        "tool", "drill", "screwdriver", "hammer", "wrench", "toolkit", "saw", "multimeter", "power tool"
    ],
    "toys": [
        "toy", "game", "puzzle", "board game", "lego", "doll", "rc car", "action figure", "plush", "juego"
    ],
    "video_game": [
        "video game", "xbox", "playstation", "nintendo", "steam", "dlc", "gamer", "gamepad", "controller"
    ],
    "unknown": [
        "misc", "other", "generic", "undefined", "product", "item", "unknown"
    ]
}


# ------------------------------------------------------------------
# Etiquetas inferidas automáticamente
# ------------------------------------------------------------------
_TAG_KEYWORDS: Dict[str, List[str]] = {
    "waterproof": ["waterproof", "water resistant", "resistente al agua"],
    "wireless": ["wireless", "bluetooth", "inalámbrico", "wifi"],
    "portable": ["portable", "pocket", "ligero", "lightweight", "compact", "foldable"],
    "gaming": ["gaming", "gamer", "rgb", "for gaming", "fps", "multiplayer"],
    "travel": ["travel", "viaje", "suitcase", "carry-on", "maleta", "portable"],
    "usb-c": ["usb-c", "type-c", "usb type c"],
    "noise-cancelling": ["noise cancelling", "noise reduction", "anc", "cancelación de ruido"],
    "fast-charging": ["fast charging", "quick charge", "carga rápida"],
    "eco-friendly": ["eco", "recycled", "biodegradable", "sostenible", "green"],
    "digital": ["digital", "online", "e-book", "streaming", "virtual", "cloud"],
    "subscription": ["monthly", "membership", "auto-renew", "kit", "bundle"],
    "family-friendly": ["kids", "family", "educational", "safe for children"],
    "limited-edition": ["limited", "exclusive", "collectible", "rare", "edition"],
    "multiplatform": ["android", "ios", "windows", "mac", "cross-platform"],
}


CATEGORY_MAPPING = {
    "Appstore for Android": "Software",
    "Google Play": "Software",
    "Software": "Software",
    "Video_Games": "Software",
    "All_Beauty": "Beauty",
    "Gift_Cards": "Gift Cards",
    "Magazine_Subscriptions": "Magazine Subscriptions",
    "Movies_and_TV": "Movies and TV",
    "Musical_Instruments": "Musical Instruments",
    "Office_Products": "Office Products",
    "Patio_Lawn_and_Garden": "Patio, Lawn and Garden",
    "Sports_and_Outdoors": "Sports and Outdoors",
    "Subscription_Boxes": "Subscription Boxes",
    "Tools_and_Home_Improvement": "Tools and Home Improvement",
    "Toys_and_Games": "Toys and Games",
    "Unknown": "Unknown",
    "Video_Games": "Video Games",
}

class DataLoader:
    """
    Cargador unificado que:
    • lee archivos desde settings.RAW_DIR
    • almacena en caché objetos validados en settings.PROC_DIR
    • obedece settings.CACHE_ENABLED
    """

    def __init__(
        self,
        *,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        cache_enabled: bool = True,  # <-- AGREGA ESTA LÍNEA
        max_workers: int = 4,
        disable_tqdm: bool = False,
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else settings.RAW_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else settings.PROC_DIR
        self.cache_enabled = cache_enabled
        self.max_workers = max_workers
        self.disable_tqdm = disable_tqdm
        self._error_lock = Lock()  # Nuevo Lock

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers para inferencia y etiquetado
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_product_type(title: str, specs: Dict[str, Any]) -> Optional[str]:
        text = (title or "").lower() + " " + " ".join(specs.values()).lower()
        for ptype, kw_list in _CATEGORY_KEYWORDS.items():
            if any(kw in text for kw in kw_list):
                return ptype
        return None

    @staticmethod
    def _extract_tags(title: str, specs: Dict[str, Any]) -> List[str]:
        text = (title or "").lower() + " " + " ".join(specs.values()).lower()
        tags = []
        for tag, kw_list in _TAG_KEYWORDS.items():
            if any(kw in text for kw in kw_list):
                tags.append(tag)
        return tags

    def _normalize_category(self, category: Optional[str], filename: str) -> str:
        """Normaliza categorías usando el nombre del archivo como fallback"""
        if not category:
            # Extraer categoría del nombre del archivo (ej: "Software.jsonl" -> "Software")
            return Path(filename).stem
        return CATEGORY_MAPPING.get(category, category)

    def _enrich_product_data(self, item: Dict, filename: str) -> Dict:
        """Llena campos faltantes y normaliza datos"""
        # Normalizar categoría
        item['main_category'] = self._normalize_category(item.get('main_category'), filename)
        
        # Inferir tipo de producto
        if not item.get('product_type'):
            item['product_type'] = item['main_category'].lower()
        
        # Extraer características de la descripción si no hay features
        description = item.get('description', '').lower()
        if not item.get('details', {}).get('features'):
            features = []
            if 'android' in description: features.append('Android')
            if 'ios' in description: features.append('iOS')
            if 'windows' in description: features.append('Windows')
            if 'mac' in description: features.append('macOS')
            if features:
                item.setdefault('details', {}).setdefault('features', []).extend(features)
        
        return item

    # ------------------------------------------------------------------
    def load_data(self, use_cache: Optional[bool] = None, output_file: Union[str, Path] = None) -> List[Product]:
        """Load all files in raw_dir, process and save to a single JSON."""
        if use_cache is None:
            use_cache = self.cache_enabled

        if output_file is None:
            output_file = self.processed_dir / "products.json"

        if use_cache and output_file.exists():
            logger.info("Cache found. Loading products from cache.")
            with open(output_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)

            valid_products = []
            for item in cached_data:
                try:
                    # Validate and clean the 'details' field
                    if "details" not in item or item["details"] is None:
                        item["details"] = {}
                    elif not isinstance(item["details"], dict):
                        logger.warning("Invalid details found in cached product - resetting to empty")
                        item["details"] = {}

                    # Validate required 'title' field
                    if not item.get("title", "").strip():
                        logger.warning("Product without title found in cache. Skipped.")
                        continue

                    valid_products.append(Product.from_dict(item))
                except Exception as e:
                    logger.warning(f"Error loading product from cache: {e}")
                    continue

            return valid_products

        # If no cache or not to be used, load and process files from raw_dir
        files = list(self.raw_dir.glob("*.json")) + list(self.raw_dir.glob("*.jsonl"))
        if not files:
            logger.warning("No product files found in %s", self.raw_dir)
            return []

        all_products: List[Product] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            future_to_file = {exe.submit(self.load_single_file, f): f for f in files}
            for future in tqdm(future_to_file, desc="Files", total=len(files), disable=self.disable_tqdm):
                try:
                    products = future.result()
                    if products:  # Only extend if we got products
                        all_products.extend(products)
                except Exception as e:
                    logger.warning(f"Error processing file: {future_to_file[future]} - {str(e)}")

        if not all_products:
            logger.error("No valid products loaded from any files!")
            return []

        logger.info("Loaded %d products from %d files", len(all_products), len(files))

        # Save processed products to JSON
        self.save_standardized_json(all_products, output_file)

        return all_products


    def load_single_file(
        self,
        raw_file: Union[str, Path],
    ) -> List[Product]:
        raw_file = Path(raw_file)
        logger.info("Procesando archivo: %s", raw_file.name)
        products = self._process_jsonl(raw_file) if raw_file.suffix.lower() == ".jsonl" else self._process_json_array(raw_file)
        return products

    # ------------------------------------------------------------------
    def _process_jsonl(self, raw_file: Path) -> List[Product]:
        products: List[Product] = []
        error_count = 0

        def _process_line(idx_line):
            nonlocal error_count
            idx, line = idx_line  # Unpack the tuple
            try:
                item = json.loads(line.strip())
                if not isinstance(item, dict):
                    logger.warning(f"Line {idx}: Expected dictionary, got {type(item)}")
                    return None, True

                # Skip only if title is missing or empty
                if not item.get("title", "").strip():
                    logger.warning(f"Line {idx}: Missing title - product skipped")
                    return None, True

                # Handle description
                if "description" in item:
                    if isinstance(item["description"], list):
                        item["description"] = " ".join(str(x) for x in item["description"] if x)
                    elif not item["description"]:
                        item["description"] = "No description available"
                else:
                    item["description"] = "No description available"

                # Handle main_category - use filename if missing
                if not item.get("main_category"):
                    item["main_category"] = Path(raw_file).stem.replace("_", " ").title()
                    
                # Handle price - clean and set default if missing
                price = item.get("price")
                if price is None:
                    item["price"] = "Price not available"
                else:
                    # Clean price string
                    if isinstance(price, str):
                        # Remove currency symbols and non-numeric characters except decimal point
                        cleaned_price = ''.join(c for c in price if c.isdigit() or c == '.')
                        if cleaned_price:
                            try:
                                item["price"] = float(cleaned_price)
                            except ValueError:
                                item["price"] = "Price not available"
                        else:
                            item["price"] = "Price not available"

                # Handle rating
                if item.get("average_rating") is None:
                    item["average_rating"] = "No rating available"

                # Handle details - ensure it exists and has basic structure
                if "details" not in item or not isinstance(item["details"], dict):
                    item["details"] = {"features": [], "specifications": {}}
                else:
                    # Ensure required sub-fields exist
                    item["details"].setdefault("features", [])
                    item["details"].setdefault("specifications", {})

                # Infer product type and tags
                specs = item["details"]["specifications"]
                if not item.get("product_type"):
                    item["product_type"] = self._infer_product_type(item.get("title", ""), specs)
                if not item.get("tags"):
                    item["tags"] = self._extract_tags(item.get("title", ""), specs)

                product = Product.from_dict(item)
                product.clean_image_urls()
                return product, False

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Error processing line {idx}: {e}. Line: {line.strip()}")
                with self._error_lock:
                    error_count += 1
                return None, True

        with raw_file.open("r", encoding="utf-8") as f:
            lines = list(enumerate(f))  # Create list of (index, line) tuples

        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            results = list(tqdm(
                exe.map(_process_line, lines),
                total=len(lines),
                desc=f"Processing {raw_file.name}",
                disable=self.disable_tqdm
            ))

        for product, is_error in results:
            if is_error:
                with self._error_lock:
                    error_count += 1
            elif product is not None:
                products.append(product)

        if error_count:
            logger.warning(
                "%s: %d/%d invalid (%.1f%% success)",
                raw_file.name,
                error_count,
                len(products) + error_count,
                100 * len(products) / max(1, len(products) + error_count),
            )

        return products

    def _process_json_array(self, raw_file: Path) -> List[Product]:
        with raw_file.open("r", encoding="utf-8-sig") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError(f"{raw_file.name} must contain a list of products")

        products: List[Product] = []
        error_count = 0

        def _process_item(item: Dict[str, Any]):
            try:
                # Skip only if title is missing
                if not item.get("title", "").strip():
                    logger.warning("Item missing title - skipped")
                    return None, True

                # Set defaults for required fields
                item.setdefault('description', 'No description available')
                item.setdefault('main_category', Path(raw_file).stem.replace("_", " ").title())
                item.setdefault('average_rating', 'No rating available')
                
                # Clean price
                price = item.get('price')
                if price is None:
                    item['price'] = 'Price not available'
                elif isinstance(price, str):
                    cleaned_price = ''.join(c for c in price if c.isdigit() or c == '.')
                    if cleaned_price:
                        try:
                            item['price'] = float(cleaned_price)
                        except ValueError:
                            item['price'] = 'Price not available'
                    else:
                        item['price'] = 'Price not available'

                # Ensure details structure
                if 'details' not in item or not isinstance(item['details'], dict):
                    item['details'] = {'features': [], 'specifications': {}}
                else:
                    item['details'].setdefault('features', [])
                    item['details'].setdefault('specifications', {})

                # Infer product type and tags
                specs = item['details']['specifications']
                if not item.get('product_type'):
                    item['product_type'] = self._infer_product_type(item['title'], specs)
                if not item.get('tags'):
                    item['tags'] = self._extract_tags(item['title'], specs)

                product = Product.from_dict(item)
                product.clean_image_urls()
                return product, False
            except Exception as e:
                logger.warning(f"Error processing item: {e}\nItem: {item}")
                return None, True

        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            results = list(tqdm(exe.map(_process_item, items), 
                        total=len(items), 
                        desc=f"Processing {raw_file.name}", 
                        disable=self.disable_tqdm))

        for product, is_error in results:
            if is_error:
                error_count += 1
            elif product is not None:
                products.append(product)

        if error_count:
            logger.warning(
                "%s: %d/%d invalid (%.1f%% success)",
                raw_file.name,
                error_count,
                len(products) + error_count,
                100 * len(products) / max(1, len(products) + error_count),
            )

        return products

    def save_standardized_json(self, products: List[Product], output_file: Union[str, Path]) -> None:
        """Guardar un único JSON estandarizado en processed/"""
        output_file = Path(output_file)
        standardized_data = [product.model_dump() for product in products]
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(standardized_data, f, ensure_ascii=False, indent=4)
        logger.info("Guardado JSON estandarizado en %s", output_file)


    def clear_cache(self):
        cache_file = self.processed_dir / "products.json"
        if cache_file.exists():
            cache_file.unlink()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Procesar archivos JSON/JSONL a JSON unificado")
    parser.add_argument("--input", type=str, default="./data/raw", help="Directorio con archivos fuente")
    parser.add_argument("--output", type=str, default="./data/processed/products.json", help="Archivo JSON de salida")
    args = parser.parse_args()

    loader = DataLoader(raw_dir=args.input)
    loader.load_data(output_file=args.output)