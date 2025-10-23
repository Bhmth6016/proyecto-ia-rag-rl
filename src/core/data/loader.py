# src/core/data/loader.py

import json
import pickle
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from threading import Lock
import hashlib

from tqdm import tqdm

from src.core.data.product import Product
from src.core.config import settings
from src.core.utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Configuración optimizada para las categorías específicas
# ------------------------------------------------------------------
_CATEGORY_KEYWORDS = {
    "video_games": [
        "video game", "xbox", "playstation", "nintendo", "steam", "dlc", 
        "gamer", "gamepad", "controller", "console", "gaming", "esrb"
    ],
    "software": [
        "software", "app", "application", "program", "license", "subscription", 
        "antivirus", "editor", "IDE", "download", "activation", "key"
    ],
    "industrial": [
        "industrial", "scientific", "laboratory", "measurement", "calibration",
        "safety", "equipment", "tool", "instrument", "manufacturing"
    ],
    "electronics": [
        "electronics", "camera", "phone", "tablet", "laptop", "computer",
        "audio", "speaker", "headphones", "battery", "charger", "usb"
    ],
    "beauty": [
        "beauty", "personal care", "makeup", "skincare", "cosmetics", "perfume",
        "serum", "lipstick", "cream", "mascara", "shampoo", "conditioner"
    ]
}

_TAG_KEYWORDS = {
    "wireless": ["wireless", "bluetooth", "wifi"],
    "portable": ["portable", "lightweight", "compact", "foldable"],
    "waterproof": ["waterproof", "water resistant"],
    "gaming": ["gaming", "gamer", "rgb"],
    "fast-charging": ["fast charging", "quick charge"],
    "eco-friendly": ["eco", "recycled", "biodegradable", "green"],
    "digital": ["digital", "online", "download", "streaming"],
    "premium": ["premium", "luxury", "exclusive"]
}

# Mapeo de nombres de archivo a categorías
FILENAME_TO_CATEGORY = {
    "Video_Games": "video_games",
    "Software": "software", 
    "Industrial_and_Scientific": "industrial",
    "Electronics": "electronics",
    "Beauty_and_Personal_Care": "beauty"
}

class DataLoader:
    """
    Cargador optimizado para las categorías específicas en inglés
    """

    def __init__(
        self,
        *,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        cache_enabled: bool = True,
        max_workers: int = 4,
        disable_tqdm: bool = False,
    ):
        # settings.RAW_DIR: Ruta por defecto a directorio de datos raw desde configuración
        self.raw_dir = Path(raw_dir) if raw_dir else settings.RAW_DIR
        
        # settings.PROC_DIR: Ruta por defecto a directorio procesado desde configuración
        self.processed_dir = Path(processed_dir) if processed_dir else settings.PROC_DIR
        
        self.cache_enabled = cache_enabled
        self.max_workers = max_workers
        self.disable_tqdm = disable_tqdm
        self._error_lock = Lock()

        # Compilar regex para optimización
        self._compiled_category_patterns = self._compile_keyword_patterns(_CATEGORY_KEYWORDS)
        self._compiled_tag_patterns = self._compile_keyword_patterns(_TAG_KEYWORDS)
        
        # Crear sets para búsqueda rápida
        self._category_keyword_sets = {cat: set(kw_list) for cat, kw_list in _CATEGORY_KEYWORDS.items()}
        self._tag_keyword_sets = {tag: set(kw_list) for tag, kw_list in _TAG_KEYWORDS.items()}

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _compile_keyword_patterns(self, keywords_dict: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
        """Compila patrones regex para búsqueda más rápida"""
        compiled = {}
        for key, keywords in keywords_dict.items():
            pattern = r'\b(?:' + '|'.join(map(re.escape, keywords)) + r')\b'
            compiled[key] = re.compile(pattern, re.IGNORECASE)
        return compiled

    def _get_file_hash(self, file_path: Path) -> str:
        """Calcula hash del archivo para detectar cambios"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_cache_file_path(self, raw_file: Path) -> Path:
        """Genera ruta de archivo de caché"""
        cache_filename = f"{raw_file.stem}_{self._get_file_hash(raw_file)[:8]}.pkl"
        return self.processed_dir / "cache" / cache_filename

    def _load_from_cache(self, raw_file: Path) -> Optional[List[Product]]:
        """Intenta cargar productos desde caché por archivo"""
        if not self.cache_enabled:
            return None

        cache_file = self._get_cache_file_path(raw_file)
        cache_dir = cache_file.parent
        cache_dir.mkdir(parents=True, exist_ok=True)

        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                current_hash = self._get_file_hash(raw_file)
                if cached_data.get('file_hash') == current_hash:
                    logger.debug(f"Cache hit for {raw_file.name}")
                    return cached_data['products']
            except Exception as e:
                logger.warning(f"Error loading cache for {raw_file.name}: {e}")
        
        return None

    def _save_to_cache(self, raw_file: Path, products: List[Product]) -> None:
        """Guarda productos en caché por archivo"""
        if not self.cache_enabled:
            return

        try:
            cache_file = self._get_cache_file_path(raw_file)
            cache_data = {
                'products': products,
                'file_hash': self._get_file_hash(raw_file)
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Error saving cache for {raw_file.name}: {e}")

    def _infer_product_type(self, title: str, specs: Dict[str, Any]) -> str:
        """Inferencia de categorías optimizada"""
        text = f"{(title or '').lower()} {' '.join(str(v).lower() for v in specs.values())}"
        text_words = set(text.split())
        
        # Primero probar con intersección de sets (más rápido)
        for category, keywords in self._category_keyword_sets.items():
            if keywords & text_words:
                return category
        
        # Fallback a regex
        for category, pattern in self._compiled_category_patterns.items():
            if pattern.search(text):
                return category
        
        return "unknown"

    def _extract_tags(self, title: str, specs: Dict[str, Any]) -> List[str]:
        """Extracción de tags optimizada"""
        text = f"{(title or '').lower()} {' '.join(str(v).lower() for v in specs.values())}"
        text_words = set(text.split())
        tags = []
        
        # Usar intersección de sets
        for tag, keywords in self._tag_keyword_sets.items():
            if keywords & text_words:
                tags.append(tag)
        
        return tags

    def _get_category_from_filename(self, filename: str) -> str:
        """Obtiene categoría basada en el nombre del archivo"""
        stem = Path(filename).stem
        return FILENAME_TO_CATEGORY.get(stem, stem.lower())

    def _clean_item(self, item: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Limpia y estandariza un producto individual"""
        # Validar título (campo requerido)
        if not item.get("title", "").strip():
            raise ValueError("Missing title")

        # Manejar descripción
        description = item.get("description")
        if not description:
            item["description"] = "No description available"
        elif isinstance(description, list):
            item["description"] = " ".join(str(x) for x in description if x)

        # Establecer categoría desde el nombre del archivo
        item["main_category"] = self._get_category_from_filename(filename)

        # Limpiar precio
        price = item.get("price")
        if price is None:
            item["price"] = "Price not available"
        elif isinstance(price, str):
            cleaned_price = ''.join(c for c in price if c.isdigit() or c == '.')
            if cleaned_price:
                try:
                    item["price"] = float(cleaned_price)
                except ValueError:
                    item["price"] = "Price not available"
            else:
                item["price"] = "Price not available"

        # Establecer rating por defecto
        if item.get("average_rating") is None:
            item["average_rating"] = "No rating available"

        # Estructurar detalles
        if "details" not in item or not isinstance(item["details"], dict):
            item["details"] = {"features": [], "specifications": {}}
        else:
            item["details"].setdefault("features", [])
            item["details"].setdefault("specifications", {})

        # Inferir tipo de producto y tags
        specs = item["details"]["specifications"]
        title = item.get("title", "")
        
        if not item.get("product_type"):
            item["product_type"] = self._infer_product_type(title, specs)
        
        if not item.get("tags"):
            item["tags"] = self._extract_tags(title, specs)

        return item

    def load_data(self, use_cache: Optional[bool] = None, output_file: Union[str, Path] = None) -> List[Product]:
        """Carga todos los archivos de las categorías específicas"""
        if use_cache is None:
            use_cache = self.cache_enabled

        if output_file is None:
            output_file = self.processed_dir / "products.json"

        # Verificar caché global
        if use_cache and Path(output_file).exists():
            logger.info("Loading products from cache")
            return self._load_global_cache(output_file)

        # Cargar solo los archivos específicos
        expected_files = [
            "Video_Games.jsonl", "Software.jsonl", "Industrial_and_Scientific.jsonl",
            "Electronics.jsonl", "Beauty_and_Personal_Care.jsonl"
        ]
        
        files = []
        for expected_file in expected_files:
            file_path = self.raw_dir / expected_file
            if file_path.exists():
                files.append(file_path)
            else:
                logger.warning(f"Expected file not found: {expected_file}")

        if not files:
            logger.warning("No product files found in %s", self.raw_dir)
            return []

        all_products: List[Product] = []
        
        # Procesar archivos en paralelo
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(files))) as exe:
            future_to_file = {exe.submit(self.load_single_file, f): f for f in files}
            for future in tqdm(future_to_file, desc="Files", total=len(files), disable=self.disable_tqdm):
                try:
                    products = future.result()
                    if products:
                        all_products.extend(products)
                except Exception as e:
                    file_path = future_to_file[future]
                    logger.warning(f"Error processing {file_path.name}: {e}")

        if not all_products:
            logger.error("No valid products loaded from any files!")
            return []

        logger.info("Loaded %d products from %d files", len(all_products), len(files))

        # Guardar productos procesados
        self.save_standardized_json(all_products, output_file)

        return all_products

    def _load_global_cache(self, cache_file: Path) -> List[Product]:
        """Carga el caché global"""
        with open(cache_file, "r", encoding="utf-8") as f:
            cached_data = json.load(f)

        valid_products = []
        for item in cached_data:
            try:
                if "details" not in item or item["details"] is None:
                    item["details"] = {}
                elif not isinstance(item["details"], dict):
                    item["details"] = {}

                if not item.get("title", "").strip():
                    continue

                # Product.from_dict(): Convierte diccionario a objeto Product
                product = Product.from_dict(item)
                valid_products.append(product)
            except Exception as e:
                logger.warning(f"Error loading product from cache: {e}")
                continue

        return valid_products

    def load_single_file(self, raw_file: Union[str, Path]) -> List[Product]:
        """Carga un solo archivo"""
        raw_file = Path(raw_file)
        
        # Intentar cargar desde caché
        cached_products = self._load_from_cache(raw_file)
        if cached_products is not None:
            return cached_products

        logger.info("Processing file: %s", raw_file.name)
        
        # Procesar archivo
        if raw_file.suffix.lower() == ".jsonl":
            products = self._process_jsonl(raw_file)
        else:
            products = self._process_json_array(raw_file)

        # Guardar en caché
        if products:
            self._save_to_cache(raw_file, products)

        return products

    def _process_jsonl(self, raw_file: Path) -> List[Product]:
        """Procesa archivo JSONL"""
        with raw_file.open("r", encoding="utf-8") as f:
            lines = list(enumerate(f))

        products = []
        error_count = 0

        def _process_line(args):
            idx, line = args
            try:
                item = json.loads(line.strip())
                if not isinstance(item, dict):
                    return None
                
                cleaned_item = self._clean_item(item, raw_file.name)
                
                # Product.from_dict(): Crea objeto Product desde diccionario
                product = Product.from_dict(cleaned_item)
                product.clean_image_urls()
                return product
            except Exception as e:
                return None

        # Procesar en paralelo
        with ProcessPoolExecutor(max_workers=self.max_workers) as exe:
            results = list(tqdm(
                exe.map(_process_line, lines),
                total=len(lines),
                desc=f"Processing {raw_file.name}",
                disable=self.disable_tqdm
            ))

        for product in results:
            if product is not None:
                products.append(product)
            else:
                error_count += 1

        if error_count:
            logger.warning(
                "%s: %d errors, %d products processed",
                raw_file.name, error_count, len(products)
            )

        return products

    def _process_json_array(self, raw_file: Path) -> List[Product]:
        """Procesa archivo JSON array"""
        with raw_file.open("r", encoding="utf-8-sig") as f:
            items = json.load(f)
        
        if not isinstance(items, list):
            raise ValueError(f"{raw_file.name} must contain a list of products")

        products = []
        error_count = 0

        def _process_item(item):
            try:
                cleaned_item = self._clean_item(item, raw_file.name)
                
                # Product.from_dict(): Convierte diccionario a objeto Product
                product = Product.from_dict(cleaned_item)
                product.clean_image_urls()
                return product
            except Exception:
                return None

        # Procesar en paralelo
        with ProcessPoolExecutor(max_workers=self.max_workers) as exe:
            results = list(tqdm(
                exe.map(_process_item, items),
                total=len(items),
                desc=f"Processing {raw_file.name}",
                disable=self.disable_tqdm
            ))

        for product in results:
            if product is not None:
                products.append(product)
            else:
                error_count += 1

        if error_count:
            logger.warning(
                "%s: %d errors, %d products processed",
                raw_file.name, error_count, len(products)
            )

        return products

    def save_standardized_json(self, products: List[Product], output_file: Union[str, Path]) -> None:
        """Guarda productos en JSON estandarizado"""
        output_file = Path(output_file)
        
        # product.model_dump(): Convierte objeto Product a diccionario
        standardized_data = [product.model_dump() for product in products]
        
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(standardized_data, f, ensure_ascii=False, indent=2)
        logger.info("Saved standardized JSON to %s", output_file)

    def clear_cache(self):
        """Limpia el caché"""
        cache_file = self.processed_dir / "products.json"
        if cache_file.exists():
            cache_file.unlink()
        
        cache_dir = self.processed_dir / "cache"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process product files to unified JSON")
    parser.add_argument("--input", type=str, default="./data/raw", help="Input directory with source files")
    parser.add_argument("--output", type=str, default="./data/processed/products.json", help="Output JSON file")
    args = parser.parse_args()

    loader = DataLoader(raw_dir=args.input)
    products = loader.load_data(output_file=args.output)
    print(f"Processed {len(products)} products")