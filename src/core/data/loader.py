# src/core/data/loader.py

import json, pickle, re, hashlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from threading import Lock
from tqdm import tqdm

from src.core.data.product import Product
from src.core.config import settings
from src.core.utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
_CATEGORY_KEYWORDS = {
    "video_games": ["video game", "xbox", "playstation", "nintendo", "steam", "controller", "gaming"],
    "software": ["software", "app", "license", "antivirus", "IDE", "download", "activation"],
    "industrial": ["industrial", "scientific", "laboratory", "equipment", "tool", "measurement"],
    "electronics": ["electronics", "camera", "phone", "laptop", "speaker", "battery", "usb"],
    "beauty": ["beauty", "makeup", "skincare", "cosmetics", "perfume", "cream", "shampoo"]
}

_TAG_KEYWORDS = {
    "wireless": ["wireless", "bluetooth", "wifi"],
    "portable": ["portable", "lightweight", "compact"],
    "waterproof": ["waterproof", "resistant"],
    "gaming": ["gaming", "rgb"],
    "eco-friendly": ["eco", "recycled", "biodegradable"],
    "digital": ["digital", "online", "download"],
    "premium": ["premium", "luxury"]
}

FILENAME_TO_CATEGORY = {
    "Video_Games": "video_games",
    "Software": "software", 
    "Industrial_and_Scientific": "industrial",
    "Electronics": "electronics",
    "Beauty_and_Personal_Care": "beauty"
}


class DataLoader:
    """Cargador optimizado de productos con caché, limpieza y clasificación automática."""

    def __init__(
        self,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        cache_enabled: bool = True,
        max_workers: int = 4,
        disable_tqdm: bool = False,
    ):
        self.raw_dir = Path(raw_dir or settings.RAW_DIR)
        self.processed_dir = Path(processed_dir or settings.PROC_DIR)
        self.cache_enabled = cache_enabled
        self.max_workers = max_workers
        self.disable_tqdm = disable_tqdm
        self._error_lock = Lock()

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self._compiled_category_patterns = self._compile_patterns(_CATEGORY_KEYWORDS)
        self._compiled_tag_patterns = self._compile_patterns(_TAG_KEYWORDS)

    # ------------------------- Utilidades internas -------------------------
    @staticmethod
    def _compile_patterns(data: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
        return {k: re.compile(r'\b(?:' + '|'.join(map(re.escape, v)) + r')\b', re.I) for k, v in data.items()}

    @staticmethod
    def _file_hash(path: Path) -> str:
        h = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                h.update(chunk)
        return h.hexdigest()

    def _cache_path(self, raw_file: Path) -> Path:
        return self.processed_dir / "cache" / f"{raw_file.stem}_{self._file_hash(raw_file)[:8]}.pkl"

    # ------------------------- Caché -------------------------
    def _load_cache(self, raw_file: Path) -> Optional[List[Product]]:
        if not self.cache_enabled:
            return None
        cache_file = self._cache_path(raw_file)
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            if data.get("file_hash") == self._file_hash(raw_file):
                return data["products"]
        except Exception as e:
            logger.warning(f"Error loading cache {raw_file.name}: {e}")
        return None

    def _save_cache(self, raw_file: Path, products: List[Product]):
        if not self.cache_enabled:
            return
        try:
            cache_file = self._cache_path(raw_file)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({"products": products, "file_hash": self._file_hash(raw_file)}, f)
        except Exception as e:
            logger.warning(f"Error saving cache {raw_file.name}: {e}")

    # ------------------------- Limpieza y clasificación -------------------------
    def _infer_category(self, title: str, specs: Dict[str, Any]) -> str:
        text = f"{title.lower()} {' '.join(map(str, specs.values())).lower()}"
        for cat, pattern in self._compiled_category_patterns.items():
            if pattern.search(text):
                return cat
        return "unknown"

    def _extract_tags(self, title: str, specs: Dict[str, Any]) -> List[str]:
        text = f"{title.lower()} {' '.join(map(str, specs.values())).lower()}"
        return [tag for tag, pattern in self._compiled_tag_patterns.items() if pattern.search(text)]

    @staticmethod
    def _get_category_from_filename(filename: str) -> str:
        return FILENAME_TO_CATEGORY.get(Path(filename).stem, "unknown")

    def _clean_item(self, item: Dict[str, Any], filename: str) -> Dict[str, Any]:
        title = item.get("title", "").strip()
        if not title:
            raise ValueError("Missing title")

        item["description"] = (
            " ".join(item["description"]) if isinstance(item.get("description"), list)
            else item.get("description") or "No description available"
        )
        item["main_category"] = self._get_category_from_filename(filename)

        price = item.get("price")
        if isinstance(price, str):
            try:
                item["price"] = float(re.sub(r"[^\d.]", "", price)) or "Price not available"
            except ValueError:
                item["price"] = "Price not available"
        elif price is None:
            item["price"] = "Price not available"

        item.setdefault("average_rating", "No rating available")
        item.setdefault("details", {"features": [], "specifications": {}})
        specs = item["details"].get("specifications", {})
        item["product_type"] = item.get("product_type") or self._infer_category(title, specs)
        item["tags"] = item.get("tags") or self._extract_tags(title, specs)
        return item

    # ------------------------- Procesamiento -------------------------
    def _process_file(self, raw_file: Path) -> List[Product]:
        """Procesa JSON o JSONL"""
        logger.info(f"Processing {raw_file.name}")
        open_fn = (lambda: (json.loads(l.strip()) for l in raw_file.open(encoding='utf-8'))) \
            if raw_file.suffix == ".jsonl" else \
            (lambda: json.load(raw_file.open(encoding='utf-8-sig')))

        items = list(open_fn())
        if not isinstance(items, list):
            raise ValueError(f"{raw_file.name} must contain a list or valid JSONL lines")

        def process(item):
            try:
                cleaned = self._clean_item(item, raw_file.name)
                p = Product.from_dict(cleaned)
                p.clean_image_urls()
                return p
            except Exception:
                return None

        with ProcessPoolExecutor(max_workers=self.max_workers) as exe:
            products = [p for p in tqdm(exe.map(process, items), total=len(items),
                                        desc=f"Processing {raw_file.name}", disable=self.disable_tqdm) if p]
        return products

    # ------------------------- Flujo principal -------------------------
    def load_data(self, use_cache: Optional[bool] = None, output_file: Optional[Union[str, Path]] = None) -> List[Product]:
        use_cache = self.cache_enabled if use_cache is None else use_cache
        output_file = Path(output_file or self.processed_dir / "products.json")

        if use_cache and output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                return [Product.from_dict(p) for p in json.load(f)]

        files = [self.raw_dir / f"{name}.jsonl" for name in FILENAME_TO_CATEGORY.keys()]
        files = [f for f in files if f.exists()]
        if not files:
            logger.warning("No files found in %s", self.raw_dir)
            return []

        products = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(files))) as exe:
            for result in tqdm(exe.map(self.load_single_file, files), total=len(files),
                               desc="Files", disable=self.disable_tqdm):
                products.extend(result or [])

        if products:
            self.save_standardized_json(products, output_file)
        return products

    def load_single_file(self, raw_file: Union[str, Path]) -> List[Product]:
        raw_file = Path(raw_file)
        cached = self._load_cache(raw_file)
        if cached is not None:
            return cached
        products = self._process_file(raw_file)
        if products:
            self._save_cache(raw_file, products)
        return products

    @staticmethod
    def save_standardized_json(products: List[Product], output_file: Union[str, Path]):
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            json.dump([p.model_dump() for p in products], f, ensure_ascii=False, indent=2)
        logger.info(f"Saved standardized JSON: {output_file}")

    def clear_cache(self):
        cache_dir = self.processed_dir / "cache"
        if cache_dir.exists():
            import shutil; shutil.rmtree(cache_dir)
            logger.info("Cache cleared.")


# ------------------------- Ejecución directa -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process product files to unified JSON")
    parser.add_argument("--input", type=str, default="./data/raw", help="Input directory")
    parser.add_argument("--output", type=str, default="./data/processed/products.json", help="Output file")
    args = parser.parse_args()

    loader = DataLoader(raw_dir=args.input)
    products = loader.load_data(output_file=args.output)
    print(f"✅ Processed {len(products)} products")
