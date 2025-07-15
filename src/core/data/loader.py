# src/core/data/loader.py
"""
DataLoader with dynamic paths & cache policy from settings.py
"""

from __future__ import annotations

import json
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from pydantic import ValidationError

from tqdm import tqdm

from src.core.data.product import Product
from src.core.config import settings
from src.core.utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Keyword maps
# ------------------------------------------------------------------
_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "backpack": ["mochila", "backpack", "bagpack", "laptop bag"],
    "headphones": ["auriculares", "headphones", "headset", "earbuds"],
    "speaker": ["altavoz", "speaker", "bluetooth speaker", "portable speaker"],
    "keyboard": ["teclado", "keyboard", "mechanical keyboard"],
    "mouse": ["ratón", "mouse", "wireless mouse"],
    "monitor": ["monitor", "pantalla", "screen", "display"],
    "camera": ["cámara", "camera", "webcam", "dslr"],
    "home_appliance": ["aspiradora", "vacuum", "microondas", "microwave"],
}

_TAG_KEYWORDS: Dict[str, List[str]] = {
    "waterproof": ["waterproof", "water resistant", "resistente al agua"],
    "wireless": ["wireless", "bluetooth", "inalámbrico", "wifi"],
    "portable": ["portable", "pocket", "ligero", "lightweight"],
    "gaming": ["gaming", "gamer", "rgb", "for gaming"],
    "travel": ["travel", "viaje", "suitcase", "carry-on"],
    "usb-c": ["usb-c", "type-c", "usb type c"],
    "noise-cancelling": ["noise cancelling", "noise reduction", "anc"],
    "fast-charging": ["fast charging", "quick charge", "carga rápida"],
}

class DataLoader:
    """
    Unified loader that:
    • reads files from settings.RAW_DIR
    • caches validated objects to settings.PROC_DIR
    • obeys settings.CACHE_ENABLED
    """

    def __init__(
        self,
        *,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        cache_enabled: Optional[bool] = None,
        max_workers: int = 4,
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else settings.RAW_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else settings.PROC_DIR
        self.cache_enabled = cache_enabled if cache_enabled is not None else settings.CACHE_ENABLED
        self.max_workers = max_workers

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_data(self, *, use_cache: bool = True) -> List[Product]:
        """Load all files in raw_dir, returning validated Product objects."""
        files = list(self.raw_dir.glob("*.json")) + list(self.raw_dir.glob("*.jsonl"))
        if not files:
            logger.warning("No product files found in %s", self.raw_dir)
            return []

        all_products: List[Product] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            future_to_file = {exe.submit(self.load_single_file, f, use_cache=use_cache): f for f in files}
            for future in tqdm(as_completed(future_to_file), total=len(files), desc="Files"):
                all_products.extend(future.result())

        logger.info("Loaded %d products from %d files", len(all_products), len(files))
        return all_products

    def load_single_file(
        self,
        raw_file: Union[str, Path],
        *,
        use_cache: bool = True,
    ) -> List[Product]:
        """Load (and optionally cache) a single file."""
        raw_file = Path(raw_file)
        cache_file = self._cache_path(raw_file)

        if use_cache and self._cache_valid(raw_file, cache_file):
            return self._load_cache(cache_file)

        products = self._process_file(raw_file)
        if self.cache_enabled and use_cache and products:
            self._save_cache(products, cache_file)
        return products

    def clear_cache(self) -> int:
        """Delete all *.pkl caches; returns #removed."""
        pickles = list(self.processed_dir.glob("*_processed.pkl"))
        removed = 0
        for pkl in pickles:
            try:
                pkl.unlink()
                removed += 1
            except Exception as e:
                logger.error("Could not delete %s: %s", pkl.name, e)
        logger.info("Deleted %d cache files", removed)
        return removed

    def export_json(
        self,
        products: List[Product],
        outfile: Union[str, Path],
        *,
        indent: int = 2,
    ) -> None:
        """Write validated products to a single JSON file."""
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)

        with outfile.open("w", encoding="utf-8") as f:
            json.dump(
                [p.dict() for p in tqdm(products, desc="Export JSON")],
                f,
                indent=indent,
                ensure_ascii=False,
                default=str,
            )
        logger.info("Exported %d products to %s", len(products), outfile)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _process_file(self, raw_file: Path) -> List[Product]:
        """JSON/JSONL -> List[Product] with validation."""
        if raw_file.suffix.lower() == ".jsonl":
            return self._process_jsonl(raw_file)
        else:
            return self._process_json_array(raw_file)

    # ---- NEW: helpers for normalization & tagging --------------------
    @staticmethod
    def _infer_product_type(title: str, specs: Dict[str, str]) -> Optional[str]:
        text = (title + " " + " ".join(specs.values())).lower()
        for ptype, kw_list in _CATEGORY_KEYWORDS.items():
            if any(kw in text for kw in kw_list):
                return ptype
        return None

    @staticmethod
    def _extract_tags(title: str, specs: Dict[str, str]) -> List[str]:
        text = (title + " " + " ".join(specs.values())).lower()
        tags = []
        for tag, kw_list in _TAG_KEYWORDS.items():
            if any(kw in text for kw in kw_list):
                tags.append(tag)
        return tags

    # ------------------------------------------------------------------
    def _process_jsonl(self, raw_file: Path) -> List[Product]:
        products: List[Product] = []
        error_count = 0

        def _process_line(idx_line):
            idx, line = idx_line
            try:
                item = json.loads(line.strip())
                if not isinstance(item, dict):
                    logger.warning(f"Line {idx}: Expected dict, got {type(item)}")
                    return None

                # Ensure description is properly formatted
                if "description" in item and isinstance(item["description"], list):
                    item["description"] = " ".join(str(x) for x in item["description"] if x)

                # Enrich
                specs = item.get("details", {}).get("specifications", {})
                if not item.get("product_type"):
                    item["product_type"] = self._infer_product_type(item.get("title", ""), specs)
                if not item.get("tags"):
                    item["tags"] = self._extract_tags(item.get("title", ""), specs)

                product = Product.from_dict(item)
                if product.title and product.title.strip():
                    product.clean_image_urls()
                    return product
                else:
                    raise ValidationError("Missing or empty title")
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Error processing line {idx}: {str(e)}")
                return None

        with raw_file.open("r", encoding="utf-8") as f:
            lines = list(enumerate(f))

        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            results = list(
                tqdm(
                    exe.map(_process_line, lines),
                    total=len(lines),
                    desc=f"Processing {raw_file.name}",
                )
            )
        for res in results:
            if res is None:
                error_count += 1
            else:
                products.append(res)

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
        with raw_file.open("r", encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError(f"{raw_file.name} must contain a list of products")

        def _process_item(item: Dict[str, Any]) -> Optional[Product]:
            try:
                specs = item.get("details", {}).get("specifications", {})
                if not item.get("product_type"):
                    item["product_type"] = self._infer_product_type(item.get("title", ""), specs)
                if not item.get("tags"):
                    item["tags"] = self._extract_tags(item.get("title", ""), specs)

                product = Product.from_dict(item)
                if product.title and product.title.strip() and product.main_category:
                    product.clean_image_urls()
                    return product
                else:
                    raise ValidationError("Missing or empty title / main_category")
            except ValidationError:
                return None

        products: List[Product] = []
        error_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            results = list(
                tqdm(
                    exe.map(_process_item, items),
                    total=len(items),
                    desc=f"Processing {raw_file.name}",
                )
            )
        for res in results:
            if res is None:
                error_count += 1
            else:
                products.append(res)

        if error_count:
            logger.warning(
                "%s: %d/%d invalid (%.1f%% success)",
                raw_file.name,
                error_count,
                len(products) + error_count,
                100 * len(products) / max(1, len(products) + error_count),
            )
        return products

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _cache_path(self, raw_file: Path) -> Path:
        return self.processed_dir / f"{raw_file.stem}_processed.pkl"

    def _cache_valid(self, raw_file: Path, cache_file: Path) -> bool:
        return (
            cache_file.exists()
            and cache_file.stat().st_mtime >= raw_file.stat().st_mtime
        )

    def _load_cache(self, cache_file: Path) -> List[Product]:
        try:
            with cache_file.open("rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning("Cache read failed: %s; re-processing", e)
            return []

    def _save_cache(self, products: List[Product], cache_file: Path) -> None:
        try:
            with cache_file.open("wb") as f:
                pickle.dump(products, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error("Cache write failed: %s", e)