# src/core/data/loader.py
"""
DataLoader with dynamic paths & cache policy from settings.py
"""

from __future__ import annotations

import json
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from tqdm import tqdm

from src.core.data.product import Product
from src.core.config import settings
from src.core.utils.logger import get_logger

logger = get_logger(__name__)


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
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else settings.RAW_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else settings.PROC_DIR
        self.cache_enabled = cache_enabled if cache_enabled is not None else settings.CACHE_ENABLED

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
        for raw_file in files:
            all_products.extend(self._load_single_file(raw_file, use_cache=use_cache))
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
        products: List[Product] = []
        error_count = 0

        if raw_file.suffix.lower() == ".jsonl":
            with raw_file.open("r", encoding="utf-8") as f:
                total = sum(1 for _ in f)
            with raw_file.open("r", encoding="utf-8") as f:
                for line in tqdm(f, total=total, desc=f"Processing {raw_file.name}"):
                    try:
                        item = json.loads(line.strip())
                        products.append(Product.from_dict(item))
                    except (json.JSONDecodeError, ValidationError):
                        error_count += 1
        else:  # JSON array
            with raw_file.open("r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                raise ValueError(f"{raw_file.name} must contain a list of products")

            for item in items:
                try:
                    products.append(Product.from_dict(item))
                except ValidationError:
                    error_count += 1

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