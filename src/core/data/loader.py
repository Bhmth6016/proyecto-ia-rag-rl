# src/core/data/loader.py
"""
DataLoader con rutas dinámicas y política de caché desde settings.py
"""

from __future__ import annotations

import json
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from tqdm import tqdm

from src.core.data.product import Product
from src.core.config import settings
from src.core.utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Mapeo de palabras clave
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
        cache_enabled: Optional[bool] = None,
        max_workers: int = 4,
        disable_tqdm: bool = False,
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else settings.RAW_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else settings.PROC_DIR
        self.cache_enabled = cache_enabled if cache_enabled is not None else settings.CACHE_ENABLED
        self.max_workers = max_workers
        self.disable_tqdm = disable_tqdm

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

    # ------------------------------------------------------------------
    def load_data(self, *, use_cache: bool = True) -> List[Product]:
        """Cargar todos los archivos en raw_dir, devolviendo objetos Product validados."""
        files = list(self.raw_dir.glob("*.json")) + list(self.raw_dir.glob("*.jsonl"))
        if not files:
            logger.warning("No se encontraron archivos de productos en %s", self.raw_dir)
            return []

        all_products: List[Product] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            future_to_file = {exe.submit(self.load_single_file, f, use_cache=use_cache): f for f in files}
            for future in tqdm(future_to_file, desc="Archivos", total=len(files), disable=self.disable_tqdm):
                all_products.extend(future.result())

        logger.info("Cargados %d productos de %d archivos", len(all_products), len(files))
        return all_products

    def load_single_file(
        self,
        raw_file: Union[str, Path],
        *,
        use_cache: bool = True,
    ) -> List[Product]:
        raw_file = Path(raw_file)
        cache_file = self.processed_dir / f"{raw_file.stem}_processed.pkl"

        if use_cache and self.cache_enabled and cache_file.exists() and cache_file.stat().st_mtime >= raw_file.stat().st_mtime:
            logger.info("Cargando desde caché: %s", cache_file.name)
            try:
                with cache_file.open("rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning("Fallo leyendo caché %s: %s, re-procesando", cache_file.name, e)

        # Procesar sin caché
        logger.info("Procesando archivo: %s", raw_file.name)
        products = self._process_jsonl(raw_file) if raw_file.suffix.lower() == ".jsonl" else self._process_json_array(raw_file)

        if self.cache_enabled and products:
            try:
                with cache_file.open("wb") as f:
                    pickle.dump(products, f)
            except Exception as e:
                logger.error("Error guardando caché %s: %s", cache_file.name, e)

        return products

    # ------------------------------------------------------------------
    def _process_jsonl(self, raw_file: Path) -> List[Product]:
        products: List[Product] = []
        error_count = 0

        def _process_line(idx_line):
            idx, line = idx_line
            try:
                item = json.loads(line.strip())
                if not isinstance(item, dict):
                    logger.warning(f"Línea {idx}: Se esperaba un diccionario, se obtuvo {type(item)}")
                    return None, True

                if "description" in item and isinstance(item["description"], list):
                    item["description"] = " ".join(str(x) for x in item["description"] if x)

                specs = item.get("details", {}).get("specifications", {})
                if not item.get("product_type"):
                    item["product_type"] = self._infer_product_type(item.get("title", ""), specs)
                if not item.get("tags"):
                    item["tags"] = self._extract_tags(item.get("title", ""), specs)

                product = Product.from_dict(item)
                if product.title and product.title.strip():
                    product.clean_image_urls()
                    return product, False
                else:
                    logger.warning(f"Línea {idx}: título ausente o vacío. Línea: {line.strip()}")
                    return None, True

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Error procesando línea {idx}: {e}. Línea: {line.strip()}")
                return None, True

        with raw_file.open("r", encoding="utf-8") as f:
            lines = list(enumerate(f))

        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            results = list(tqdm(exe.map(_process_line, lines), total=len(lines), desc=f"Procesando {raw_file.name}", disable=self.disable_tqdm))

        for product, is_error in results:
            if is_error:
                error_count += 1
            elif product is not None:
                products.append(product)

        if error_count:
            logger.warning(
                "%s: %d/%d inválidos (%.1f%% éxito)",
                raw_file.name,
                error_count,
                len(products) + error_count,
                100 * len(products) / max(1, len(products) + error_count),
            )

        return products

    def _process_json_array(self, raw_file: Path) -> List[Product]:
        with raw_file.open("r", encoding="utf-8-sig") as f:  # Use utf-8-sig to handle BOM
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError(f"{raw_file.name} debe contener una lista de productos")

        products: List[Product] = []
        error_count = 0

        @staticmethod
        def _process_item(item: Dict[str, Any]):
            try:
                # Ensure required fields with sensible defaults
                item.setdefault('title', 'Untitled Product')
                item.setdefault('main_category', item.get('category', 'Uncategorized'))
                item.setdefault('description', '')
                item.setdefault('tags', [])
                
                # Process specifications if they exist
                specs = item.get('details', {}).get('specifications', {})
                
                # Infer product type if not specified
                if not item.get('product_type'):
                    item['product_type'] = self._infer_product_type(item['title'], specs)
                
                # Create product instance
                product = Product.from_dict(item)
                return product, False
                
            except Exception as e:
                logger.warning(f"Error processing item: {e}\nItem: {item}")
                return None, True

        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            results = list(tqdm(exe.map(_process_item, items), total=len(items), desc=f"Procesando {raw_file.name}", disable=self.disable_tqdm))

        for product, is_error in results:
            if is_error:
                error_count += 1
            elif product is not None:
                products.append(product)

        if error_count:
            logger.warning(
                "%s: %d/%d inválidos (%.1f%% éxito)",
                raw_file.name,
                error_count,
                len(products) + error_count,
                100 * len(products) / max(1, len(products) + error_count),
            )

        return products
    
    def clear_cache(self) -> int:
        """Elimina todos los archivos .pkl en el directorio de procesados"""
        cache_files = list(self.processed_dir.glob("*.pkl"))
        for f in cache_files:
            f.unlink()
        return len(cache_files)