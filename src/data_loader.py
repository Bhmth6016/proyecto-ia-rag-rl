import json
import pickle
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Iterator
from pydantic import BaseModel, Field, field_validator, ValidationError
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductModel(BaseModel):
    main_category: Optional[str] = Field(default=None)
    title: Optional[str] = Field(default=None)
    average_rating: Optional[float] = Field(default=None, ge=0, le=5)
    price: Optional[float] = Field(default=None, ge=0)
    images: Optional[dict] = Field(default_factory=dict)  # Nunca será None
    categories: Optional[List[str]] = Field(default_factory=list)  # Nunca será None
    details: Optional[dict] = Field(default_factory=dict)  # Nunca será None

    @field_validator('price', mode='before')
    def parse_price(cls, value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        # Reutiliza tu vieja función `parse_price` aquí
        return parse_price(str(value))

    @field_validator('categories', mode='before')
    def parse_categories(cls, value):
        if value is None:
            return []
        if not isinstance(value, list):
            return []
        return [str(cat).strip() for cat in value if cat and str(cat).strip()]

    @field_validator('details', mode='before')
    def parse_details(cls, value):
        return {} if value is None or not isinstance(value, dict) else value
def parse_float(value: Union[str, int, float, None]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
def parse_price(price_str: Union[str, int, float, None]) -> Optional[float]:
    """Tu implementación original (la que ya funcionaba bien)."""
    if price_str is None:
        return None
    if isinstance(price_str, (int, float)):
        return float(price_str)
    price_str = str(price_str).lower().replace('from', '').replace('$', '').replace(',', '').strip()
    if price_str in ('', '—', '-', 'n/a', 'null'):
        return None
    match = re.search(r'(\d+\.?\d*)', price_str)
    return float(match.group(1)) if match else None

class DataLoader:
    def __init__(self, raw_dir: Optional[Union[str, Path]] = None,
                 processed_dir: Optional[Union[str, Path]] = None):
        script_dir = Path(__file__).parent.resolve()
        base_dir = script_dir.parent / "data"
        self._raw_dir = Path(raw_dir) if raw_dir else (base_dir / "raw")
        self._processed_dir = Path(processed_dir) if processed_dir else (base_dir / "processed")
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)

    @property
    def raw_dir(self) -> Path:
        return self._raw_dir

    @property
    def processed_dir(self) -> Path:
        return self._processed_dir

    def _get_raw_files(self) -> List[Path]:
        return list(self.raw_dir.glob("meta_*.jsonl"))

    def _get_cache_file(self, source_file: Path) -> Path:
        return self.processed_dir / f"{source_file.stem}_processed.pkl"

    def _load_single_file(self, file_path: Path) -> List[Dict]:
        data = []
        error_count = 0
        lines = file_path.read_text(encoding='utf-8').splitlines()
        for i, line in enumerate(lines, 1):
            try:
                item = json.loads(line)
                if processed := self._process_item(item):
                    data.append(processed)
            except json.JSONDecodeError:
                error_count += 1
                logger.warning(f"[{file_path.name}] JSON inválido en línea {i}")
            except Exception as e:
                error_count += 1
                logger.warning(f"[{file_path.name}] Error en línea {i}: {e}")
        if error_count > 0:
            logger.info(f"{file_path.name} → {error_count}/{len(lines)} errores ({round((1-error_count/len(lines))*100, 2)}% éxito)")
        return data

    def _process_item(self, item: Dict) -> Optional[Dict]:
        try:
            images = []
            if isinstance(item.get('images'), list):
                for img in item['images']:
                    if isinstance(img, dict) and 'large' in img:
                        images.append({'large': str(img['large'])})
            product_data = {
                "main_category": str(item.get("main_category", "")).strip() or None,
                "title": str(item.get("title", "")).strip() or None,
                "average_rating": item.get("average_rating"),
                "price": item.get("price"),
                "images": images[0] if images else None,
                "categories": item.get("categories"),
                "details": item.get("details", {})
            }
            return ProductModel(**product_data).model_dump()
        except Exception as e:
            logger.error(f"Error procesando ítem: {e}")
            return None

    def _needs_processing(self, raw_file: Path, cache_file: Path) -> bool:
        return not cache_file.exists() or raw_file.stat().st_mtime > cache_file.stat().st_mtime

    def load_data(self, use_cache: bool = True) -> List[Dict]:
        all_data = []
        total_errors = 0
        total_items = 0
        for raw_file in self._get_raw_files():
            cache_file = self._get_cache_file(raw_file)
            if use_cache and not self._needs_processing(raw_file, cache_file):
                try:
                    with cache_file.open('rb') as f:
                        cached_data = pickle.load(f)
                        all_data.extend(cached_data)
                        total_items += len(cached_data)
                    continue
                except Exception as e:
                    logger.warning(f"No se pudo leer la caché '{cache_file.name}': {e}")
            file_data = self._load_single_file(raw_file)
            total_items += len(file_data)
            try:
                with cache_file.open('wb') as f:
                    pickle.dump(file_data, f)
                all_data.extend(file_data)
            except Exception as e:
                logger.error(f"Fallo al guardar la caché '{cache_file.name}': {e}")
        logger.info(f"Carga finalizada: {len(all_data)} productos | Éxito: {round((1 - total_errors / max(1, total_items)) * 100, 2)}%")
        return all_data

    def load_by_main_category(self, use_cache: bool = True) -> Dict[str, List[Dict]]:
        categorized = defaultdict(list)
        for product in self.load_data(use_cache):
            if category := product.get("main_category"):
                categorized[category].append(product)
        return dict(categorized)

    def load_in_batches(self, batch_size: int = 1000) -> Iterator[List[Dict]]:
        for raw_file in self._get_raw_files():
            batch = []
            for i, line in enumerate(raw_file.read_text(encoding='utf-8').splitlines()):
                try:
                    item = json.loads(line)
                    if processed := self._process_item(item):
                        batch.append(processed)
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                except Exception as e:
                    logger.warning(f"Error en línea {i+1} de {raw_file.name}: {e}")
            if batch:
                yield batch
