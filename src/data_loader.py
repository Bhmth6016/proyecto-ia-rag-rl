import json
import pickle
import re
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_price(price_str):
    """Convierte un string de precio a float o None si no es válido."""
    if not isinstance(price_str, str):
        return None
    price_str = price_str.lower().replace('from', '').replace('$', '').strip()
    if price_str in ('', '—', '-', 'n/a'):
        return None
    match = re.search(r'(\d+(\.\d+)?)', price_str)
    if match:
        return float(match.group(1))
    return None

def parse_rating(rating) -> Optional[float]:
    """Convierte un rating a float o None si no es válido."""
    if rating is None:
        return None
    try:
        return float(rating)
    except (ValueError, TypeError):
        return None

def parse_categories(categories) -> Optional[list]:
    """Valida y normaliza las categorías."""
    if categories is None:
        return None
    if not isinstance(categories, list):
        return None
    return [str(cat).strip() for cat in categories if cat and str(cat).strip()] or None

def parse_details(details) -> Optional[dict]:
    """Valida los detalles del producto."""
    if details is None:
        return None
    if not isinstance(details, dict):
        return None
    return {k: v for k, v in details.items() if v is not None} or None

def parse_main_category(category) -> Optional[str]:
    """Valida la categoría principal."""
    if category is None:
        return None
    category = str(category).strip()
    return category if category else None

class DataLoader:
    def __init__(self, raw_dir: str = None, processed_dir: str = None):
        script_dir = Path(__file__).parent.resolve()
        base_dir = script_dir.parent / "data"
        
        self.raw_dir = Path(raw_dir) if raw_dir else (base_dir / "raw")
        self.processed_dir = Path(processed_dir) if processed_dir else (base_dir / "processed")

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _get_raw_files(self) -> List[Path]:
        return list(self.raw_dir.glob("meta_*.jsonl"))

    def _get_cache_file(self, source_file: Path) -> Path:
        return self.processed_dir / f"{source_file.stem}_processed.pkl"

    def _load_single_file(self, file_path: Path) -> List[Dict]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    processed_item = self._process_item(item)
                    if processed_item:
                        data.append(processed_item)
                except Exception as e:
                    logger.warning(f"Error procesando línea en {file_path.name}: {e}")
        return data
    def _process_item(self, item: Dict) -> Optional[Dict]:
        try:
            # Procesamiento de imágenes
            main_image = None
            if item.get('images'):
                for img in item.get('images', []):
                    if isinstance(img, dict):
                        if img.get('variant') == 'MAIN' and 'large' in img:
                            main_image = img['large']
                            break
                if not main_image and item['images']:
                    first_img = item['images'][0]
                    if isinstance(first_img, dict):
                        main_image = first_img.get('large')

            # Procesamiento de todos los campos
            processed_item = {
                "main_category": parse_main_category(item.get("main_category")),
                "title": str(item.get("title", "")).strip() or None,
                "average_rating": parse_rating(item.get("average_rating")),
                "price": parse_price(item.get("price")),
                "images": {"large": main_image} if main_image else None,
                "categories": parse_categories(item.get("categories")),
                "details": parse_details(item.get("details"))  # <-- Aquí estaba el error (paréntesis extra)
            }

            # Validación mínima del ítem
            if not processed_item["title"]:
                logger.debug(f"Ítem sin título válido: {item.get('asin')}")
                return None

            return processed_item

        except Exception as e:
            logger.error(f"Error procesando ítem {item.get('asin', 'sin_asin')}: {e}")
            return None

    def _needs_processing(self, raw_file: Path, cache_file: Path) -> bool:
        """Verifica si el archivo necesita ser procesado."""
        if not cache_file.exists():
            return True
        if raw_file.stat().st_mtime > cache_file.stat().st_mtime:
            return True
        try:
            with open(cache_file, 'rb') as f:
                pickle.load(f)
            return False
        except:
            return True

    def load_data(self, use_cache: bool = True) -> List[Dict]:
        all_data = []
        raw_files = self._get_raw_files()
        
        if not raw_files:
            logger.warning(f"No se encontraron archivos en {self.raw_dir}")
            return []

        for raw_file in raw_files:
            cache_file = self._get_cache_file(raw_file)
            
            if use_cache and not self._needs_processing(raw_file, cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        all_data.extend(pickle.load(f))
                    logger.info(f"Datos cargados desde caché: {cache_file.name}")
                    continue
                except Exception as e:
                    logger.warning(f"Error cargando caché {cache_file.name}: {e}")

            file_data = self._load_single_file(raw_file)
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(file_data, f)
                logger.info(f"Datos guardados en caché: {cache_file.name}")
            except Exception as e:
                logger.error(f"Error guardando caché {cache_file.name}: {e}")

            all_data.extend(file_data)

        return all_data