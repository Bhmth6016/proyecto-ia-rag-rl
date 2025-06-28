import pickle
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from pydantic import ValidationError
from src.validaciones import ProductModel, clean_string

logger = logging.getLogger(__name__)

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
                if not isinstance(item, dict):
                    raise ValueError("El ítem no es un diccionario")
                    
                if processed := self._process_item(item):
                    data.append(processed)
            except json.JSONDecodeError:
                error_count += 1
                logger.warning(f"[{file_path.name}] JSON inválido en línea {i}")
            except Exception as e:
                error_count += 1
                logger.warning(f"[{file_path.name}] Error en línea {i}: {str(e)}")
                
        if error_count > 0:
            logger.info(f"{file_path.name} → {error_count}/{len(lines)} errores ({round((1-error_count/len(lines))*100, 2)}% éxito)")
        return data

    def _process_item(self, item: Dict) -> Optional[Dict]:
        try:
            images = []
            raw_images = item.get('images')
            if isinstance(raw_images, list):
                for img in raw_images:
                    if isinstance(img, dict) and 'large' in img:
                        img_url = img['large']
                        if isinstance(img_url, str) and img_url.strip():
                            images.append({'large': img_url.strip()})
            
            product_data = {
                "main_category": clean_string(item.get("main_category")),
                "title": clean_string(item.get("title")),
                "average_rating": item.get("average_rating"),
                "price": item.get("price"),
                "images": images[0] if images else {},
                "categories": item.get("categories"),
                "details": item.get("details", {})
            }
            
            return ProductModel(**product_data).model_dump()
            
        except ValidationError as e:
            logger.warning(f"Error de validación en ítem: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error inesperado procesando ítem: {str(e)}")
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
                    logger.warning(f"No se pudo leer la caché '{cache_file.name}': {str(e)}")
            
            file_data = self._load_single_file(raw_file)
            total_items += len(file_data)
            
            try:
                with cache_file.open('wb') as f:
                    pickle.dump(file_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                all_data.extend(file_data)
            except Exception as e:
                logger.error(f"Fallo al guardar la caché '{cache_file.name}': {str(e)}")
                
        logger.info(f"Carga finalizada: {len(all_data)} productos | Éxito: {round((1 - total_errors / max(1, total_items)) * 100, 2)}%")
        return all_data