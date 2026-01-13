# src/data/loader.py
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_raw_products(
    file_path: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:

    if file_path:
        return _load_single_file(file_path, limit)
    else:
        return _load_all_files(limit)


def _load_single_file(
    file_path: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    products = []
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        logger.warning(f"  Archivo no encontrado: {file_path}")
        return []
    
    try:
        logger.info(f" Cargando: {file_path_obj.name}")
        
        with open(file_path_obj, 'r', encoding='utf-8') as f:
            lines_read = 0
            lines_valid = 0
            
            for line in f:
                if limit is not None and lines_valid >= limit:
                    break
                
                lines_read += 1
                
                try:
                    data = json.loads(line.strip())
                    
                    if not isinstance(data, dict):
                        continue
                    
                    title = data.get('title')
                    if not title:
                        continue
                    
                    if 'main_category' not in data:
                        data['main_category'] = data.get('categories', [''])[0] if data.get('categories') else ''
                    
                    if 'id' not in data:
                        data['id'] = f"{file_path_obj.stem}_{lines_read}"
                    
                    products.append(data)
                    lines_valid += 1
                    
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"Error procesando línea: {e}")
                    continue
        
        logger.info(f"  ✓ Líneas leídas: {lines_read}")
        logger.info(f"  ✓ Productos válidos: {lines_valid}")
        
        return products
        
    except Exception as e:
        logger.error(f"  ✗ Error cargando {file_path_obj.name}: {e}")
        return []


def _load_all_files(
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    products: List[Dict[str, Any]] = []
    data_dir = Path("data/raw")

    if not data_dir.exists():
        logger.warning("  Directorio data/raw no existe")
        data_dir.mkdir(parents=True, exist_ok=True)
        return []

    # 🔹 Descubrir archivos dinámicamente
    jsonl_files = sorted(data_dir.glob("*.jsonl"))

    if not jsonl_files:
        logger.warning("  No se encontraron archivos .jsonl en data/raw")
        return []

    logger.info(f" Procesando {len(jsonl_files)} archivos encontrados")

    for file_idx, file_path in enumerate(jsonl_files):
        if limit is not None and len(products) >= limit:
            logger.info(f"  Límite alcanzado: {limit} productos")
            break

        logger.info(f"\n Procesando ({file_idx + 1}/{len(jsonl_files)}): {file_path.name}")

        remaining = None
        if limit is not None:
            remaining = limit - len(products)
            if remaining <= 0:
                break

        try:
            file_products = _load_single_file(
                str(file_path),
                limit=remaining
            )
            products.extend(file_products)

            logger.info(f"  ✓ Productos de archivo: {len(file_products)}")
            logger.info(f"  ✓ Total acumulado: {len(products)}")

        except OSError as e:
            logger.error(f"  ✗ Error cargando {file_path.name}: {e}")
            continue

    logger.info("\n" + "=" * 60)
    logger.info(" CARGA COMPLETA")
    logger.info(f"   Total productos: {len(products)}")
    logger.info(f"   Archivos procesados: {len(jsonl_files)}")
    logger.info("=" * 60)

    return products
