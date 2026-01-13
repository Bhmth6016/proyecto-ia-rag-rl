# src/data/loader.py
"""
Data Loader para Amazon JSONL - Versión actualizada
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_raw_products(file_path: Optional[str] = None, limit: int = None) -> List[Dict[str, Any]]:
    """
    Carga productos desde archivos JSONL de Amazon
    
    Args:
        file_path: Ruta a un archivo específico (si es None, carga de todos)
        limit: Límite máximo de productos a cargar
    
    Returns:
        Lista de productos crudos
    """
    products = []
    
    if file_path:
        # Cargar desde un archivo específico
        return _load_single_file(file_path, limit)
    else:
        # Cargar desde todos los archivos
        return _load_all_files(limit)


def _load_single_file(file_path: str, limit: int = None) -> List[Dict[str, Any]]:
    """Carga productos desde un solo archivo"""
    products = []
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        logger.warning(f"⚠️  Archivo no encontrado: {file_path}")
        return []
    
    try:
        logger.info(f"📄 Cargando: {file_path_obj.name}")
        
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
                    
                    # Normalizar datos básicos
                    if 'main_category' not in data:
                        data['main_category'] = data.get('categories', [''])[0] if data.get('categories') else ''
                    
                    # Añadir identificador único si no existe
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


def _load_all_files(limit: int = None) -> List[Dict[str, Any]]:
    """Carga productos desde todos los archivos"""
    products = []
    data_dir = Path("data/raw")
    
    if not data_dir.exists():
        logger.warning("⚠️  Directorio data/raw no existe")
        data_dir.mkdir(parents=True, exist_ok=True)
        return []
    
    # Lista completa de archivos
    required_files = [
        "meta_Automotive_10000.jsonl",
        "meta_Beauty_and_Personal_Care_10000.jsonl", 
        "meta_Books_10000.jsonl",
        "meta_Clothing_Shoes_and_Jewelry_10000.jsonl",
        "meta_Electronics_10000.jsonl",
        "meta_Home_and_Kitchen_10000.jsonl",
        "meta_Sports_and_Outdoors_10000.jsonl",
        "meta_Toys_and_Games_10000.jsonl",
        "meta_Video_Games_10000.jsonl"
    ]
    
    # Buscar archivos existentes
    jsonl_files = []
    for filename in required_files:
        file_path = data_dir / filename
        if file_path.exists():
            jsonl_files.append(file_path)
        else:
            logger.warning(f"⚠️  Archivo no encontrado: {filename}")
    
    logger.info(f"📂 Procesando {len(jsonl_files)}/9 archivos")
    
    # Procesar archivos
    for file_idx, file_path in enumerate(jsonl_files):
        if limit is not None and len(products) >= limit:
            logger.info(f"⏹️  Límite alcanzado: {limit} productos")
            break
            
        try:
            logger.info(f"\n📄 Procesando ({file_idx+1}/{len(jsonl_files)}): {file_path.name}")
            
            # Calcular cuántos productos necesitamos de este archivo
            remaining = None
            if limit is not None:
                remaining = limit - len(products)
                if remaining <= 0:
                    break
            
            file_products = _load_single_file(str(file_path), limit=remaining)
            products.extend(file_products)
            
            logger.info(f"  ✓ Productos de archivo: {len(file_products)}")
            logger.info(f"  ✓ Total acumulado: {len(products)}")
            
            # Mostrar estadísticas del archivo
            if file_products:
                sample_size = min(3, len(file_products))
                logger.info(f"  📝 Ejemplos del archivo:")
                for i in range(sample_size):
                    product = file_products[i]
                    title = product.get('title', 'Sin título')
                    if len(title) > 40:
                        title = title[:37] + "..."
                    logger.info(f"    {i+1}. {title}")
                
        except Exception as e:
            logger.error(f"  ✗ Error cargando {file_path.name}: {e}")
            continue
    
    logger.info(f"\n" + "="*60)
    logger.info(f"✅ CARGA COMPLETA")
    logger.info(f"   Total productos: {len(products)}")
    logger.info(f"   Archivos procesados: {len(jsonl_files)}/9")
    logger.info("="*60)
    
    # Estadísticas completas
    if products:
        categories = {}
        for product in products:
            cat = product.get('main_category', 'Unknown')
            if isinstance(cat, str):
                categories[cat] = categories.get(cat, 0) + 1
            elif isinstance(cat, list) and cat:
                cat_str = str(cat[0]) if len(cat) > 0 else 'Unknown'
                categories[cat_str] = categories.get(cat_str, 0) + 1
        
        logger.info(f"\n📊 DISTRIBUCIÓN POR CATEGORÍA:")
        sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        for cat, count in sorted_cats[:20]:  # Mostrar top 20
            percentage = (count / len(products)) * 100
            logger.info(f"  {cat[:30]:30s}: {count:6d} ({percentage:5.1f}%)")
        
        if len(sorted_cats) > 20:
            logger.info(f"  ... y {len(sorted_cats) - 20} categorías más")
    
    return products