# src/data/loader.py
"""
Data Loader para Amazon JSONL - Optimizado
"""
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_raw_products(limit: int = None) -> List[Dict[str, Any]]:
    """
    Carga TODOS los productos desde archivos JSONL de Amazon
    
    Args:
        limit: Si es None, carga todos los productos
    
    Returns:
        Lista de productos crudos
    """
    products = []
    data_dir = Path("data/raw")
    
    if not data_dir.exists():
        logger.warning("⚠️  Directorio data/raw no existe")
        # Crear directorio si no existe
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
    
    # Procesar TODOS los archivos
    for file_idx, file_path in enumerate(jsonl_files):
        if limit is not None and len(products) >= limit:
            logger.info(f"⏹️  Límite alcanzado: {limit} productos")
            break
            
        try:
            logger.info(f"\n📄 Procesando ({file_idx+1}/{len(jsonl_files)}): {file_path.name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                file_products = []
                lines_read = 0
                lines_valid = 0
                
                for line in f:
                    if limit is not None and len(products) >= limit:
                        break
                    
                    lines_read += 1
                    
                    try:
                        data = json.loads(line.strip())
                        
                        # Verificar que tenga datos mínimos
                        if not isinstance(data, dict):
                            continue
                        
                        # Asegurar que tenga título
                        title = data.get('title')
                        if not title:
                            continue
                        
                        # Normalizar datos básicos
                        if 'main_category' not in data:
                            data['main_category'] = data.get('categories', [''])[0] if data.get('categories') else ''
                        
                        # Añadir identificador único si no existe
                        if 'id' not in data:
                            data['id'] = f"{file_path.stem}_{lines_read}"
                        
                        # Añadir a productos
                        products.append(data)
                        file_products.append(data)
                        lines_valid += 1
                        
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.debug(f"Error procesando línea: {e}")
                        continue
            
            logger.info(f"  ✓ Líneas leídas: {lines_read}")
            logger.info(f"  ✓ Productos válidos: {len(file_products)}")
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
        
        logger.info(f"\n📊 DISTRIBUCIÓN POR CATEGORÍA:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(products)) * 100
            logger.info(f"  {cat:25s}: {count:6d} ({percentage:5.1f}%)")
    
    return products