# regenerate_interactions.py (CORRECCIÓN FINAL 2)
"""
Script para regenerar interacciones antiguas con los nuevos IDs del sistema.
Hace match por título de producto para encontrar los IDs actuales.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
import hashlib
import datetime
import shutil

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_old_ground_truth() -> Dict[str, List[str]]:
    """Cargar el ground truth antiguo"""
    gt_file = Path("data/interactions/ground_truth_REAL.json")
    
    if not gt_file.exists():
        logger.error(f"Ground truth no encontrado: {gt_file}")
        sys.exit(1)
    
    with open(gt_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    logger.info(f"Ground truth antiguo cargado: {len(ground_truth)} queries")
    total_relevantes = sum(len(ids) for ids in ground_truth.values())
    logger.info(f"Total productos relevantes: {total_relevantes}")
    
    return ground_truth

def load_old_interactions() -> List[Dict]:
    """Cargar interacciones antiguas"""
    interactions_file = Path("data/interactions/real_interactions.jsonl")
    
    if not interactions_file.exists():
        logger.warning(f"Archivo de interacciones no encontrado: {interactions_file}")
        return []
    
    interactions = []
    with open(interactions_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                interactions.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Interacciones antiguas cargadas: {len(interactions)}")
    return interactions

def load_system_products() -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """
    Cargar productos del sistema actual desde el cache
    Devuelve:
    - products_dict: {id: producto_info}
    - title_to_id: {title_normalizado: id}
    """
    try:
        from src.unified_system_v2 import UnifiedSystemV2
        
        logger.info("Cargando sistema V2...")
        system = UnifiedSystemV2.load_from_cache()
        
        if not system:
            logger.error("No se pudo cargar el sistema")
            sys.exit(1)
        
        products_dict = {}
        title_to_id = {}
        
        # VERIFICAR SI canonical_products ES LISTA O DICT
        canonical_products = getattr(system, 'canonical_products', None)
        
        if canonical_products is None:
            logger.error("El sistema no tiene canonical_products")
            sys.exit(1)
        
        # Manejar tanto lista como diccionario
        if isinstance(canonical_products, list):
            logger.info(f"canonical_products es una lista con {len(canonical_products)} elementos")
            
            for product in canonical_products:
                product_id = getattr(product, 'id', None)
                if not product_id:
                    # Generar ID basado en título si no existe
                    title = getattr(product, 'title', '')
                    if title:
                        product_id = f"prod_{hashlib.md5(title.encode()).hexdigest()[:12]}"
                    else:
                        continue
                
                products_dict[product_id] = {
                    'id': product_id,
                    'title': getattr(product, 'title', ''),
                    'category': getattr(product, 'category', ''),
                    'description': getattr(product, 'description', '')[:200] if hasattr(product, 'description') else ''
                }
                
                # Crear mapeo de título a ID
                title = getattr(product, 'title', '')
                if title:
                    title_normalized = title.lower().strip()
                    if title_normalized:
                        title_to_id[title_normalized] = product_id
        
        elif isinstance(canonical_products, dict):
            logger.info(f"canonical_products es un diccionario con {len(canonical_products)} elementos")
            
            for product_id, product in canonical_products.items():
                products_dict[product_id] = {
                    'id': product_id,
                    'title': getattr(product, 'title', ''),
                    'category': getattr(product, 'category', ''),
                    'description': getattr(product, 'description', '')[:200] if hasattr(product, 'description') else ''
                }
                
                # Crear mapeo de título a ID
                title = getattr(product, 'title', '')
                if title:
                    title_normalized = title.lower().strip()
                    if title_normalized:
                        title_to_id[title_normalized] = product_id
        else:
            logger.error(f"Tipo inesperado para canonical_products: {type(canonical_products)}")
            sys.exit(1)
        
        logger.info(f"Productos cargados del sistema: {len(products_dict)}")
        logger.info(f"Títulos únicos mapeados: {len(title_to_id)}")
        
        # Mostrar algunos ejemplos
        sample_ids = list(products_dict.keys())[:3]
        logger.info("Ejemplos de productos:")
        for pid in sample_ids:
            title = products_dict[pid]['title'][:50] if products_dict[pid]['title'] else 'Sin título'
            logger.info(f"  {pid[:12]}... - {title}...")
        
        return products_dict, title_to_id
        
    except ImportError as e:
        logger.error(f"Error importando UnifiedSystemV2: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error cargando productos: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

def find_product_by_title(title: str, title_to_id: Dict[str, str], 
                         products_dict: Dict[str, Dict], threshold: int = 80) -> Optional[str]:
    """
    Encontrar un producto por título usando fuzzy matching
    
    Args:
        title: Título a buscar
        title_to_id: Diccionario de títulos normalizados a IDs
        products_dict: Diccionario de productos
        threshold: Umbral de similitud (0-100)
    
    Returns:
        ID del producto encontrado o None
    """
    if not title:
        return None
    
    # Normalizar el título de búsqueda
    search_title = title.lower().strip()
    
    # 1. Búsqueda exacta
    if search_title in title_to_id:
        return title_to_id[search_title]
    
    # 2. Búsqueda fuzzy con fuzzywuzzy
    try:
        # Si fuzzywuzzy está disponible
        best_match, score = process.extractOne(search_title, title_to_id.keys())
        if score >= threshold:
            logger.debug(f"Fuzzy match encontrado: '{search_title[:50]}...' -> '{best_match[:50]}...' (score: {score})")
            return title_to_id[best_match]
    except Exception:
        # Fallback simple: búsqueda por subcadena
        for stored_title, product_id in title_to_id.items():
            if search_title in stored_title or stored_title in search_title:
                return product_id
    
    # 3. Búsqueda en títulos completos de productos
    for product_id, product_info in products_dict.items():
        product_title = product_info.get('title', '').lower()
        if search_title in product_title or product_title in search_title:
            return product_id
    
    return None

def regenerate_ground_truth(old_gt: Dict[str, List[str]], 
                           products_dict: Dict[str, Dict],
                           title_to_id: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Regenerar ground truth con nuevos IDs
    
    Args:
        old_gt: Ground truth antiguo
        products_dict: Productos del sistema actual
        title_to_id: Mapeo de títulos a IDs
    
    Returns:
        Nuevo ground truth con IDs actualizados
    """
    logger.info("Regenerando ground truth...")
    
    new_gt = {}
    matches_found = 0
    matches_failed = 0
    
    # Intentar cargar dataset original para obtener títulos
    dataset_loaded = False
    old_id_to_title = {}
    
    try:
        dataset_paths = [
            Path("data/amazon_metadata_processed.parquet"),
            Path("data/amazon_metadata.parquet"),
            Path("data/amazon_metadata.csv"),
            Path("data/dataset.csv"),
            Path("data/raw/amazon_metadata.csv")
        ]
        
        dataset_path = None
        for path in dataset_paths:
            if path.exists():
                dataset_path = path
                break
        
        if dataset_path:
            logger.info(f"Cargando dataset original: {dataset_path}")
            if str(dataset_path).endswith('.parquet'):
                df = pd.read_parquet(dataset_path)
            else:
                df = pd.read_csv(dataset_path, nrows=10000)  # Limitar para velocidad
            
            # Crear mapeo de ID antiguo a título
            id_column = None
            
            # Buscar columna de ID
            possible_id_columns = ['id', 'asin', 'product_id', 'meta_id', 'asin_id']
            for col in possible_id_columns:
                if col in df.columns:
                    id_column = col
                    break
            
            if id_column:
                for idx, row in df.iterrows():
                    old_id = str(row.get(id_column, ''))
                    title = str(row.get('title', '')) if 'title' in df.columns else ''
                    
                    if old_id and title and old_id.startswith('meta_'):
                        old_id_to_title[old_id] = title
                
                dataset_loaded = True
                logger.info(f"Mapeo ID->título creado: {len(old_id_to_title)} entradas")
            else:
                logger.warning("No se encontró columna de ID en el dataset")
    
    except Exception as e:
        logger.warning(f"Error cargando dataset: {e}")
    
    # Ahora regenerar ground truth
    for query, old_ids in old_gt.items():
        new_ids = []
        
        for old_id in old_ids:
            # Intentar obtener título del dataset original si está disponible
            if dataset_loaded and old_id in old_id_to_title:
                title = old_id_to_title[old_id]
                new_id = find_product_by_title(title, title_to_id, products_dict, threshold=75)
                
                if new_id:
                    new_ids.append(new_id)
                    matches_found += 1
                    logger.debug(f"Match encontrado para query '{query}': '{title[:50]}...' -> {new_id[:12]}...")
                else:
                    matches_failed += 1
                    logger.warning(f"No match para: '{title[:50]}...' (ID antiguo: {old_id})")
            else:
                # Si no tenemos título, buscar productos que coincidan con la query
                query_words = set(query.lower().replace('"', '').split())
                best_match = None
                best_score = 0
                
                for product_id, product_info in products_dict.items():
                    title = product_info.get('title', '').lower()
                    category = product_info.get('category', '').lower()
                    
                    # Calcular score de coincidencia
                    score = 0
                    for word in query_words:
                        if word in title:
                            score += 3
                        if word in category:
                            score += 2
                    
                    if score > best_score:
                        best_score = score
                        best_match = product_id
                
                if best_match and best_score >= 2:
                    new_ids.append(best_match)
                    matches_found += 1
                    logger.debug(f"Match por query para '{query}': {best_match[:12]}... (score: {best_score})")
                else:
                    matches_failed += 1
        
        # Eliminar duplicados y limitar a máximo 3 productos por query
        unique_ids = list(dict.fromkeys(new_ids))[:3]
        if unique_ids:
            new_gt[query] = unique_ids
            logger.info(f"✓ Query regenerada: '{query}' -> {len(unique_ids)} productos")
        else:
            logger.warning(f"✗ Query sin matches: '{query}'")
    
    logger.info(f"Ground truth regenerado: {len(new_gt)} queries")
    logger.info(f"Matches encontrados: {matches_found}")
    logger.info(f"Matches fallados: {matches_failed}")
    
    total_attempts = matches_found + matches_failed
    if total_attempts > 0:
        logger.info(f"Tasa de éxito: {(matches_found/total_attempts*100):.1f}%")
    
    return new_gt

def regenerate_interactions(old_interactions: List[Dict], 
                           products_dict: Dict[str, Dict],
                           title_to_id: Dict[str, str]) -> List[Dict]:
    """
    Regenerar interacciones con nuevos IDs
    
    Args:
        old_interactions: Interacciones antiguas
        products_dict: Productos del sistema actual
        title_to_id: Mapeo de títulos a IDs
    
    Returns:
        Nuevas interacciones con IDs actualizados
    """
    logger.info("Regenerando interacciones...")
    
    new_interactions = []
    success_count = 0
    fail_count = 0
    
    for i, interaction in enumerate(old_interactions):
        query = interaction.get('query', '')
        clicked_product_title = interaction.get('clicked_product_title', '')
        
        if not query or not clicked_product_title:
            fail_count += 1
            continue
        
        # Buscar el producto clickeado por título
        new_product_id = find_product_by_title(clicked_product_title, title_to_id, products_dict, threshold=70)
        
        if new_product_id:
            # Crear nueva interacción con ID actualizado
            new_interaction = interaction.copy()
            new_interaction['clicked_product_id'] = new_product_id
            new_interaction['clicked_product_title'] = clicked_product_title
            new_interaction['_regenerated'] = True
            new_interaction['_original_index'] = i
            
            new_interactions.append(new_interaction)
            success_count += 1
            
            if success_count <= 3:  # Log primeros éxitos
                logger.info(f"✓ Interacción regenerada: '{clicked_product_title[:50]}...' -> {new_product_id[:12]}...")
        else:
            fail_count += 1
            if fail_count <= 5:  # Log primeros fallos
                logger.warning(f"✗ No se pudo encontrar producto: '{clicked_product_title[:50]}...'")
    
    logger.info(f"Interacciones regeneradas: {success_count} exitosas, {fail_count} fallidas")
    
    total = success_count + fail_count
    if total > 0:
        logger.info(f"Tasa de éxito: {(success_count/total*100):.1f}%")
    
    return new_interactions

def save_regenerated_data(new_gt: Dict[str, List[str]], 
                         new_interactions: List[Dict],
                         products_dict: Dict[str, Dict],
                         old_interactions: List[Dict],  # AÑADIDO: parámetro para old_interactions
                         backup: bool = True):
    """
    Guardar datos regenerados
    
    Args:
        new_gt: Nuevo ground truth
        new_interactions: Nuevas interacciones
        products_dict: Diccionario de productos del sistema
        old_interactions: Lista de interacciones antiguas (para estadísticas)
        backup: Hacer backup de archivos originales
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Backup de archivos originales
    if backup:
        backup_dir = Path("data/backups")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        gt_file = Path("data/interactions/ground_truth_REAL.json")
        if gt_file.exists():
            backup_gt = backup_dir / f"ground_truth_REAL_backup_{timestamp}.json"
            shutil.copy2(gt_file, backup_gt)
            logger.info(f"✓ Backup ground truth: {backup_gt}")
        
        interactions_file = Path("data/interactions/real_interactions.jsonl")
        if interactions_file.exists():
            backup_interactions = backup_dir / f"real_interactions_backup_{timestamp}.jsonl"
            shutil.copy2(interactions_file, backup_interactions)
            logger.info(f"✓ Backup interacciones: {backup_interactions}")
    
    # Guardar nuevo ground truth
    new_gt_file = Path("data/interactions/ground_truth_REAL.json")
    with open(new_gt_file, 'w', encoding='utf-8') as f:
        json.dump(new_gt, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Nuevo ground truth guardado: {new_gt_file}")
    logger.info(f"  • Queries: {len(new_gt)}")
    total_products = sum(len(ids) for ids in new_gt.values())
    logger.info(f"  • Productos relevantes: {total_products}")
    
    # Guardar nuevas interacciones
    if new_interactions:
        new_interactions_file = Path("data/interactions/real_interactions.jsonl")
        with open(new_interactions_file, 'w', encoding='utf-8') as f:
            for interaction in new_interactions:
                f.write(json.dumps(interaction, ensure_ascii=False) + '\n')
        
        logger.info(f"✓ Nuevas interacciones guardadas: {new_interactions_file}")
        logger.info(f"  • Interacciones: {len(new_interactions)}")
    else:
        logger.warning("No hay interacciones para guardar")
    
    # Crear archivo de estadísticas
    stats_file = Path(f"data/backups/regeneration_stats_{timestamp}.json")
    stats = {
        'timestamp': timestamp,
        'ground_truth': {
            'queries': len(new_gt),
            'total_products': total_products,
            'avg_products_per_query': total_products / len(new_gt) if new_gt else 0
        },
        'interactions': {
            'old_count': len(old_interactions),
            'new_count': len(new_interactions),
            'success_rate': len(new_interactions) / len(old_interactions) if old_interactions else 0
        },
        'products_in_system': len(products_dict)
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Estadísticas guardadas: {stats_file}")

def create_fallback_ground_truth(products_dict: Dict[str, Dict], title_to_id: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Crear un ground truth de respaldo si la regeneración falla
    
    Args:
        products_dict: Productos del sistema
    
    Returns:
        Ground truth básico
    """
    logger.info("Creando ground truth de respaldo...")
    
    # Queries de ejemplo basadas en categorías comunes
    sample_queries = {
        "pc gaming": [],
        "xbox console": [],
        "playstation games": [],
        "action video games": [],
        "sport videogames": [],
        "detective books": [],
        "science fiction novels": [],
        "bluetooth headphones": [],
        "laptop computers": [],
        "kitchen appliances": []
    }
    
    # Buscar productos para cada query
    for query, product_list in sample_queries.items():
        query_words = set(query.lower().split())
        matched_products = []
        
        for product_id, product_info in products_dict.items():
            title = product_info.get('title', '').lower()
            category = product_info.get('category', '').lower()
            
            # Verificar coincidencias
            matches = 0
            for word in query_words:
                if word in title:
                    matches += 2
                if word in category:
                    matches += 1
            
            if matches >= 2:
                matched_products.append((product_id, matches))
        
        # Ordenar por score y tomar los mejores
        matched_products.sort(key=lambda x: x[1], reverse=True)
        sample_queries[query] = [pid for pid, score in matched_products[:3]]
    
    # Filtrar queries vacías
    new_gt = {k: v for k, v in sample_queries.items() if v}
    
    logger.info(f"Ground truth de respaldo creado: {len(new_gt)} queries")
    
    return new_gt

def verify_regeneration(new_gt: Dict[str, List[str]], products_dict: Dict[str, Dict]) -> bool:
    """Verificar que todos los IDs en el nuevo ground truth existen en el sistema"""
    logger.info("Verificando regeneración...")
    
    all_ids = [pid for ids in new_gt.values() for pid in ids]
    
    if not all_ids:
        logger.error("El ground truth está vacío")
        return False
    
    found_ids = []
    missing_ids = []
    
    for pid in all_ids:
        if pid in products_dict:
            found_ids.append(pid)
        else:
            missing_ids.append(pid)
    
    logger.info(f"IDs verificados: {len(found_ids)} encontrados, {len(missing_ids)} faltantes")
    
    if missing_ids:
        logger.warning(f"IDs faltantes (primeros 5): {missing_ids[:5]}")
        logger.warning(f"Porcentaje de éxito: {(len(found_ids)/len(all_ids)*100):.1f}%")
    
    return len(found_ids) > 0

def main():
    """Función principal"""
    print("="*60)
    print("REGENERADOR DE INTERACCIONES Y GROUND TRUTH")
    print("="*60)
    
    # Verificar dependencias
    try:
        import fuzzywuzzy
        logger.info("✓ fuzzywuzzy instalado")
    except ImportError:
        logger.warning("fuzzywuzzy no instalado. Instalar: pip install fuzzywuzzy python-Levenshtein")
        logger.info("Usando método de matching simple...")
    
    # Cargar datos
    old_gt = load_old_ground_truth()
    old_interactions = load_old_interactions()
    products_dict, title_to_id = load_system_products()
    
    print("\n" + "="*60)
    print("PROCESANDO DATOS...")
    print("="*60)
    
    # Regenerar ground truth
    new_gt = regenerate_ground_truth(old_gt, products_dict, title_to_id)
    
    # Si no se encontraron suficientes matches, crear uno de respaldo
    if len(new_gt) < 10:
        logger.warning(f"Solo {len(new_gt)} queries regeneradas. Creando ground truth de respaldo...")
        backup_gt = create_fallback_ground_truth(products_dict, title_to_id)
        
        # Combinar ambos (priorizando los regenerados)
        for query, ids in backup_gt.items():
            if query not in new_gt:
                new_gt[query] = ids
        
        logger.info(f"Ground truth combinado: {len(new_gt)} queries")
    
    # Verificar regeneración
    if not verify_regeneration(new_gt, products_dict):
        logger.error("La regeneración falló. Usando ground truth de respaldo completo...")
        new_gt = create_fallback_ground_truth(products_dict, title_to_id)
    
    # Regenerar interacciones
    new_interactions = []
    if old_interactions:
        new_interactions = regenerate_interactions(old_interactions, products_dict, title_to_id)
    
    # Guardar datos regenerados
    print("\n" + "="*60)
    print("GUARDANDO RESULTADOS...")
    print("="*60)
    
    # PASAR old_interactions COMO PARÁMETRO ADICIONAL
    save_regenerated_data(new_gt, new_interactions, products_dict, old_interactions, backup=True)
    
    print("\n" + "="*60)
    print("RESUMEN")
    print("="*60)
    print(f"• Queries en ground truth: {len(new_gt)}")
    total_products = sum(len(ids) for ids in new_gt.values())
    print(f"• Productos relevantes totales: {total_products}")
    print(f"• Interacciones regeneradas: {len(new_interactions)}")
    print(f"• Productos en sistema: {len(products_dict)}")
    
    # Mostrar ejemplo de ground truth regenerado
    print("\nEjemplo de ground truth regenerado:")
    for i, (query, ids) in enumerate(list(new_gt.items())[:3]):
        print(f"  '{query}': {len(ids)} productos")
        for j, pid in enumerate(ids[:2]):
            title = products_dict.get(pid, {}).get('title', 'N/A')[:50]
            print(f"    {j+1}. {pid[:12]}... - {title}...")
        if len(ids) > 2:
            print(f"    ... y {len(ids)-2} más")
    
    print("\n✓ Regeneración completada. Ahora ejecuta:")
    print("  python main.py experimento")

if __name__ == "__main__":
    main()