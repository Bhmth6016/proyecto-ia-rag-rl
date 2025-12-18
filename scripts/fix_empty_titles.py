#!/usr/bin/env python3
# scripts/fix_empty_titles.py

"""
Script para reparar productos con títulos vacíos en el dataset.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_empty_titles_in_file(input_file: Path, output_file: Path = None):
    """
    Repara títulos vacíos en un archivo de productos.
    
    Args:
        input_file: Archivo JSON de entrada
        output_file: Archivo de salida (opcional, sobrescribe input si es None)
    """
    if not input_file.exists():
        logger.error(f"❌ Archivo no encontrado: {input_file}")
        return
    
    if output_file is None:
        output_file = input_file
    
    logger.info(f"🔧 Reparando títulos vacíos en: {input_file}")
    
    try:
        # Cargar productos
        with open(input_file, 'r', encoding='utf-8') as f:
            products = json.load(f)
    except Exception as e:
        logger.error(f"❌ Error cargando archivo: {e}")
        return
    
    if not isinstance(products, list):
        logger.error("❌ El archivo debe contener una lista de productos")
        return
    
    total_fixed = 0
    total_products = len(products)
    
    for i, product in enumerate(products):
        if not isinstance(product, dict):
            continue
        
        # Verificar si el título está vacío o es inválido
        title = product.get('title', '')
        needs_fix = (
            not title or 
            not isinstance(title, str) or 
            not title.strip() or
            title == 'Unknown Product' or
            len(title.strip()) < 1
        )
        
        if needs_fix:
            # Generar título automáticamente
            new_title = generate_title_for_product(product)
            
            if new_title and new_title.strip():
                old_title = product.get('title', '')
                product['title'] = new_title
                product['title_fixed'] = True
                product['original_title'] = old_title
                total_fixed += 1
                
                if i % 50 == 0:
                    logger.info(f"🔄 Producto {i}: '{old_title}' → '{new_title[:50]}...'")
    
    # Guardar productos reparados
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"❌ Error guardando archivo: {e}")
        return
    
    logger.info(f"✅ Reparación completada!")
    logger.info(f"📊 Estadísticas:")
    logger.info(f"   • Productos totales: {total_products}")
    logger.info(f"   • Títulos reparados: {total_fixed}")
    logger.info(f"   • Porcentaje: {total_fixed/max(1, total_products)*100:.1f}%")
    logger.info(f"   • Guardado en: {output_file}")

def generate_title_for_product(product: Dict[str, Any]) -> str:
    """Genera título automático para un producto."""
    # Prioridad de fuentes para generar título
    sources = []
    
    # 1. Categoría principal
    if product.get('main_category') and product['main_category'] != 'General':
        cat_map = {
            'Electronics': 'Producto Electrónico',
            'Books': 'Libro',
            'Clothing': 'Prenda de Ropa',
            'Home & Kitchen': 'Artículo para el Hogar',
            'Sports & Outdoors': 'Equipo Deportivo',
            'Beauty': 'Producto de Belleza',
            'Toys & Games': 'Juguete',
            'Automotive': 'Producto Automotriz',
            'Office Products': 'Artículo de Oficina',
            'Video Games': 'Videojuego',
            'Health': 'Producto para la Salud'
        }
        category = product['main_category']
        readable_cat = cat_map.get(category, f"Producto de {category}")
        sources.append(readable_cat)
    
    # 2. Tipo de producto
    if product.get('product_type'):
        sources.append(product['product_type'])
    
    # 3. Marca
    if product.get('brand'):
        sources.append(f"{product['brand']}")
    
    # 4. Descripción (extraer primeras palabras)
    if product.get('description'):
        desc = str(product['description'])
        # Extraer palabras significativas
        words = desc.split()[:4]
        if len(words) >= 2:
            keyword_title = " ".join(words).capitalize()
            sources.append(keyword_title)
    
    # 5. Características/features
    if product.get('features') and isinstance(product['features'], list):
        features = product['features'][:2]
        if features:
            features_title = " ".join(features[:2]).capitalize()
            sources.append(features_title)
    
    # Seleccionar el mejor título
    if sources:
        # Priorizar títulos más cortos y descriptivos
        best_title = min(sources, key=lambda x: (len(x), -len(x.split())))
        
        # Capitalizar correctamente
        words = best_title.split()
        if len(words) > 0:
            words[0] = words[0].capitalize()
            best_title = " ".join(words)
        
        return best_title[:150]
    
    # Título por defecto
    return "Producto sin nombre"

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Repara productos con títulos vacíos en el dataset"
    )
    
    parser.add_argument(
        'input_file',
        type=Path,
        help='Archivo JSON de entrada con productos'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Archivo de salida (opcional, sobrescribe input por defecto)'
    )
    
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Crear copia de respaldo antes de modificar'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Verificar que el archivo existe
    if not args.input_file.exists():
        print(f"❌ Error: El archivo {args.input_file} no existe")
        sys.exit(1)
    
    # Crear copia de respaldo
    if args.backup:
        import shutil
        import time
        backup_file = args.input_file.parent / f"{args.input_file.stem}_backup_{time.strftime('%Y%m%d_%H%M%S')}{args.input_file.suffix}"
        try:
            shutil.copy2(args.input_file, backup_file)
            print(f"📋 Copia de respaldo creada: {backup_file}")
        except Exception as e:
            print(f"⚠️ No se pudo crear copia de respaldo: {e}")
    
    # Ejecutar reparación
    fix_empty_titles_in_file(args.input_file, args.output)
    
    print("\n" + "="*60)
    print("✅ REPARACIÓN COMPLETADA")
    print("="*60)
    print(f"📄 Archivo procesado: {args.input_file}")
    if args.output:
        print(f"💾 Archivo de salida: {args.output}")

if __name__ == "__main__":
    main()