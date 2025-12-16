#!/usr/bin/env python3
"""Reparar categorías de productos - Versión Mejorada y AUTÓNOMA"""

import json
import re
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

def _extract_category_from_title(title: str) -> str:
    """Extrae categoría del título usando palabras clave generales - FUNCIÓN AUTÓNOMA"""
    if not title:
        return "General"
    
    title_lower = title.lower()
    
    # 🔥 DICCIONARIO GENERALIZADO PARA E-COMMERCE (LOCAL, SIN DEPENDENCIAS)
    category_keywords = {
        'Electronics': [
            'laptop', 'computer', 'pc', 'macbook', 'notebook', 'desktop',
            'tablet', 'smartphone', 'phone', 'mobile', 'monitor', 'keyboard',
            'mouse', 'printer', 'scanner', 'camera', 'headphones', 'earphones',
            'speaker', 'tv', 'television', 'electronic', 'device', 'gadget',
            'usb', 'hdmi', 'cable', 'charger', 'battery', 'router', 'modem',
            'smartwatch', 'fitness tracker', 'drone', 'projector'
        ],
        'Clothing': [
            'shirt', 't-shirt', 'pants', 'jeans', 'dress', 'jacket', 'hoodie',
            'sweater', 'sweatshirt', 'shorts', 'skirt', 'blouse', 'coat',
            'underwear', 'socks', 'shoes', 'sneakers', 'boots', 'sandals',
            'hat', 'cap', 'gloves', 'scarf', 'belt', 'tie', 'suit', 'uniform'
        ],
        'Home & Kitchen': [
            'kitchen', 'cookware', 'appliance', 'furniture', 'sofa', 'bed',
            'chair', 'table', 'desk', 'lamp', 'light', 'rug', 'carpet',
            'curtain', 'blanket', 'pillow', 'mattress', 'cabinet', 'shelf'
        ],
        'Books': [
            'book', 'novel', 'author', 'edition', 'hardcover', 'paperback',
            'kindle', 'ebook', 'textbook', 'magazine', 'comic', 'biography'
        ],
        'Sports & Outdoors': [
            'fitness', 'exercise', 'gym', 'yoga', 'outdoor', 'camping',
            'hiking', 'running', 'training', 'bike', 'bicycle', 'ball',
            'soccer', 'basketball', 'tennis', 'golf', 'swimming', 'fishing'
        ],
        'Beauty': [
            'makeup', 'cosmetic', 'skincare', 'perfume', 'serum', 'lotion',
            'shampoo', 'conditioner', 'hair', 'nail', 'lipstick', 'mascara',
            'brush', 'mirror', 'cream', 'oil', 'soap', 'deodorant'
        ],
        'Toys & Games': [
            'toy', 'lego', 'puzzle', 'doll', 'kids', 'children', 'toddler',
            'action figure', 'board game', 'video game', 'game', 'console'
        ],
        'Automotive': [
            'car', 'auto', 'vehicle', 'engine', 'tire', 'motor', 'battery',
            'oil', 'filter', 'brake', 'light', 'tool', 'accessory', 'parts'
        ],
        'Office Products': [
            'office', 'stationery', 'paper', 'pen', 'pencil', 'notebook',
            'printer', 'scanner', 'desk', 'chair', 'lamp', 'folder', 'binder'
        ],
        'Health': [
            'vitamin', 'supplement', 'medicine', 'first aid', 'thermometer',
            'bandage', 'mask', 'sanitizer', 'pill', 'tablet', 'syrup'
        ],
        'Video Games': [
            'nintendo', 'playstation', 'xbox', 'switch', 'wii', 'gamecube',
            'ps4', 'ps5', 'xbox one', 'game', 'video game', 'videogame'
        ]
    }
    
    for category, keywords in category_keywords.items():
        if any(kw in title_lower for kw in keywords):
            return category
    
    return "General"

def _extract_category_from_description(description: str) -> str:
    """Extrae categoría de la descripción - FUNCIÓN AUTÓNOMA"""
    if not description:
        return "General"
    
    desc_lower = description.lower()
    
    category_keywords = {
        'Video Games': ['nintendo','playstation','xbox','switch','ps5','videogame','console'],
        'Electronics': ['iphone','samsung','android','tablet','laptop','pc','macbook'],
        'Books': ['book','novel','author','paperback','kindle','fiction'],
        'Clothing': ['shirt','jeans','dress','hoodie','apparel'],
        'Home & Kitchen': ['kitchen','cookware','appliance','furniture'],
        'Sports & Outdoors': ['fitness','gym','camping','running','training'],
        'Beauty': ['makeup','cosmetic','skincare','serum','hair'],
        'Toys & Games': ['toy','lego','board game','kids','children'],
        'Automotive': ['car','vehicle','engine','battery'],
        'Office Products': ['office','stationery','desk','supplies'],
        'Health': ['vitamin','supplement','medicine','first aid','thermometer']
    }

    scores = {
        cat: sum(1 for kw in words if kw in desc_lower)
        for cat, words in category_keywords.items()
    }

    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    
    return "General"

def fix_products_categories(products_file: Path, backup: bool = True, verbose: bool = False) -> int:
    """Repara automáticamente las categorías de los productos - VERSIÓN AUTÓNOMA."""
    print(f"\n🔧 REPARANDO CATEGORÍAS EN: {products_file}")
    
    if not products_file.exists():
        print(f"❌ Archivo no encontrado: {products_file}")
        return 0
    
    try:
        # Cargar productos
        with open(products_file, 'r', encoding='utf-8') as f:
            products = json.load(f)
        
        print(f"📦 Productos cargados: {len(products)}")
        
        fixed_count = 0
        
        for idx, product in enumerate(products):
            original_category = product.get('main_category', 'General')
            
            # 🔥 NO SOBREESCRIBIR CATEGORÍAS BUENAS EXISTENTES
            good_categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 
                              'Sports & Outdoors', 'Beauty', 'Toys & Games', 'Automotive',
                              'Office Products', 'Health', 'Video Games']
            
            if (original_category and 
                original_category in good_categories):
                if verbose and idx < 10:  # Solo mostrar primeros 10 para verbose
                    print(f"   • Manteniendo: '{original_category}' para '{product.get('title', '')[:30]}...'")
                continue
            
            # Extraer texto del producto
            title = product.get('title', '')
            description = product.get('description', '')
            
            new_category = None
            
            # Primero intentar con el título
            if title:
                new_category = _extract_category_from_title(title)
            
            # Si no encontró en el título, intentar con la descripción
            if not new_category or new_category == 'General':
                if description:
                    new_category = _extract_category_from_description(description)
            
            # Asignar nueva categoría si se encontró una buena
            if new_category and new_category != 'General':
                product['main_category'] = new_category
                
                # 🔥 MANEJO SEGURO DE LISTA CATEGORIES
                if 'categories' not in product or not isinstance(product['categories'], list):
                    product['categories'] = []
                
                # Añadir categoría principal si no está
                if new_category not in product['categories']:
                    product['categories'].append(new_category)
                
                fixed_count += 1
                
                # Log detallado solo si verbose o para primeros 20
                if verbose or fixed_count <= 20:
                    title_short = title[:40] if title else 'Sin título'
                    print(f"   • '{title_short}...' -> {original_category} -> {new_category}")
        
        # 🔥 BACKUP SEGURO
        if backup and fixed_count > 0:
            backup_file = products_file.with_suffix('.json.backup')
            try:
                shutil.copy2(products_file, backup_file)
                print(f"\n📦 Backup creado: {backup_file}")
            except Exception as e:
                print(f"⚠️  Error creando backup: {e}")
        
        # Guardar productos reparados
        with open(products_file, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Categorías reparadas: {fixed_count}/{len(products)} productos")
        print(f"💾 Guardado en: {products_file}")
        
        if fixed_count == 0:
            print("💡 Todos los productos ya tenían categorías adecuadas")
        
        return fixed_count
        
    except Exception as e:
        print(f"❌ Error reparando categorías: {e}")
        import traceback
        traceback.print_exc()
        return 0

def find_products_file() -> Optional[Path]:
    """Busca automáticamente el archivo products.json."""
    possible_paths = [
        Path("data/processed/products.json"),
        Path("../data/processed/products.json"),
        Path("../../data/processed/products.json"),
        Path("../../../data/processed/products.json"),
        Path.cwd() / "data" / "processed" / "products.json",
        Path.cwd().parent / "data" / "processed" / "products.json"
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✅ Encontrado: {path}")
            return path
    
    return None

def main():
    """Función principal con manejo de argumentos mejorado."""
    # Parsear argumentos manualmente
    args = sys.argv[1:]
    
    # Variables
    products_file = None
    verbose = False
    no_backup = False
    manual_file = None
    
    # Procesar argumentos
    i = 0
    while i < len(args):
        arg = args[i]
        
        if arg in ["-v", "--verbose"]:
            verbose = True
        elif arg in ["--no-backup"]:
            no_backup = True
        elif arg in ["-h", "--help"]:
            print("Uso: python fix_categories.py [ARCHIVO] [OPCIONES]")
            print("\nOpciones:")
            print("  -v, --verbose    Muestra detalles del proceso")
            print("  --no-backup      No crear backup del archivo original")
            print("  -h, --help       Muestra esta ayuda")
            print("\nEjemplos:")
            print("  python fix_categories.py")
            print("  python fix_categories.py -v")
            print("  python fix_categories.py data/processed/products.json --no-backup")
            return
        elif not arg.startswith("-"):
            # Debe ser el archivo
            manual_file = Path(arg)
            if not manual_file.exists():
                print(f"❌ Archivo no encontrado: {manual_file}")
                return
        i += 1
    
    # Determinar archivo
    if manual_file:
        products_file = manual_file
    else:
        print("🔍 Buscando archivo products.json...")
        products_file = find_products_file()
    
    if not products_file:
        print("❌ No se pudo encontrar archivo products.json")
        print("💡 Especifica la ruta manualmente:")
        print("   python fix_categories.py ruta/a/products.json")
        return
    
    # Ejecutar reparación
    fixed = fix_products_categories(
        products_file, 
        backup=not no_backup, 
        verbose=verbose
    )
    
    if fixed > 0:
        print("\n🎯 RECOMENDACIÓN: Ahora ejecuta 'python main.py index' para reconstruir el índice")
    else:
        print("\n💡 No se encontraron categorías para reparar")

if __name__ == "__main__":
    main()