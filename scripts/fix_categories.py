# scripts/fix_categories.py
#!/usr/bin/env python3
"""Reparar categorías de productos - Versión Mejorada"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def fix_products_categories(products_file: Path, backup: bool = True):
    """Repara automáticamente las categorías de los productos."""
    print(f"\n🔧 REPARANDO CATEGORÍAS EN: {products_file}")
    
    if not products_file.exists():
        print(f"❌ Archivo no encontrado: {products_file}")
        return
    
    try:
        # Cargar productos
        with open(products_file, 'r', encoding='utf-8') as f:
            products = json.load(f)
        
        print(f"📦 Productos cargados: {len(products)}")
        
        # 🔥 USAR MÉTODO DE Product SI ESTÁ DISPONIBLE, SINO USAR DICCIONARIO LOCAL
        try:
            from src.core.data.product import Product
            use_product_method = True
            print("✅ Usando método de extracción de Product")
        except ImportError:
            use_product_method = False
            print("⚠️ Product no disponible, usando diccionario local")
            
            # Diccionario de respaldo
            category_keywords = {
                'Electronics': [
                    'iphone', 'samsung', 'android', 'smartphone', 'tablet', 'laptop', 'computer',
                    'pc', 'macbook', 'electronic', 'wireless', 'bluetooth', 'usb', 'cable',
                    'charger', 'battery', 'headphone', 'earphone', 'speaker', 'mouse', 'keyboard'
                ],
                'Video Games': [
                    'nintendo', 'playstation', 'xbox', 'switch', 'wii', 'gamecube', 'ps4', 'ps5',
                    'xbox one', 'game', 'video game', 'videogame', 'controller', 'console'
                ],
                'Books': [
                    'book', 'novel', 'author', 'edition', 'hardcover', 'paperback', 'kindle',
                    'literature', 'story', 'fiction', 'non-fiction'
                ],
                'Clothing': [
                    'shirt', 't-shirt', 'pants', 'jeans', 'dress', 'jacket', 'hoodie',
                    'sweater', 'shoe', 'sneaker', 'boot', 'hat', 'cap', 'belt', 'watch'
                ],
                'Home & Kitchen': [
                    'kitchen', 'home', 'furniture', 'appliance', 'cookware', 'pan', 'pot',
                    'knife', 'spoon', 'fork', 'plate', 'cup', 'glass', 'sofa', 'bed', 'chair'
                ],
                'Sports & Outdoors': [
                    'sport', 'fitness', 'gym', 'yoga', 'running', 'training', 'camping',
                    'hiking', 'bike', 'bicycle', 'ball', 'soccer', 'basketball', 'tennis'
                ],
                'Beauty': [
                    'beauty', 'makeup', 'cosmetic', 'skincare', 'perfume', 'lipstick',
                    'eyeliner', 'mascara', 'cream', 'lotion', 'shampoo', 'conditioner'
                ],
                'Toys & Games': [
                    'toy', 'lego', 'doll', 'action figure', 'puzzle', 'board game', 'card game',
                    'kids', 'children', 'educational', 'building', 'block'
                ],
                'Automotive': [
                    'car', 'auto', 'vehicle', 'tire', 'engine', 'motor', 'battery', 'oil',
                    'filter', 'tool', 'accessory', 'parts'
                ],
                'Office Products': [
                    'office', 'stationery', 'pen', 'pencil', 'notebook', 'paper', 'folder',
                    'printer', 'scanner', 'desk', 'chair', 'lamp'
                ]
            }
        
        fixed_count = 0
        
        for product in products:
            original_category = product.get('main_category', 'General')
            
            # 🔥 NO SOBREESCRIBIR CATEGORÍAS BUENAS EXISTENTES
            if original_category and original_category != 'General' and original_category != 'Other':
                continue
            
            # Extraer texto del producto
            title = product.get('title', '').lower()
            description = product.get('description', '').lower()
            combined_text = f"{title} {description}"
            
            new_category = None
            
            if use_product_method:
                # 🔥 USAR MÉTODO DE Product CON TÍTULO
                try:
                    new_category = Product._extract_category_from_title(product.get('title', ''))
                    if not new_category and description:
                        # Si no encontró en título, intentar con descripción
                        # (necesitaríamos un método _extract_category_from_description)
                        pass
                except Exception as e:
                    print(f"⚠️ Error usando método Product: {e}")
                    use_product_method = False  # Fallback a diccionario
            
            if not use_product_method or not new_category:
                # 🔥 USAR DICCIONARIO LOCAL CON SISTEMA DE PUNTUACIÓN
                best_category = None
                best_score = 0
                
                for category, keywords in category_keywords.items():
                    score = 0
                    for keyword in keywords:
                        if keyword in combined_text:
                            score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_category = category
                
                # Solo asignar si hay al menos 2 coincidencias
                if best_category and best_score >= 2:
                    new_category = best_category
                    print(f"   • Puntuación: {best_score} para '{best_category}'")
            
            # Asignar nueva categoría si se encontró una buena
            if new_category:
                product['main_category'] = new_category
                
                # 🔥 MANEJO SEGURO DE LISTA CATEGORIES
                if 'categories' not in product or not isinstance(product['categories'], list):
                    product['categories'] = []
                
                # Añadir categoría principal si no está
                if new_category not in product['categories']:
                    product['categories'].append(new_category)
                
                fixed_count += 1
                
                # Log detallado
                title_short = product.get('title', '')[:40]
                print(f"   • '{title_short}...' -> {original_category} -> {new_category}")
        
        # 🔥 BACKUP SEGURO
        if backup and fixed_count > 0:
            backup_file = products_file.with_suffix('.json.backup')
            import shutil
            shutil.copy2(products_file, backup_file)
            print(f"\n📦 Backup creado: {backup_file}")
        
        # Guardar productos reparados
        with open(products_file, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Categorías reparadas: {fixed_count}/{len(products)} productos")
        print(f"💾 Guardado en: {products_file}")
        
        if fixed_count == 0:
            print("💡 Todos los productos ya tenían categorías adecuadas")
        
    except Exception as e:
        print(f"❌ Error reparando categorías: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    # Obtener archivo de argumentos o usar predeterminado
    if len(sys.argv) > 1:
        products_file = Path(sys.argv[1])
    else:
        # Intentar importar settings
        try:
            from src.core.config import settings
            products_file = settings.PROC_DIR / "products.json"
        except ImportError:
            products_file = Path("data/processed/products.json")
    
    # Opción --no-backup
    backup = "--no-backup" not in sys.argv
    
    fix_products_categories(products_file, backup=backup)