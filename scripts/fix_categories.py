# scripts/fix_categories.py
import json
import logging
from pathlib import Path
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def fix_products_categories(products_file: Path):
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
        
        # Diccionario de categorías mejorado
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
            'Clothing & Accessories': [
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
            
            # Si ya tiene una categoría buena, mantenerla
            if original_category and original_category != 'General' and original_category != 'Other':
                continue
            
            # Extraer texto del producto
            title = product.get('title', '').lower()
            description = product.get('description', '').lower()
            combined_text = f"{title} {description}"
            
            # Buscar categoría por palabras clave
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
            
            # Si encontramos categoría con al menos 1 coincidencia
            if best_category and best_score >= 1:
                product['main_category'] = best_category
                
                # Asegurar que categories sea una lista
                if 'categories' not in product or not isinstance(product['categories'], list):
                    product['categories'] = []
                
                # Añadir categoría principal si no está
                if best_category not in product['categories']:
                    product['categories'].append(best_category)
                
                fixed_count += 1
        
        # Guardar productos reparados
        with open(products_file, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Categorías reparadas: {fixed_count}/{len(products)} productos")
        print(f"💾 Guardado en: {products_file}")
        
    except Exception as e:
        print(f"❌ Error reparando categorías: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        products_file = Path(sys.argv[1])
    else:
        products_file = Path("data/processed/products.json")
    
    fix_products_categories(products_file)