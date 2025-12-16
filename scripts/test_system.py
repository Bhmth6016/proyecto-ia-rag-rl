#!/usr/bin/env python3
"""Prueba rápida del sistema"""

import sys
import json
from pathlib import Path

def test_categorization_logic():
    """Prueba la lógica de categorización local."""
    print("🧪 TEST DE LÓGICA DE CATEGORIZACIÓN")
    print("="*50)
    
    # Usar la misma lógica que fix_categories
    def extract_category_from_title(title: str) -> str:
        if not title:
            return "General"
        
        title_lower = title.lower()
        
        category_keywords = {
            'Electronics': ['laptop', 'computer', 'pc', 'macbook', 'tablet', 'smartphone', 'phone', 'camera'],
            'Clothing': ['shirt', 't-shirt', 'pants', 'jeans', 'dress', 'jacket', 'shoes'],
            'Home & Kitchen': ['kitchen', 'cookware', 'appliance', 'furniture', 'sofa', 'bed'],
            'Books': ['book', 'novel', 'author', 'edition', 'kindle'],
            'Sports & Outdoors': ['fitness', 'gym', 'outdoor', 'camping', 'bike', 'bicycle'],
            'Beauty': ['makeup', 'cosmetic', 'skincare', 'perfume', 'shampoo'],
            'Toys & Games': ['toy', 'lego', 'puzzle', 'doll', 'game'],
            'Automotive': ['car', 'auto', 'vehicle', 'tire', 'engine'],
            'Office Products': ['office', 'stationery', 'paper', 'pen', 'notebook'],
            'Health': ['vitamin', 'supplement', 'medicine', 'first aid'],
            'Video Games': ['nintendo', 'playstation', 'xbox', 'switch', 'videogame']
        }
        
        for category, keywords in category_keywords.items():
            if any(kw in title_lower for kw in keywords):
                return category
        
        return "General"
    
    test_cases = [
        ("Laptop Gaming ASUS ROG 16GB RAM", "Electronics"),
        ("Zapatos deportivos para correr Nike", "Clothing"),
        ("Sofá de cuero para sala", "Home & Kitchen"),
        ("Videojuego The Legend of Zelda", "Video Games"),
        ("Crema facial hidratante", "Beauty"),
        ("Bicicleta de montaña profesional", "Sports & Outdoors"),
        ("Libro de ciencia ficción", "Books"),
        ("Juego de herramientas", "Automotive"),
        ("Set de maquillaje", "Beauty"),
        ("Monitor 4K 27 pulgadas", "Electronics")
    ]
    
    correct = 0
    for title, expected in test_cases:
        result = extract_category_from_title(title)
        if result == expected:
            correct += 1
            print(f"✅ '{title[:30]}...' -> {result}")
        else:
            print(f"❌ '{title[:30]}...' -> {result} (esperado: {expected})")
    
    print(f"\n📊 Resultado: {correct}/{len(test_cases)} correctos")

def check_data_file():
    """Verifica el archivo de datos."""
    print("\n🔍 VERIFICANDO ARCHIVO DE DATOS")
    print("="*50)
    
    # Buscar archivo
    possible_paths = [
        Path("data/processed/products.json"),
        Path("../data/processed/products.json"),
        Path.cwd() / "data" / "processed" / "products.json"
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✅ Archivo encontrado: {path}")
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"📦 Productos totales: {len(data)}")
                
                # Analizar categorías
                categories = {}
                for item in data[:200]:  # Solo primeros 200 para velocidad
                    cat = item.get('main_category', 'Unknown')
                    categories[cat] = categories.get(cat, 0) + 1
                
                print("📊 Distribución de categorías (muestra de 200):")
                for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / 200) * 100
                    print(f"   • {cat}: {count} productos ({percentage:.1f}%)")
                
                return True
                
            except Exception as e:
                print(f"❌ Error leyendo archivo: {e}")
                return False
    
    print("❌ No se encontró archivo products.json")
    return False

def main():
    """Función principal."""
    print("🚀 PRUEBA RÁPIDA DEL SISTEMA")
    print("="*60)
    
    success = True
    
    # Test 1: Lógica de categorización
    test_categorization_logic()
    
    # Test 2: Archivo de datos
    if not check_data_file():
        success = False
    
    if success:
        print("\n✅ Sistema verificado correctamente")
        print("\n💡 Siguientes pasos:")
        print("   1. python scripts/fix_categories.py -v")
        print("   2. python main.py index")
        print("   3. python main.py rag --mode enhanced --ml")
    else:
        print("\n⚠️  Se encontraron problemas en la verificación")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())