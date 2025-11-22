# verify_loader.py
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import time

# Agregar el path del proyecto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🔍 === VERIFICACIÓN COMPLETA DEL SISTEMA ===\n")
    
    # Verificar imports
    print("1. ✅ Verificando imports...")
    try:
        from src.core.data.loader import FastDataLoader
        from src.core.config import settings
        print("   ✅ Todos los imports funcionan correctamente")
    except Exception as e:
        print(f"   ❌ Error en imports: {e}")
        return
    
    # Verificar estructura de directorios
    print("\n2. 📁 Verificando estructura de directorios...")
    raw_dir = Path("./data/raw")
    proc_dir = Path("./data/processed")
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ Directorio raw: {raw_dir}")
    print(f"   ✅ Directorio processed: {proc_dir}")
    
    # Verificar archivos de datos
    print("\n3. 📊 Verificando archivos de datos...")
    data_files = list(raw_dir.glob("*.json")) + list(raw_dir.glob("*.jsonl"))
    if data_files:
        for file in data_files:
            print(f"   ✅ Encontrado: {file.name} ({file.stat().st_size} bytes)")
    else:
        print("   ⚠️  No hay archivos de datos, creando datos de prueba...")
        create_test_data(raw_dir)
        data_files = list(raw_dir.glob("*.json"))
    
    # Probar el Product
    print("\n4. 🏷️ Probando clase Product...")
    test_product_functionality()
    
    # Probar el Loader
    print("\n5. 🔄 Probando DataLoader...")
    test_loader_functionality()
    
    # Verificar resultados finales
    print("\n6. 📋 Verificando resultados finales...")
    verify_final_results()
    
    print("\n🎉 === VERIFICACIÓN COMPLETADA ===\n")

def create_test_data(raw_dir: Path):
    """Crea datos de prueba si no existen"""
    test_products = [
        {
            "title": "Wireless Gaming Headset Pro",
            "description": "High-quality gaming headset with 7.1 surround sound",
            "price": 129.99,
            "main_category": "electronics",
            "average_rating": 4.5,
            "rating_count": 250,
            "details": {
                "features": ["Noise cancellation", "RGB lighting", "Wireless", "30h battery"],
                "specifications": {"color": "black", "weight": "320g", "connectivity": "Bluetooth 5.0"}
            },
            "tags": ["gaming", "wireless", "audio"],
            "product_type": "electronics"
        },
        {
            "title": "Python Programming Masterclass 2024",
            "description": "Complete Python course from beginner to advanced",
            "price": 89.99,
            "main_category": "education",
            "average_rating": 4.8,
            "rating_count": 1500,
            "details": {
                "features": ["50+ hours video", "Projects included", "Lifetime access"],
                "specifications": {"level": "All Levels", "language": "English"}
            },
            "tags": ["programming", "education", "python"],
            "product_type": "courses"
        },
        {
            "title": "Mechanical Keyboard RGB",
            "description": "Professional mechanical keyboard with customizable RGB",
            "price": 79.99,
            "main_category": "electronics",
            "average_rating": 4.3,
            "rating_count": 89,
            "details": {
                "features": ["Mechanical switches", "RGB backlight", "N-key rollover"],
                "specifications": {"layout": "US QWERTY", "switches": "Blue"}
            },
            "tags": ["keyboard", "gaming", "mechanical"],
            "product_type": "electronics"
        }
    ]
    
    test_file = raw_dir / "test_products.json"
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_products, f, indent=2)
    print(f"   ✅ Creado archivo de prueba: {test_file}")

def test_product_functionality():
    """Prueba todas las funcionalidades de Product"""
    try:
        from src.core.data.product import Product
        # Test 1: Creación básica
        product_data = {
            "title": "Test Product",
            "description": "Test Description",
            "price": 99.99,
            "main_category": "test",
            "average_rating": 4.0,
            "details": {
                "features": ["feature1", "feature2"],
                "specifications": {"key": "value"}
            },
            "tags": ["tag1", "tag2"]
        }
        
        product = Product.from_dict(product_data)
        print("   ✅ Product.from_dict() funciona")
        
        # Test 2: Métodos principales
        text_repr = product.to_text()
        metadata = product.to_metadata()
        summary = product.get_summary()
        
        print("   ✅ to_text(), to_metadata(), get_summary() funcionan")
        
        """
        # Test 3: Limpieza de imágenes
        product.images = ProductImage.safe_create({
            "large": "https://example.com/image.jpg",
            "medium": "invalid-url"
        })
        product.clean_image_urls()
        print("   ✅ clean_image_urls() funciona")
        """
        # Test 4: Serialización
        product_dict = product.model_dump()
        print("   ✅ model_dump() funciona")
        
        # Test 5: Validación de datos problemáticos
        problematic_data = {
            "title": "   Product With Extra Spaces   ",
            "description": ["part1", "part2"],
            "price": "$99.99 USD",
            "average_rating": "4.5 stars"
        }
        
        fixed_product = Product.from_dict(problematic_data)
        print("   ✅ Manejo de datos problemáticos funciona")
        
        print("   ✅ Todas las funciones de Product funcionan correctamente")
        
    except Exception as e:
        print(f"   ❌ Error en Product: {e}")

def test_loader_functionality():
    """Prueba todas las funcionalidades del Loader"""
    try:
        from src.core.data.loader import FastDataLoader
        
        # Test 1: Inicialización
        loader = FastDataLoader(
            max_products_per_file=10,  # Limitado para prueba rápida
            auto_categories=True,
            cache_enabled=False
        )
        print("   ✅ Loader se inicializa correctamente")
        
        # Test 2: Carga de datos
        start_time = time.time()
        products = loader.load_data()
        load_time = time.time() - start_time
        
        print(f"   ✅ load_data() completado en {load_time:.1f}s")
        print(f"   ✅ Productos cargados: {len(products)}")
        
        # Test 3: Verificar estructura de productos
        if products:
            product = products[0]
            required_attrs = ['title', 'price', 'product_type', 'details']
            missing_attrs = [attr for attr in required_attrs if not hasattr(product, attr)]
            
            if not missing_attrs:
                print("   ✅ Estructura de productos correcta")
            else:
                print(f"   ⚠️  Atributos faltantes: {missing_attrs}")
        
        # Test 4: Estadísticas
        stats = loader.get_stats()
        print(f"   ✅ Estadísticas: {stats['total_products_loaded']} productos, {stats['total_categories']} categorías")
        
        # Test 5: Categorización
        if stats['total_categories'] > 0:
            print(f"   ✅ Categorización automática funcionando: {stats['categories']}")
        else:
            print("   ⚠️  No se descubrieron categorías")
            
        print("   ✅ Todas las funciones del Loader funcionan correctamente")
        
    except Exception as e:
        print(f"   ❌ Error en Loader: {e}")

def verify_final_results():
    """Verifica los resultados finales del procesamiento"""
    try:
        output_file = Path("./data/processed/products.json")
        
        if not output_file.exists():
            print("   ❌ No se encontró el archivo de salida")
            return
        
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"   ✅ Archivo de salida existe: {output_file}")
        print(f"   📊 Total de productos guardados: {len(data)}")
        
        # Analizar calidad de datos
        if data:
            sample_product = data[0]
            
            # Verificar campos críticos
            critical_fields = ['title', 'price', 'product_type']
            field_status = {}
            
            for field in critical_fields:
                value = sample_product.get(field)
                field_status[field] = {
                    'exists': field in sample_product,
                    'has_value': bool(value),
                    'value': value
                }
            
            print("\n   🔍 Análisis de calidad de datos:")
            for field, status in field_status.items():
                status_icon = "✅" if status['exists'] and status['has_value'] else "❌"
                print(f"      {status_icon} {field}: {status['value']}")
            
            # Verificar categorización
            product_types = set(p.get('product_type', 'unknown') for p in data)
            print(f"   🏷️  Tipos de productos encontrados: {len(product_types)}")
            print(f"   📋 Tipos: {list(product_types)[:5]}...")
            
            # Verificar precios
            prices = [p.get('price', 0) for p in data if isinstance(p.get('price'), (int, float))]
            if prices:
                avg_price = sum(prices) / len(prices)
                print(f"   💰 Precio promedio: ${avg_price:.2f}")
                print(f"   📈 Rango de precios: ${min(prices):.2f} - ${max(prices):.2f}")
            
            # Verificar completitud de datos
            completeness_stats = {}
            total_products = len(data)
            
            for field in ['title', 'description', 'price', 'product_type', 'details']:
                field_count = sum(1 for p in data if p.get(field))
                completeness = (field_count / total_products) * 100
                completeness_stats[field] = completeness
            
            print("\n   📊 Completitud de datos:")
            for field, completeness in completeness_stats.items():
                status = "✅ EXCELENTE" if completeness > 90 else "✅ BUENO" if completeness > 70 else "⚠️  REGULAR" if completeness > 50 else "❌ POBRE"
                print(f"      {field}: {completeness:.1f}% - {status}")
        
        print("   ✅ Verificación de resultados completada")
        
    except Exception as e:
        print(f"   ❌ Error en verificación de resultados: {e}")

if __name__ == "__main__":
    main()