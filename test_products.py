# test_products.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.data.product import Product, ProductImage, ProductDetails

def test_edge_cases():
    """Prueba casos extremos y datos problemáticos"""
    print("🧪 === PRUEBAS DE CASOS EXTREMOS ===\n")
    
    test_cases = [
        {
            "name": "Datos mínimos",
            "data": {"title": "Minimal Product"}
        },
        {
            "name": "Precios complejos", 
            "data": {
                "title": "Product with complex price",
                "price": "$99.99 USD + taxes",
                "description": "Test product"
            }
        },
        {
            "name": "Listas como strings",
            "data": {
                "title": "Product with string lists",
                "tags": "tag1, tag2, tag3",
                "categories": "cat1,cat2"
            }
        },
        {
            "name": "URLs de imágenes problemáticas",
            "data": {
                "title": "Product with bad images",
                "images": {
                    "large": "invalid-url",
                    "medium": "https://example.com/valid.jpg",
                    "small": None
                }
            }
        },
        {
            "name": "Rating fuera de rango",
            "data": {
                "title": "Product with bad rating",
                "average_rating": 10.5,  # Fuera de 0-5
                "rating_count": -5  # Negativo
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. Testing: {test_case['name']}")
        try:
            product = Product.from_dict(test_case['data'])
            print(f"   ✅ PROCESADO: {product.title}")
            print(f"      Precio: {getattr(product, 'price', 'N/A')}")
            print(f"      Rating: {getattr(product, 'average_rating', 'N/A')}")
            print(f"      Tipo: {getattr(product, 'product_type', 'N/A')}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
        print()

def test_product_methods():
    """Prueba todos los métodos de Product"""
    print("🔧 === PRUEBAS DE MÉTODOS ===\n")
    
    product_data = {
        "title": "Complete Test Product",
        "description": "This is a complete test product with all features",
        "price": 149.99,
        "main_category": "electronics",
        "average_rating": 4.7,
        "rating_count": 342,
        "details": {
            "features": ["Wireless", "Noise Cancelling", "Long Battery"],
            "specifications": {"color": "Black", "weight": "250g"}
        },
        "tags": ["premium", "wireless", "audio"],
        "product_type": "headphones"
    }
    
    product = Product.from_dict(product_data)
    
    methods_to_test = [
        ("to_text()", lambda: product.to_text()),
        ("to_metadata()", lambda: product.to_metadata()),
        ("get_summary()", lambda: product.get_summary()),
        ("model_dump()", lambda: product.model_dump()),
        ("clean_image_urls()", lambda: product.clean_image_urls()),
    ]
    
    for method_name, method_call in methods_to_test:
        try:
            result = method_call()
            print(f"✅ {method_name}: OK")
            if method_name == "get_summary()":
                print(f"   Resumen: {result}")
        except Exception as e:
            print(f"❌ {method_name}: ERROR - {e}")

if __name__ == "__main__":
    test_edge_cases()
    test_product_methods()
    print("\n🎉 Todas las pruebas completadas")