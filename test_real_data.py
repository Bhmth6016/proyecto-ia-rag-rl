# test_real_data.py
"""
Prueba con datos reales de productos.
"""

import json
from pathlib import Path

def create_test_data():
    """Crea archivos de prueba con datos reales simulados."""
    
    test_data = [
        {
            "asin": "B08N5WRWNW",
            "title": "Echo Dot (4th Gen) | Smart speaker with Alexa | Charcoal",
            "description": "Meet the all-new Echo Dot - Our most popular smart speaker with Alexa. The sleek, compact design delivers crisp vocals and balanced bass for full sound.",
            "price": "$49.99",
            "main_category": "Electronics",
            "brand": "Amazon",
            "categories": ["Smart Home", "Speakers"],
            "features": ["Alexa", "Voice control", "Smart home hub"]
        },
        {
            "asin": "B07XJ8C8F7", 
            "title": "Apple iPad Air (10.9-inch, Wi-Fi, 64GB) - Space Gray (4th Generation)",
            "description": "Stunning 10.9-inch Liquid Retina display with True Tone and P3 wide color. A14 Bionic chip with Neural Engine. USB-C connector for charging and accessories.",
            "price": "$599.00",
            "main_category": "Electronics",
            "brand": "Apple",
            "categories": ["Tablets", "Computers & Tablets"],
            "features": ["10.9-inch display", "A14 Bionic chip", "USB-C"]
        }
    ]
    
    # Guardar como JSONL
    output_file = Path("./data/raw/test_amazon.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ Datos de prueba creados en: {output_file}")
    return output_file

def test_with_real_data():
    """Prueba con los datos reales creados."""
    
    from src.core.data.loader import FastDataLoader
    
    # Crear datos de prueba
    test_file = create_test_data()
    
    # Configurar loader
    loader = FastDataLoader(
        raw_dir=test_file.parent,
        processed_dir=Path("./data/processed/test"),
        ml_enabled=True,
        ml_features=["category", "entities", "tags", "embedding"],
        use_progress_bar=True,
        max_products_per_file=10
    )
    
    print("\n🚀 Cargando datos con ML...")
    products = loader.load_data()
    
    print(f"\n✅ {len(products)} productos cargados")
    
    # Mostrar resultados
    for i, product in enumerate(products[:2]):
        print(f"\n📦 Producto {i+1}:")
        print(f"   Título: {product.title}")
        print(f"   Precio: ${product.price if product.price else 'N/A'}")
        
        if hasattr(product, 'predicted_category') and product.predicted_category:
            print(f"   📊 Categoría ML: {product.predicted_category}")
        
        if hasattr(product, 'ml_tags') and product.ml_tags:
            print(f"   🏷️  Tags ML: {product.ml_tags}")
        
        if hasattr(product, 'extracted_entities') and product.extracted_entities:
            print(f"   🔍 Entidades: {len(product.extracted_entities)} grupos")
        
        if hasattr(product, 'embedding') and product.embedding:
            print(f"   📐 Embedding: {len(product.embedding)} dimensiones")

if __name__ == "__main__":
    test_with_real_data()