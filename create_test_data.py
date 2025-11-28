# create_test_data.py
import json
from pathlib import Path
from datetime import datetime, timedelta
import random

def create_comprehensive_test_data():
    """Crea datos de prueba COMPLETOS para todas las evaluaciones"""
    
    # Productos gaming de ejemplo
    gaming_products = [
        {"id": "B09V3HN1KC", "title": "God of War Ragnarok PS5", "category": "games"},
        {"id": "B0BDJHN2GS", "title": "HyperX Cloud III Wireless", "category": "audio"},
        {"id": "B0C5N4VYF2", "title": "Logitech G PRO X Superlight 2", "category": "peripherals"},
        {"id": "B0B72K7H9N", "title": "Call of Duty: Modern Warfare III", "category": "games"},
        {"id": "B09B8RBFDD", "title": "Elden Ring", "category": "games"},
        {"id": "B0CHH3VYF2", "title": "Razer Huntsman V3 Pro", "category": "peripherals"},
        {"id": "B0C8HN2GS5", "title": "Samsung Odyssey G7", "category": "monitors"},
        {"id": "B0B5N4VYF3", "title": "PlayStation 5 Slim", "category": "consoles"},
        {"id": "B0BDJHN3HS", "title": "Xbox Series X", "category": "consoles"},
        {"id": "B0C5N4VYF4", "title": "Nintendo Switch OLED", "category": "consoles"}
    ]
    
    # Consultas de ejemplo
    test_queries = [
        "juegos de acción para ps5",
        "auriculares gaming inalámbricos",
        "teclado mecánico para gaming",
        "monitor 144hz gaming",
        "mejores juegos nintendo switch",
        "consola xbox series x",
        "ratón gaming ligero",
        "silla gamer ergonómica",
        "ssd para ps5",
        "juegos multiplayer pc"
    ]
    
    # Crear datos de feedback para RLHF
    feedback_data = []
    for i in range(20):  # Crear 20 ejemplos
        query = random.choice(test_queries)
        product = random.choice(gaming_products)
        
        feedback_data.append({
            "query": query,
            "response": f"Te recomiendo {product['title']}, excelente para gaming",
            "rating": random.randint(4, 5),  # Mayormente feedback positivo
            "selected_product_id": product['id'],
            "timestamp": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()
        })
    
    # Crear directorios
    Path("data/feedback").mkdir(parents=True, exist_ok=True)
    Path("data/users").mkdir(parents=True, exist_ok=True)
    
    # 1. Guardar datos de feedback para RLHF
    with open("data/feedback/success_queries.log", "w", encoding="utf-8") as f:
        for item in feedback_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # 2. Crear algunos usuarios de prueba
    users = []
    for i in range(10):
        users.append({
            "id": f"user_{i}",
            "name": f"Usuario {i}",
            "preferred_categories": ["games", "electronics", "gaming"],
            "search_history": random.sample(test_queries, 3)
        })
    
    with open("data/users/test_users.json", "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    
    # 3. Crear archivo de productos procesados básico si no existe
    products_file = Path("data/processed/products.json")
    if not products_file.exists():
        products_file.parent.mkdir(parents=True, exist_ok=True)
        with open(products_file, "w", encoding="utf-8") as f:
            json.dump(gaming_products, f, ensure_ascii=False, indent=2)
    
    print("✅ Datos de prueba COMPLETOS creados:")
    print(f"   - {len(feedback_data)} ejemplos de feedback para RLHF")
    print(f"   - {len(users)} usuarios de prueba")
    print(f"   - {len(gaming_products)} productos de ejemplo")

if __name__ == "__main__":
    create_comprehensive_test_data()