import json
from pathlib import Path

def run_user_interface():
    category_file = Path.home() / "OneDrive" / "Documentos" / "Github" / "proyecto-ia-rag-rl-data" / "category_filters.json"

    if not category_file.exists():
        print("No se encontró el archivo de categorías. Ejecuta primero category_tree.")
        return

    with open(category_file, "r", encoding="utf-8") as f:
        category_map = json.load(f)

    while True:
        print("\nCategorías disponibles:")
        for i, cat in enumerate(category_map.keys()):
            print(f"{i + 1}. {cat}")
        
        try:
            choice = int(input("Selecciona una categoría por número: ")) - 1
            selected_category = list(category_map.keys())[choice]
        except (ValueError, IndexError):
            print("Selección inválida. Intenta de nuevo.")
            continue

        items = category_map[selected_category]
        print(f"\nMostrando 20 productos de la categoría: {selected_category}\n")

        for i, item in enumerate(items[:20], 1):
            print(f"{i}. {item['title']} - Precio: {item['price']} - Rating: {item['average_rating']}")

        respuesta = input("\n¿Te interesa alguno? (escribe el nombre exacto o 'otros' para ver otra categoría): ").strip().lower()
        if respuesta == "otros":
            continue
        else:
            print(f"\nHas seleccionado: {respuesta}")
            break
