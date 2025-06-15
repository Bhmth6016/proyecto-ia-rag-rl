# ui_interface.py
import pickle

from src.category_selector.category_tree import load_category_tree

MAX_PRODUCTS = 20

# ======================
# FUNCIONES UTILITARIAS
# ======================
def format_price(price) -> str:
    """Formatea el precio como string con $ o devuelve N/A"""
    if price is None:
        return 'N/A'
    try:
        return f"${float(price):.2f}"
    except (ValueError, TypeError):
        return str(price)

def safe_join(items, default: str = 'N/A') -> str:
    """Convierte una lista a string separado por comas de forma segura"""
    if not items:
        return default
    try:
        return ', '.join(str(item) for item in items if item is not None)
    except Exception:
        return default
    
def get_float_input(prompt: str, default: float, min_val: float = None, max_val: float = None) -> float:
    while True:
        try:
            value = input(prompt)
            num = float(value) if value else default
            if (min_val is not None and num < min_val) or (max_val is not None and num > max_val):
                print(f"¡Error! Ingrese un valor entre {min_val} y {max_val}.")
                continue
            return num
        except ValueError:
            print("¡Error! Ingrese un número válido.")
            
def show_details(details: dict) -> None:
    """Muestra los detalles adicionales del producto"""
    if not details:
        return
    print("\nDetalles adicionales:")
    for k, v in details.items():
        print(f"- {k}: {v if v is not None else 'N/A'}")
                   
def show_categories(category_tree):
    print("\nCategorías disponibles:")
    for i, category in enumerate(category_tree.keys()):
        print(f"{i + 1}. {category}")
    choice = input("Seleccione una categoría por número: ")
    try:
        choice_index = int(choice) - 1
        category = list(category_tree.keys())[choice_index]
        return category
    except (ValueError, IndexError):
        print("Selección inválida.")
        return None

def show_filters(filters):
    print("\nFiltros disponibles:")
    print(f"1. Precio: {filters['price_range'] if filters['price_range'] else 'No disponible'}")
    print(f"2. Rating promedio: {filters['average_rating']}")
    print(f"3. Etiquetas de categoría: {filters['category_tags']}")
    
    # Opcionalmente podrías permitir al usuario elegir aplicar filtros aquí
    return filters
def apply_filters(products, filters):
    filtered = []
    
    for product in products:
        # 1. Filtro por rating
        if "average_rating" in filters and filters["average_rating"]:
            rating = product.get("average_rating")
            if rating is None or rating not in filters["average_rating"]:
                continue
        
        # 2. Filtro por rango de precio (solo si hay datos)
        if "price_range" in filters and len(filters["price_range"]) == 2:
            price = product.get("price")
            min_price, max_price = filters["price_range"]
            if price is None or not (min_price <= price <= max_price):
                continue
        
        # 3. Filtro por categorías (solo si hay etiquetas)
        if "category_tags" in filters and filters["category_tags"]:
            product_categories = product.get("categories", []) or []
            if not any(cat in filters["category_tags"] for cat in product_categories):
                continue
        
        filtered.append(product)
    
    return filtered[:MAX_PRODUCTS]  # Asegura el límite máximo# Mantén el límite de productos

def show_products(products):
    print("\nProductos encontrados:")
    for i, product in enumerate(products, 1):
        title = product.get("title", "Sin título") or "Sin título"
        price = product.get("price", "N/A") if product.get("price") is not None else "N/A"
        rating = product.get("average_rating", "N/A") if product.get("average_rating") is not None else "N/A"
        print(f"{i}. {title} - Precio: {price} - Rating: {rating}")
    
    print(f"{len(products)+1}. Ver otros resultados")
    print(f"{len(products)+2}. Salir")
    
    while True:
        choice = input("\nSeleccione una opción: ")
        try:
            choice_index = int(choice)
            if 1 <= choice_index <= len(products):
                return {"action": "select", "product": products[choice_index-1]}
            elif choice_index == len(products)+1:
                return {"action": "retry"}
            elif choice_index == len(products)+2:
                return {"action": "exit"}
        except ValueError:
            print("Opción inválida. Intente nuevamente.")
            
            
def run_interface():
    category_tree = load_category_tree()
    
    while True:
        # Paso 1: Selección de categoría
        category = show_categories(category_tree)
        if category is None:
            continue

        file_path = category_tree[category]["file_path"]
        filters = category_tree[category]["filters"]

        # Paso 2: Preguntar si quiere aplicar filtros
        apply_filters_choice = input("\n¿Desea aplicar filtros? (s/n): ").lower()
        user_filters = {}

        if apply_filters_choice == 's':
            print("\nFiltros disponibles para esta categoría:")
            show_filters(filters)
            
            # Opción para seleccionar qué filtros aplicar
            filter_choice = input("\nIngrese los números de filtros a aplicar (ej: 1,3): ")
            selected_indices = [int(idx.strip()) for idx in filter_choice.split(',') if idx.strip().isdigit()]

            # Construir user_filters basado en selección
            if 1 in selected_indices and filters["price_range"]:
                min_price = float(input(f"Ingrese precio mínimo ({filters['price_range'][0]}): ") or filters["price_range"][0])
                max_price = float(input(f"Ingrese precio máximo ({filters['price_range'][1]}): ") or filters["price_range"][1])
                user_filters["price_range"] = [min_price, max_price]

            if 2 in selected_indices and filters["average_rating"]:
                print("Ratings disponibles:", filters["average_rating"])
                selected_ratings = input("Ingrese ratings separados por comas (ej: 4,5): ")
                user_filters["average_rating"] = [int(r) for r in selected_ratings.split(',') if r.strip().isdigit()]

            if 3 in selected_indices and filters["category_tags"]:
                print("Etiquetas disponibles:", filters["category_tags"])
                selected_tags = input("Ingrese etiquetas separadas por comas: ")
                user_filters["category_tags"] = [tag.strip() for tag in selected_tags.split(',')]

        # Cargar productos
        with open(file_path, "rb") as f:
            products = pickle.load(f)

        # Aplicar filtros solo si se seleccionaron
        filtered_products = apply_filters(products, user_filters) if user_filters else products[:MAX_PRODUCTS]

        # Resto del flujo...
        result = show_products(filtered_products)
        
        if result["action"] == "select":
            product = result["product"]
            print("\n" + "═"*50)
            print("PRODUCTO SELECCIONADO".center(50))
            print("═"*50)
            
            print(f"Nombre: {product.get('title', 'N/A') or 'N/A'}")
            print(f"Precio: {format_price(product.get('price'))}")
            print(f"Rating: {product.get('average_rating', 'N/A')}")
            print(f"Categorías: {safe_join(product.get('categories'))}")
            
            # Llamada corregida al nombre de la función
            show_details(product.get('details', {}))
            
            print("═"*50 + "\n")
            
            # Aquí iría tu llamada al modelo de RL para recompensa/castigo
            feedback = input("¿Le gustó este producto? (s/n): ").lower()
            if feedback == 's':
                print("¡Perfecto! Guardando preferencia...")
                # Lógica de recompensa al RL
            else:
                print("Buscando alternativas...")
                # Lógica de castigo al RL
                continue
            
            if input("¿Desea realizar otra búsqueda? (s/n): ").lower() != 's':
                print("¡Gracias por usar nuestro sistema!")
                break
                
        elif result["action"] == "retry":
            continue
            
        elif result["action"] == "exit":
            print("¡Hasta pronto!")
            break
