import json
import logging
from typing import List, Dict, Optional, Union, Any
from src.validaciones import ProductModel
from src.data_loader import DataLoader
from src.config.settings import CATEGORY_FILTERS_FILE
from src.utils.types import Filters, CategoryFilters

logger = logging.getLogger(__name__)

MAX_PRODUCTS = 20

class ProductInterface:
    def __init__(self, loader=None, filters: Optional[CategoryFilters] = None, products: Optional[List[Dict[str, Any]]] = None):
        self.loader = loader or DataLoader()
        self.filters = filters or self.load_filters()
        self.products = products or []
        self.selected_category = None
        self.selected = {}

    def load_filters(self) -> CategoryFilters:
        try:
            with open(CATEGORY_FILTERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"No se pudo cargar filtros: {e}")
            return {
                'global': {
                    'price_range': [0, 0],
                    'ratings': [],
                    'details': {},
                    'categories': []
                },
                'by_category': {}
            }

    def get_current_filters(self) -> Filters:
        if self.selected_category and self.filters['by_category'].get(self.selected_category):
            return self.filters['by_category'][self.selected_category]
        return self.filters['global']

    def apply_filters(self, products: List[Dict[str, Any]], filters: Filters) -> List[Dict[str, Any]]:
        if not products or not filters:
            return products[:MAX_PRODUCTS]

        filtered = []
        for product in products:
            if not isinstance(product, dict):
                continue

            valid = True

            # Filtro por precio
            if 'price_range' in filters:
                price = product.get('price')
                if isinstance(price, (int, float)):
                    if not (filters['price_range'][0] <= price <= filters['price_range'][1]):
                        valid = False
                else:
                    valid = False

            # Filtro por rating
            if valid and 'ratings' in filters:
                rating = product.get('average_rating')
                if not isinstance(rating, (int, float)) or round(rating) not in filters['ratings']:
                    valid = False

            # Filtro por características
            if valid and 'details' in filters:
                detail_match = False
                for k, values in filters['details'].items():
                    detail_value = str(product.get('details', {}).get(k, '')).strip()
                    if detail_value in values:
                        detail_match = True
                        break
                valid = detail_match

            if valid:
                filtered.append(product)
                if len(filtered) >= MAX_PRODUCTS:
                    break

        logger.info(f"Filtrados {len(filtered)} productos de {len(products)} totales")
        return filtered

    def show_main_menu(self) -> str:
        print("\n" + "="*60)
        print(" SISTEMA DE RECOMENDACIÓN DE AMAZON ".center(60))
        print("="*60)

        categories = sorted(self.filters['by_category'].keys())
        for i, cat in enumerate(categories, 1):
            count = len(self.filters['by_category'][cat]['products'])
            print(f"{i}. {cat} ({count} productos)")

        print("\n0. Salir")
        print("="*60)

        while True:
            choice = input("\nSeleccione una categoría (0-{}): ".format(len(categories)))
            if choice == '0':
                return 'exit'
            if choice.isdigit() and 1 <= int(choice) <= len(categories):
                return categories[int(choice)-1]
            print("¡Opción inválida! Por favor ingrese un número entre 0 y {}".format(len(categories)))

    def show_filters_menu(self, category: str) -> Dict[str, Any]:
        current_filters = self.get_current_filters()
        selected = {}

        print(f"\nFiltros para {category}:")
        print("1. Precio")
        print("2. Rating")
        print("3. Características")
        print("0. Volver")

        while True:
            choice = input("\nSeleccione filtros (ej: 1,3): ").strip()

            if choice == '0':
                return None

            if '1' in choice:
                min_p, max_p = current_filters['price_range']
                print(f"\nRango disponible: {self.format_price(min_p)} - {self.format_price(max_p)}")
                try:
                    min_val = float(input(f"Nuevo mínimo ({self.format_price(min_p)}): ") or min_p)
                    max_val = float(input(f"Nuevo máximo ({self.format_price(max_p)}): ") or max_p)
                    selected['price_range'] = [min_val, max_val]
                except ValueError:
                    print("¡Valor inválido! Usando valores por defecto")
                    selected['price_range'] = [min_p, max_p]

            if '2' in choice:
                available_ratings = current_filters.get('ratings', [])
                print(f"\nRatings disponibles: {', '.join(map(str, available_ratings))}")
                ratings_input = input("Ingrese ratings deseados (separados por coma): ")
                selected_ratings = {int(r.strip()) for r in ratings_input.split(',') if r.strip().isdigit()}
                valid_ratings = [r for r in selected_ratings if r in available_ratings]

                if valid_ratings:
                    selected['ratings'] = valid_ratings
                else:
                    print("No se seleccionaron ratings válidos")
                    continue

            if '3' in choice:
                available_features = current_filters.get('details', {})
                if not available_features:
                    print("No hay características disponibles para filtrar")
                    continue

                print("\nCaracterísticas disponibles:")
                features = list(available_features.items())
                for i, (k, v) in enumerate(features, 1):
                    print(f"{i}. {k}: {', '.join(v[:3])}{'...' if len(v) > 3 else ''}")

                try:
                    feature_choice = input("\nSeleccione característica (número o 0 para cancelar): ")
                    if feature_choice == '0':
                        continue
                    feature_choice = int(feature_choice)
                    if 1 <= feature_choice <= len(features):
                        key, _ = features[feature_choice-1]
                        values_input = input(f"Valores deseados para {key} (separados por coma): ")
                        values = [v.strip() for v in values_input.split(',') if v.strip()]

                        if values:
                            selected.setdefault('details', {})[key] = values
                    else:
                        print("¡Número de característica inválido!")
                        continue
                except ValueError:
                    print("¡Debe ingresar un número!")
                    continue

            if selected:
                return selected

            print("\n¡No se seleccionaron filtros válidos! Intente nuevamente o ingrese 0 para volver.")

    def format_price(self, price: Optional[Union[float, str]]) -> str:
        if price is None:
            return "N/A"
        try:
            return f"${float(price):.2f}"
        except (ValueError, TypeError):
            return str(price)

    def show_products(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not products:
            print("\nNo hay productos con estos filtros")
            return {'action': 'filter'}

        print("\n" + "="*60)
        print(f" PRODUCTOS ({len(products)}) ".center(60))
        print("="*60)

        for i, product in enumerate(products, 1):
            title = (product.get('title') or 'Sin título')[:50]
            print(f"{i}. {title}")
            print(f"   Precio: {self.format_price(product.get('price'))} | Rating: {product.get('average_rating', 'N/A')}")

        print("\nOpciones:")
        print("F. Cambiar filtros")
        print("C. Cambiar categoría")
        print("S. Salir")
        print("="*60)

        while True:
            choice = input("\nSeleccione producto (1-{}) u opción: ".format(len(products))).upper()

            if choice.isdigit() and 1 <= int(choice) <= len(products):
                return {'action': 'select', 'product': products[int(choice)-1]}
            elif choice == 'F':
                return {'action': 'filter'}
            elif choice == 'C':
                return {'action': 'back'}
            elif choice == 'S':
                return {'action': 'exit'}
            print("Opción inválida. Intente nuevamente.")

    def show_product_detail(self, product: Dict[str, Any]) -> None:
        print("\n" + "="*60)
        print(" DETALLES ".center(60))
        print("="*60)
        print(f"Nombre: {product.get('title', 'N/A')}")
        print(f"Precio: {self.format_price(product.get('price'))}")
        print(f"Rating: {product.get('average_rating', 'N/A')}")
        print(f"Categoría: {product.get('main_category', 'N/A')}")

        if categories := product.get('categories'):
            print(f"\nOtras categorías: {', '.join(categories)}")

        if details := product.get('details'):
            print("\nEspecificaciones:")
            for k, v in details.items():
                print(f"- {k}: {v if v else 'N/A'}")

        print("="*60)
        input("\nPresione Enter para continuar...")

    def run(self) -> None:
        try:
            while True:
                category = self.show_main_menu()
                if category == 'exit':
                    break

                current_products = self.filters['by_category'][category]['products']

                while True:
                    filters = self.show_filters_menu(category)
                    if filters is None:
                        break

                    filtered = self.apply_filters(current_products, filters)

                    if not filtered:
                        print("\nNo se encontraron productos con estos filtros")
                        continue

                    result = self.show_products(filtered)

                    if result['action'] == 'select':
                        self.show_product_detail(result['product'])
                    elif result['action'] == 'filter':
                        continue
                    elif result['action'] == 'back':
                        break
                    elif result['action'] == 'exit':
                        return

        except KeyboardInterrupt:
            print("\nOperación cancelada por el usuario")
        except Exception as e:
            logger.error(f"Error crítico: {e}")
        finally:
            print("\nSesión finalizada")

def run_interface(products: List[Dict[str, Any]], filters: Optional[Dict[str, Any]] = None) -> None:
    loader = DataLoader()
    interface = ProductInterface(loader, filters, products)
    interface.run()