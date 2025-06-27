import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from collections import defaultdict
from src.data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_PRODUCTS = 20

class ProductInterface:
    def __init__(self, loader=None, filters_path=None, products_path=None):
        self.loader = loader
        self.filters_path = filters_path or "data/processed/category_filters.json"
        self.products_path = products_path or "data/processed"
        self.selected_category = None
        self.filters = self.load_filters()
        self.products = []
        self.selected = {}
        self.category_tree = {}  # Añadir esta línea

    def load_products(self):
        """Cargar productos desde el loader"""
        try:
            self.products = self.loader.load_data(use_cache=True)
            self.category_tree = self._build_category_tree()
            return True
        except Exception as e:
            logger.error(f"Error cargando productos: {e}")
            return False

    def load_filters(self):
        try:
            with open(self.filters_path, "r", encoding="utf-8") as f:
                filters = json.load(f)
            return filters
        except Exception as e:
            print(f" Error cargando filtros desde {self.filters_path}: {e}")
            return {}

    
    def _build_category_tree(self) -> Dict[str, List[Dict]]:
        tree = defaultdict(list)
        for product in self.products:
            if not isinstance(product, dict):
                continue
            category = product.get('main_category')
            if not isinstance(category, str):
                category = 'Uncategorized'
            tree[category].append(product)
        return dict(tree)
    
    def _extract_filters(self) -> Dict[str, Any]:
        filters = {
            'price_range': {'min': float('inf'), 'max': 0},
            'ratings': set(),
            'categories': set(),
            'details': defaultdict(set)
        }

        for product in self.products:
            if not isinstance(product, dict):
                continue
                
            price = product.get('price')
            if isinstance(price, (int, float)) and not isinstance(price, bool):
                filters['price_range']['min'] = min(filters['price_range']['min'], price)
                filters['price_range']['max'] = max(filters['price_range']['max'], price)
            
            rating = product.get('average_rating')
            if isinstance(rating, (int, float)) and not isinstance(rating, bool):
                rounded = int(round(rating))
                if 0 <= rounded <= 5:
                    filters['ratings'].add(rounded)
            
            categories = product.get('categories')
            if isinstance(categories, list):
                valid_categories = set()
                for cat in categories:
                    if cat is None:
                        continue
                    try:
                        cleaned = str(cat).strip()
                        if cleaned:
                            valid_categories.add(cleaned.lower())
                    except (AttributeError, TypeError):
                        continue
                filters['categories'].update(valid_categories)
            
            details = product.get('details')
            if isinstance(details, dict):
                for key, value in details.items():
                    if value is None:
                        continue
                    try:
                        cleaned_value = str(value).strip()
                        if cleaned_value and isinstance(key, str):
                            filters['details'][key.strip().lower()].add(cleaned_value)
                    except (AttributeError, TypeError):
                        continue
        
        return {
            'price_range': [
                filters['price_range']['min'] if filters['price_range']['min'] != float('inf') else 0,
                max(filters['price_range']['max'], 0)
            ],
            'ratings': sorted({r for r in filters['ratings'] if isinstance(r, int) and 0 <= r <= 5}),
            'categories': sorted(
                {c for c in filters['categories'] if isinstance(c, str) and c},
                key=str.lower
            ),
            'details': {
                k: sorted({v for v in values if isinstance(v, str) and v})
                for k, values in filters['details'].items()
                if isinstance(k, str) and k
            }
        }
    
    @staticmethod
    def format_price(price: Optional[Union[float, str]]) -> str:
        if price is None:
            return "N/A"
        try:
            return f"${float(price):.2f}"
        except (ValueError, TypeError):
            return str(price)
    
    def show_main_menu(self) -> str:
        print("\n" + "="*60)
        print(" SISTEMA DE RECOMENDACIÓN ".center(60))
        print("="*60)
        
        categories = sorted(self.category_tree.keys())
        for i, cat in enumerate(categories, 1):
            count = len(self.category_tree[cat])
            print(f"{i}. {cat} ({count} productos)")
        
        print("\n0. Salir")
        print("="*60)
        
        while True:
            choice = input("\nSeleccione una categoría: ")
            if choice == '0':
                return 'exit'
            if choice.isdigit() and 1 <= int(choice) <= len(categories):
                return categories[int(choice)-1]
            print("Opción inválida. Intente nuevamente.")
    
    def show_filters_menu(self, category: str) -> Dict[str, Any]:
        if not self.filters:
            logger.debug(f"No hay filtros disponibles para {category}")
            return {}
        
        
        print(f"\nFiltros para {category}:")
        print("1. Precio")
        print("2. Rating")
        print("3. Características")
        print("4. Limpiar todos los filtros")
        print("0. Volver sin cambiar")
        
        selected = {}
        while True:
            choice = input("\nSeleccione filtros (ej: 1,3): ").strip()
            
            if choice == '0':
                return None  # Indicar que el usuario quiere volver
            
            if choice == '4':
                return {} 
            
            if '1' in choice:
                min_p, max_p = self.filters['price_range']
                print(f"\nRango actual: {self.format_price(min_p)} - {self.format_price(max_p)}")
                try:
                    min_val = float(input(f"Mínimo ({self.format_price(min_p)}): ") or min_p)
                    max_val = float(input(f"Máximo ({self.format_price(max_p)}): ") or max_p)
                    selected['price_range'] = [min_val, max_val]
                except ValueError:
                    print("¡Valor inválido! Usando valores por defecto")
                    selected['price_range'] = [min_p, max_p]
            
            if '2' in choice:
                print("\nRatings:", ', '.join(map(str, self.filters['ratings'])))
                ratings = input("Ingresa los ratings deseados separados por coma (e.g., 4,5): ")
                raw_ratings = [r.strip() for r in ratings.split(',')]
                valid_ratings = []
                invalid_ratings = []

                for r in raw_ratings:
                    if r.isdigit() and 1 <= int(r) <= 5:
                        valid_ratings.append(int(r))
                    else:
                        invalid_ratings.append(r)

                if not valid_ratings:
                    print(" No se ingresaron ratings válidos. Intenta de nuevo.")
                    return self.show_filters_menu(self.filters)

                if invalid_ratings:
                    print(f" Estos valores fueron ignorados por no ser ratings válidos: {', '.join(invalid_ratings)}")

                selected['ratings'] = valid_ratings

            
            if '3' in choice:
                print("\nCaracterísticas disponibles:")
                features = list(self.filters['details'].items())
                
                for i, (k, v) in enumerate(features, 1):
                    print(f"{i}. {k}: {', '.join(v[:3])}{'...' if len(v) > 3 else ''}")
                
                try:
                    feature_choice = int(input("\nSeleccione característica (número): "))
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
            
            # Si se seleccionó algún filtro, retornar
            if selected:
                return selected
            
            print("\n¡No se seleccionaron filtros válidos! Intente nuevamente.")
            
            return selected
    
    def apply_filters(self, products: List[Dict], filters: Dict) -> List[Dict]:
        if not products or not filters:
            return products[:MAX_PRODUCTS]
        
        filtered = []
        for product in products:
            if not isinstance(product, dict):
                continue
            
            valid = True
            
            if 'price_range' in filters:
                price = product.get('price')
                if isinstance(price, (int, float)):
                    if not (filters['price_range'][0] <= price <= filters['price_range'][1]):
                        valid = False
                else:
                    valid = False
            
            if valid and 'ratings' in filters:
                rating = product.get('average_rating')
                if not isinstance(rating, (int, float)) or round(rating) not in filters['ratings']:
                    valid = False
            
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
            
            
            print("\n¡Selección inválida! Intente nuevamente")
        
        return filtered
    
    def show_products(self, products: List[Dict]) -> Dict[str, Any]:
        if not products:
            print("\nNo hay productos con estos filtros")
            return {'action': 'filter'}  # Sugerir cambiar filtros
        
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
    
    def show_product_detail(self, product: Dict) -> None:
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
        if not self.load_products():
            return
        
        navigation_stack = []
        current_category = None
        current_filters = {}
        current_products = []
        
        try:
            while True:
                # Menú principal cuando no hay categoría seleccionada
                if not current_category:
                    action = self.show_main_menu()
                    if action == 'exit':
                        break
                    current_category = action
                    current_products = self.category_tree.get(current_category, [])
                    continue
                
                # Mostrar filtros si no se han aplicado o se solicitan nuevos
                if not current_filters or navigation_stack and navigation_stack[-1] == 'new_filters':
                    current_filters = self.show_filters_menu(current_category)
                    if navigation_stack and navigation_stack[-1] == 'new_filters':
                        navigation_stack.pop()
                
                # Aplicar filtros y mostrar productos
                filtered_products = self.apply_filters(current_products, current_filters)
                result = self.show_products(filtered_products)
                
                # Manejar acciones del usuario
                if result['action'] == 'select':
                    self.show_product_detail(result['product'])
                elif result['action'] == 'filter':
                    navigation_stack.append('new_filters')  # Solicitar nuevos filtros
                elif result['action'] == 'back':
                    current_category = None  # Volver al menú principal
                    current_filters = {}
                elif result['action'] == 'exit':
                    break
                    
        except KeyboardInterrupt:
            print("\nOperación cancelada")
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            print("\nGracias por usar el sistema")

def run_interface(products: List[Dict[str, Any]], filters_path: Optional[Path] = None) -> None:
    loader = DataLoader()
    interface = ProductInterface(loader)
    interface.run()