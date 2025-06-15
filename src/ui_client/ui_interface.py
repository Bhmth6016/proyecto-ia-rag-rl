import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from collections import defaultdict
from src.data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_PRODUCTS = 20

class ProductInterface:
    def __init__(self, data_loader: DataLoader):
        self.loader = data_loader
        self.products = []
        self.category_tree = {}
        self.filters = {}
        
    def load_products(self, use_cache: bool = True) -> bool:
        try:
            self.products = self.loader.load_data(use_cache)
            self.category_tree = self._build_category_tree()
            self.filters = self._extract_filters()
            return True
        except Exception as e:
            logger.error(f"Error cargando productos: {e}")
            return False
    
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
        selected = {}
        
        if not self.filters.get('categories'):
            logger.warning(f"No hay categorías disponibles para filtrar en {category}")
            return selected
        print(f"\nFiltros para {category}:")
        print("1. Precio")
        print("2. Rating")
        print("3. Características")
        print("0. Saltar")
        
        selected = {}
        choice = input("\nSeleccione filtros (ej: 1,3): ")
        
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
            ratings = input("Incluir ratings (ej: 4,5): ")
            selected['ratings'] = [int(r) for r in ratings.split(',') if r.strip().isdigit()]
        
        if '3' in choice:
            print("\nCaracterísticas disponibles:")
            for i, (k, v) in enumerate(self.filters['details'].items(), 1):
                print(f"{i}. {k}: {', '.join(v[:3])}{'...' if len(v) > 3 else ''}")
            
            feature = input("\nSeleccione característica: ")
            if feature.isdigit() and 1 <= int(feature) <= len(self.filters['details']):
                key = list(self.filters['details'].keys())[int(feature)-1]
                values = [v.strip() for v in input(f"Valores para {key} (separados por coma): ").split(',') if v.strip()]
                if values:
                    selected['details'] = {key: values}
        
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
                if not isinstance(price, (int, float)):
                    valid = False
                elif not (filters['price_range'][0] <= price <= filters['price_range'][1]):
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
        
        return filtered
    
    def show_products(self, products: List[Dict]) -> Dict[str, Any]:
        if not products:
            print("\nNo hay productos con estos filtros")
            return {'action': 'retry'}
        
        print("\n" + "="*60)
        print(f" PRODUCTOS ({len(products)}) ".center(60))
        print("="*60)
        
        for i, product in enumerate(products, 1):
            title = (product.get('title') or 'Sin título')[:50]
            print(f"{i}. {title}")
            print(f"   Precio: {self.format_price(product.get('price'))} | Rating: {product.get('average_rating', 'N/A')}")
        
        print("\nOpciones:")
        print(f"{len(products)+1}. Nuevos filtros")
        print(f"{len(products)+2}. Volver")
        print(f"{len(products)+3}. Salir")
        print("="*60)
        
        while True:
            choice = input("\nSelección: ")
            if choice.isdigit():
                num = int(choice)
                if 1 <= num <= len(products):
                    return {'action': 'select', 'product': products[num-1]}
                elif num == len(products)+1:
                    return {'action': 'filter'}
                elif num == len(products)+2:
                    return {'action': 'back'}
                elif num == len(products)+3:
                    return {'action': 'exit'}
            print("Opción inválida")
    
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
                if not navigation_stack:
                    action = self.show_main_menu()
                    if action == 'exit':
                        break
                    current_category = action
                    navigation_stack.append(('category', current_category))
                    current_products = self.category_tree[current_category]
                    continue
                
                if navigation_stack[-1][0] == 'category':
                    current_filters = self.show_filters_menu(current_category)
                    filtered = self.apply_filters(current_products, current_filters)
                    navigation_stack.append(('filtered', current_filters))
                
                elif navigation_stack[-1][0] == 'filtered':
                    result = self.show_products(
                        self.apply_filters(current_products, current_filters)
                    )
                    
                    if result['action'] == 'select':
                        self.show_product_detail(result['product'])
                    elif result['action'] == 'filter':
                        navigation_stack.pop()
                    elif result['action'] == 'back':
                        navigation_stack.pop()
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