# src/core/category_search/category_tree.py
import json
import logging
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from .filters import extract_global_filters


logger = logging.getLogger(__name__)

@dataclass
class CategoryNode:
    name: str
    products: List[Dict]
    filters: Dict[str, Union[List, Dict]]
    parent: Optional['CategoryNode'] = None
    children: List['CategoryNode'] = None

    def __post_init__(self):
        self.children = self.children or []

class CategoryTree:
    def __init__(self, products: List[Dict], min_products_per_category: int = 10):
        self.products = products
        self.min_products = min_products_per_category
        self.root = CategoryNode(name="root", products=[], filters={})
        self._category_map = {}
        
    def build_tree(self) -> CategoryNode:
        """Construye el árbol de categorías jerárquico"""
        # 1. Agrupar productos por categoría principal
        categories = self._group_by_main_category()
        
        # 2. Construir estructura jerárquica
        for category_path, products in categories.items():
            self._add_category_path(category_path, products)
            
        # 3. Generar filtros para cada categoría
        self._generate_filters()
        
        return self.root
    
    def _group_by_main_category(self) -> Dict[str, List[Dict]]:
        """Agrupa productos por su categoría principal"""
        categories = defaultdict(list)
        
        for product in self.products:
            if not self._is_valid_product(product):
                continue
                
            main_category = self._normalize_category(product.get('main_category'))
            if main_category:
                categories[main_category].append(product)
                
        # Filtrar categorías con muy pocos productos
        return {k: v for k, v in categories.items() if len(v) >= self.min_products}
    
    def _add_category_path(self, category_path: str, products: List[Dict]):
        """Añade una categoría al árbol jerárquico"""
        parts = self._split_category_path(category_path)
        current_node = self.root
        
        for part in parts:
            if part not in [child.name for child in current_node.children]:
                new_node = CategoryNode(
                    name=part,
                    products=[],
                    filters={},
                    parent=current_node
                )
                current_node.children.append(new_node)
                current_node = new_node
                self._category_map[part] = current_node
            else:
                current_node = next(child for child in current_node.children if child.name == part)
        
        current_node.products = products
    
    def _generate_filters(self):
        """Genera filtros para cada categoría en el árbol"""
        def _process_node(node: CategoryNode):
            if node.products:
                node.filters = self._extract_filters_from_products(node.products)
            
            for child in node.children:
                _process_node(child)
        
        _process_node(self.root)
    
    def _extract_filters_from_products(self, products: List[Dict]) -> Dict[str, Union[List, Dict]]:
        """Extrae filtros de una lista de productos"""
        filters = {
            "price_range": [float('inf'), 0],
            "ratings": set(),
            "brands": set(),
            "features": defaultdict(set)
        }
        
        for product in products:
            # Filtro por precio
            if isinstance(product.get('price'), (int, float)):
                filters["price_range"][0] = min(filters["price_range"][0], product['price'])
                filters["price_range"][1] = max(filters["price_range"][1], product['price'])
            
            # Filtro por rating
            if isinstance(product.get('average_rating'), (int, float)):
                rounded = round(product['average_rating'])
                if 0 <= rounded <= 5:
                    filters["ratings"].add(rounded)
            
            # Filtro por marca
            if isinstance(product.get('details', {}).get('Brand'), str):
                filters["brands"].add(product['details']['Brand'].strip())
            
            # Filtro por características
            if isinstance(product.get('details'), dict):
                for k, v in product['details'].items():
                    if k != 'Brand' and v is not None:
                        filters["features"][k].add(str(v).strip())
        
        return {
            "price_range": [
                filters["price_range"][0] if filters["price_range"][0] != float('inf') else 0,
                max(filters["price_range"][1], 0)
            ],
            "ratings": sorted(filters["ratings"]),
            "brands": sorted(filters["brands"]),
            "features": {k: sorted(v) for k, v in filters["features"].items()}
        }
    
    def save_to_json(self, file_path: Union[str, Path]):
        """Guarda el árbol de categorías en formato JSON"""
        def _node_to_dict(node: CategoryNode) -> Dict:
            return {
                "name": node.name,
                "product_count": len(node.products),
                "filters": node.filters,
                "children": [_node_to_dict(child) for child in node.children]
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(_node_to_dict(self.root), f, indent=2, ensure_ascii=False)
            
    def _extract_filters_from_products(self, products: List[Dict]) -> Dict[str, Any]:
        """Usa la función del módulo filters"""
        return extract_global_filters(products)
    
    @staticmethod
    def _is_valid_product(product: Dict) -> bool:
        """Valida que un producto tenga la estructura mínima requerida"""
        return (
            isinstance(product, dict) and 
            isinstance(product.get('title'), str) and 
            product.get('title').strip() and
            isinstance(product.get('main_category'), str)
        )
    
    @staticmethod
    def _normalize_category(category: Optional[str]) -> Optional[str]:
        """Normaliza nombres de categoría"""
        if not category:
            return None
            
        # Eliminar prefijos comunes y normalizar
        category = re.sub(r'^meta_', '', category)
        category = re.sub(r'_processed\.pkl$', '', category)
        category = re.sub(r'[^\w\s-]', '', category)
        category = category.replace('_', ' ').strip()
        
        return category.title() if category else None
    
    @staticmethod
    def _split_category_path(category_path: str) -> List[str]:
        """Divide una ruta de categoría en sus componentes jerárquicos"""
        # Implementación básica - puede mejorarse según necesidades
        return [part.strip() for part in category_path.split('>') if part.strip()]

def generar_categorias_y_filtros(products: List[Dict], output_file: Path) -> Optional[Dict]:
    """Función de conveniencia para la interfaz original"""
    try:
        tree = CategoryTree(products)
        root = tree.build_tree()
        
        # Convertir a formato compatible con la UI existente
        result = {
            "global": tree._extract_filters_from_products(products),
            "by_category": {}
        }
        
        # Procesar categorías principales (primer nivel)
        for category_node in root.children:
            result["by_category"][category_node.name] = {
                **category_node.filters,
                "products": category_node.products
            }
        
        # Guardar archivo
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result
        
    except Exception as e:
        logger.error(f"Error generando categorías: {str(e)}", exc_info=True)
        return None