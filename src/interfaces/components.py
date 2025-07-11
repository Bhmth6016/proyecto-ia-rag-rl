# src/interfaces/components.py
from typing import List, Dict, Any
from textual.widgets import Static, DataTable

class ProductDetail(Static):
    """Widget para mostrar detalles de producto"""
    def render(self, product: Dict) -> str:
        return f"""
        {product['title']}
        Price: ${product.get('price', 'N/A')}
        Rating: {product.get('rating', 'N/A')}/5
        """

class ProductTable(DataTable):
    """Tabla interactiva de productos"""
    def populate(self, products: List[Dict]):
        self.add_columns("ID", "Name", "Price")
        for p in products:
            self.add_row(p['id'], p['title'][:50], f"${p.get('price', 'N/A')}")