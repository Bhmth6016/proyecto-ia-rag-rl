from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, Button, SelectionList, Static, Input, 
    DataTable, Label, TabbedContent, TabPane
)
from textual.containers import ScrollableContainer, Container, Horizontal, Vertical
from textual.screen import Screen, ModalScreen
from textual import on, work
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import webbrowser

@dataclass
class CategoryNode:
    """Estructura para nodos del árbol de categorías"""
    name: str
    products: List[Dict]
    children: List['CategoryNode']
    parent: Optional['CategoryNode'] = None

class ProductDetailScreen(ModalScreen):
    """Pantalla modal para detalles de producto"""
    CSS = """
    ProductDetailScreen {
        align: center middle;
        width: 80%;
        height: 80%;
        border: round $accent;
    }
    """

    def __init__(self, product: Dict):
        super().__init__()
        self.product = product

    def compose(self) -> ComposeResult:
        yield Container(
            Label(f"[b]{self.product.get('title', 'Unknown')}[/]"),
            Static(f"Price: ${self.product.get('price', 'N/A')}"),
            Static(f"Rating: {self.product.get('average_rating', 'N/A')}/5"),
            Static(self._format_details()),
            Horizontal(
                Button("View on Amazon", id="amazon"),
                Button("Back", id="back"),
            ),
            id="detail-container"
        )

    def _format_details(self) -> str:
        details = self.product.get('details', {})
        return "\n".join(f"{k}: {v}" for k, v in details.items())

    @on(Button.Pressed, "#amazon")
    def open_amazon(self):
        if 'url' in self.product:
            webbrowser.open(self.product['url'])

    @on(Button.Pressed, "#back")
    def back(self):
        self.dismiss()

class AmazonProductUI(App):
    """Interfaz principal de la aplicación"""
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 2fr;
    }
    
    #sidebar {
        border-right: solid $panel;
    }
    
    #product-table {
        height: 80%;
    }
    
    #search-input {
        width: 100%;
        margin-bottom: 1;
    }
    
    .highlight {
        text-style: bold;
        background: $accent;
    }
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "toggle_search", "Search"),
        ("f", "toggle_filters", "Filters")
    ]

    def __init__(self, products: List[Dict], category_tree: Any, rag_agent: Any):
        super().__init__()
        self.products = products
        self.category_tree = category_tree
        self.rag_agent = rag_agent
        self.current_category = None
        self.filtered_products = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="sidebar"):
            yield Input(placeholder="Search...", id="search-input")
            yield SelectionList(
                *[(cat.name, cat) for cat in self.category_tree.root.children],
                id="category-list"
            )
        
        with Container():
            with TabbedContent():
                with TabPane("Products"):
                    yield DataTable(id="product-table")
                with TabPane("Recommendations"):
                    yield Static("Ask our AI assistant...", id="recommendations")
        
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#product-table", DataTable)
        table.add_columns("ID", "Title", "Price", "Rating")
        self._update_product_table(self.products)

    def _update_product_table(self, products: List[Dict]):
        table = self.query_one("#product-table", DataTable)
        table.clear()
        for idx, product in enumerate(products[:100]):  # Limit to 100 for performance
            table.add_row(
                str(idx),
                product.get('title', 'Unknown')[:50],
                f"${product.get('price', 'N/A')}",
                str(product.get('average_rating', 'N/A')))
    
    @on(SelectionList.Selected, "#category-list")
    def category_selected(self, event: SelectionList.Selected):
        selected_category = event.selection_list.selected[0]
        if isinstance(selected_category, CategoryNode):
            self.current_category = selected_category
            self.filtered_products = selected_category.products
            self._update_product_table(self.filtered_products)

    @on(Input.Changed, "#search-input")
    def on_search_changed(self, event: Input.Changed):
        if not event.value:
            self._update_product_table(self.filtered_products or self.products)
            return
        
        search_term = event.value.lower()
        filtered = [
            p for p in (self.filtered_products or self.products)
            if search_term in p.get('title', '').lower()
        ]
        self._update_product_table(filtered)

    @on(DataTable.RowSelected, "#product-table")
    def show_product_detail(self, event: DataTable.RowSelected):
        product_idx = int(event.row_key.value)
        products = self.filtered_products or self.products
        if 0 <= product_idx < len(products):
            self.push_screen(ProductDetailScreen(products[product_idx]))

    @work(thread=True)
    def get_recommendations(self, query: str):
        """Obtiene recomendaciones usando el agente RAG (en segundo plano)"""
        response = self.rag_agent.query(query)
        self.call_from_thread(
            self.update_recommendations,
            response['response']
        )

    def update_recommendations(self, text: str):
        """Actualiza la UI con las recomendaciones"""
        self.query_one("#recommendations", Static).update(text)

    def action_toggle_search(self):
        search = self.query_one("#search-input")
        search.focus()

def launch_ui(products: List[Dict], category_tree: Any, rag_agent: Any):
    """Función para iniciar la interfaz desde main.py"""
    app = AmazonProductUI(products, category_tree, rag_agent)
    app.run()