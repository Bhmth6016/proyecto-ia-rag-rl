# src/interfaces/ui.py
import logging
import itertools
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from enum import Enum, auto
import curses
from curses import wrapper, textpad
from dataclasses import dataclass

from src.core.utils.logger import get_logger
from src.core.data.loader import DataLoader
from src.core.category_search.category_tree import CategoryTree
from src.core.category_search.filters import ProductFilter, extract_global_filters

logger = get_logger(__name__)

class UIMode(Enum):
    CATEGORY = auto()
    FILTER = auto()
    PRODUCT = auto()
    SEARCH = auto()

@dataclass
class UIState:
    current_mode: UIMode = UIMode.CATEGORY
    current_category: Optional[str] = None
    current_product_index: int = 0
    current_page: int = 0
    current_filter: Optional[ProductFilter] = None
    filtered_products: List[Dict] = None
    selected_index: int = 0
    search_query: str = ""

class AmazonProductUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.state = UIState()
        self.loader = DataLoader()
        self.products = self.loader.load_data()
        self.category_tree = CategoryTree(self.products)
        self.category_tree.build_tree()
        self._setup_curses()
        self._load_filters()
        self.page_size = 10
        self.max_products_to_cache = 1000  # Límite para caché de productos

    def _setup_curses(self):
        """Initialize curses settings"""
        curses.curs_set(0)  # Hide cursor
        self.stdscr.keypad(True)  # Enable special keys
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Header
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Selected item
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Instructions
        curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Highlight
        curses.init_pair(5, curses.COLOR_RED, curses.COLOR_BLACK)  # Warnings
        curses.init_pair(6, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Pagination info

    def _load_filters(self):
        """Load or generate product filters with caching"""
        filters_file = Path("data/processed/category_filters.json")
        if filters_file.exists():
            with open(filters_file, 'r', encoding='utf-8') as f:
                self.global_filters = json.load(f)
        else:
            self.global_filters = extract_global_filters(self.products)
            # Cache the filters for future use
            with open(filters_file, 'w', encoding='utf-8') as f:
                json.dump(self.global_filters, f)

    def run(self):
        """Main UI loop with error handling"""
        while True:
            try:
                self.stdscr.clear()
                self._draw_header()
                
                if self.state.current_mode == UIMode.CATEGORY:
                    self._show_category_menu()
                elif self.state.current_mode == UIMode.FILTER:
                    self._show_filter_menu()
                elif self.state.current_mode == UIMode.PRODUCT:
                    self._show_product_view()
                elif self.state.current_mode == UIMode.SEARCH:
                    self._show_search()
                
                self.stdscr.refresh()
                key = self.stdscr.getch()
                
                if not self._handle_input(key):
                    break
                    
            except Exception as e:
                logger.error(f"UI Error: {str(e)}", exc_info=True)
                self._show_error(str(e))
                self.stdscr.getch()  # Wait for key press
                continue

    def _draw_header(self):
        """Draw the UI header with current context"""
        title = " Amazon Product Explorer "
        if self.state.current_category:
            title += f"- {self.state.current_category[:20]} "  # Trim long category names
        
        # Center the title
        title_x = max(0, (curses.COLS - len(title)) // 2)
        self.stdscr.addstr(0, title_x, title, curses.color_pair(1))
        
        # Status line with current mode
        status = {
            UIMode.CATEGORY: "Browse categories",
            UIMode.FILTER: "Apply filters",
            UIMode.PRODUCT: "View products",
            UIMode.SEARCH: "Search products"
        }.get(self.state.current_mode, "")
        
        self.stdscr.addstr(1, 0, status, curses.color_pair(3))
        self.stdscr.hline(2, 0, curses.ACS_HLINE, curses.COLS)

    def _show_category_menu(self):
        """Display category selection menu with efficient rendering"""
        categories = self._get_current_categories()
        
        self.stdscr.addstr(3, 0, "Available Categories:", curses.A_BOLD)
        
        for idx, category in enumerate(categories[:curses.LINES-6], 1):
            y = idx + 3
            display_text = f"{category.name} ({len(category.products)} products)"
            
            if idx - 1 == self.state.selected_index:
                self.stdscr.addstr(y, 0, f"> {display_text}", curses.color_pair(2))
            else:
                self.stdscr.addstr(y, 0, f"  {display_text}")

        # Footer instructions
        self._draw_footer("[↑/↓] Navigate [Enter] Select [S] Search [Q] Quit")

    def _show_filter_menu(self):
        """Display filter options with current filter status"""
        self.stdscr.addstr(3, 0, "Filter Options:", curses.A_BOLD)
        
        options = [
            "1. Price Range",
            "2. Ratings",
            "3. Brands",
            "4. Features",
            "0. Back to Categories"
        ]
        
        for idx, option in enumerate(options, 4):
            if idx - 4 == self.state.selected_index:
                self.stdscr.addstr(idx, 0, option, curses.color_pair(2))
            else:
                self.stdscr.addstr(idx, 0, option)

        # Show current filter status
        if self.state.current_filter:
            self.stdscr.addstr(10, 0, "Current Filters:", curses.A_BOLD)
            filters = self.state.current_filter.to_dict()
            y = 11
            for name, value in filters.items():
                if value:
                    self.stdscr.addstr(y, 2, f"{name}: {value}")
                    y += 1

        self._draw_footer("[↑/↓] Navigate [Enter] Apply [Q] Back")

    def _show_product_view(self):
        """Display products with pagination using efficient generators"""
        if not self.state.filtered_products:
            self.stdscr.addstr(3, 0, "No products found with current filters!", curses.color_pair(5))
            return
            
        # Calculate pagination info
        total_products = len(self.state.filtered_products)
        total_pages = (total_products + self.page_size - 1) // self.page_size
        start_idx = self.state.current_page * self.page_size
        end_idx = min(start_idx + self.page_size, total_products)
        
        # Display pagination info
        pagination_info = f"Page {self.state.current_page + 1}/{total_pages} - Products {start_idx + 1}-{end_idx} of {total_products}"
        self.stdscr.addstr(3, 0, pagination_info, curses.color_pair(6))
        
        # Use generator for efficient display of large datasets
        products_to_show = itertools.islice(
            self.state.filtered_products,
            start_idx,
            end_idx
        )
        
        # Display products
        for i, product in enumerate(products_to_show, 4):
            display_text = f"{product.get('title', 'Unknown')[:50]} (${product.get('price', 'N/A')})"
            if i - 4 == self.state.selected_index:
                self.stdscr.addstr(i, 0, display_text, curses.color_pair(2))
            else:
                self.stdscr.addstr(i, 0, display_text)

        self._draw_footer("[↑/↓] Navigate [←→] Pages [Enter] Details [B] Back [Q] Quit")

    def _show_product_detail(self, product: Dict):
        """Display detailed product view with scrollable content"""
        self.stdscr.clear()
        self._draw_header()
        
        # Basic product info
        self.stdscr.addstr(3, 0, product.get('title', 'Unknown Product'), curses.A_BOLD)
        self.stdscr.addstr(4, 0, f"Price: ${product.get('price', 'N/A')}", curses.color_pair(4))
        self.stdscr.addstr(5, 0, f"Rating: {product.get('average_rating', 'N/A')}/5")
        
        # Details with scrollable content
        scroll_pos = 0
        max_lines = curses.LINES - 8
        
        while True:
            self.stdscr.clear()
            self._draw_header()
            
            # Re-draw basic info
            self.stdscr.addstr(3, 0, product.get('title', 'Unknown Product'), curses.A_BOLD)
            self.stdscr.addstr(4, 0, f"Price: ${product.get('price', 'N/A')}", curses.color_pair(4))
            self.stdscr.addstr(5, 0, f"Rating: {product.get('average_rating', 'N/A')}/5")
            
            # Draw details with current scroll position
            y = 7
            details = self._get_product_details(product)
            for line in details[scroll_pos:scroll_pos + max_lines]:
                self.stdscr.addstr(y, 0, line)
                y += 1

            # Footer with scroll instructions
            self._draw_footer("[↑/↓] Scroll [B] Back to List [Q] Quit")
            self.stdscr.refresh()
            
            key = self.stdscr.getch()
            if key == curses.KEY_UP and scroll_pos > 0:
                scroll_pos -= 1
            elif key == curses.KEY_DOWN and scroll_pos < len(details) - max_lines:
                scroll_pos += 1
            elif key == ord('b'):
                return
            elif key == ord('q'):
                raise SystemExit

    def _get_product_details(self, product: Dict) -> List[str]:
        """Prepare product details for display with efficient formatting"""
        details = []
        details.append("Details:")
        
        if 'details' in product:
            for key, value in product['details'].items():
                if isinstance(value, dict):
                    details.append(f"- {key}:")
                    for subkey, subvalue in value.items():
                        details.append(f"  • {subkey}: {subvalue}")
                elif isinstance(value, list):
                    details.append(f"- {key}: {', '.join(map(str, value))}")
                else:
                    details.append(f"- {key}: {value}")
        
        # Ensure all lines are strings and have reasonable length
        return [str(line)[:curses.COLS-1] for line in details]

    def _show_search(self):
        """Display search interface with results"""
        self.stdscr.addstr(3, 0, "Search Products:", curses.A_BOLD)
        
        if self.state.search_query:
            # Use generator for efficient search results
            results = (
                p for p in self.products 
                if re.search(self.state.search_query, p.get('title', ''), re.IGNORECASE)
            )
            
            displayed_results = list(itertools.islice(results, self.page_size * (self.state.current_page + 1)))
            
            if displayed_results:
                for i, product in enumerate(displayed_results[
                    self.state.current_page * self.page_size:
                    (self.state.current_page + 1) * self.page_size
                ], 5):
                    display_text = f"{product.get('title', 'Unknown')[:50]} (${product.get('price', 'N/A')})"
                    if i - 5 == self.state.selected_index:
                        self.stdscr.addstr(i, 0, display_text, curses.color_pair(2))
                    else:
                        self.stdscr.addstr(i, 0, display_text)
            else:
                self.stdscr.addstr(5, 0, "No results found", curses.color_pair(5))
        
        # Show search box
        self.stdscr.addstr(curses.LINES - 3, 0, "Search:", curses.color_pair(3))
        search_box = textpad.Textbox(self.stdscr.subwin(1, curses.COLS - 8, curses.LINES - 3, 8))
        
        self._draw_footer("[Enter] Search [Esc] Cancel")
        self.stdscr.refresh()
        
        # Edit mode for search
        curses.curs_set(1)
        search_edit = search_box.edit()
        curses.curs_set(0)
        
        if search_edit.strip():
            self.state.search_query = search_edit.strip()
            self.state.current_mode = UIMode.SEARCH
            self.state.current_page = 0
            self.state.selected_index = 0

    def _show_error(self, message: str):
        """Display error message at bottom of screen"""
        self.stdscr.addstr(curses.LINES - 1, 0, f"Error: {message[:curses.COLS-1]}", curses.color_pair(5))

    def _draw_footer(self, message: str):
        """Draw footer with instructions"""
        self.stdscr.addstr(curses.LINES - 2, 0, message[:curses.COLS], curses.color_pair(3))

    def _handle_input(self, key: int) -> bool:
        """Process user input with error handling"""
        try:
            if key == ord('q'):
                return False
                
            if self.state.current_mode == UIMode.CATEGORY:
                return self._handle_category_input(key)
            elif self.state.current_mode == UIMode.FILTER:
                return self._handle_filter_input(key)
            elif self.state.current_mode == UIMode.PRODUCT:
                return self._handle_product_input(key)
            elif self.state.current_mode == UIMode.SEARCH:
                return self._handle_search_input(key)
                
            return True
        except Exception as e:
            logger.error(f"Input handling error: {str(e)}")
            self._show_error(str(e))
            return True

    def _handle_category_input(self, key: int) -> bool:
        """Handle input in category mode"""
        categories = self._get_current_categories()
        max_items = len(categories) - 1
        
        if key == curses.KEY_DOWN:
            self.state.selected_index = min(self.state.selected_index + 1, max_items)
        elif key == curses.KEY_UP:
            self.state.selected_index = max(0, self.state.selected_index - 1)
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if categories:
                selected = categories[self.state.selected_index]
                self.state.current_category = selected.name
                self.state.current_mode = UIMode.FILTER
                self.state.selected_index = 0
        elif key == ord('s'):
            self.state.current_mode = UIMode.SEARCH
            self.state.search_query = ""
        return True

    def _handle_filter_input(self, key: int) -> bool:
        """Handle input in filter mode"""
        if key == curses.KEY_DOWN:
            self.state.selected_index = min(self.state.selected_index + 1, 4)
        elif key == curses.KEY_UP:
            self.state.selected_index = max(0, self.state.selected_index - 1)
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if self.state.selected_index == 4:  # Back
                self.state.current_mode = UIMode.CATEGORY
                self.state.current_category = None
            else:
                self._apply_filter(self.state.selected_index + 1)
        elif key == ord('q'):
            self.state.current_mode = UIMode.CATEGORY
        return True

    def _handle_product_input(self, key: int) -> bool:
        """Handle input in product mode with pagination"""
        if not self.state.filtered_products:
            return True
            
        total_products = len(self.state.filtered_products)
        total_pages = (total_products + self.page_size - 1) // self.page_size
        max_select = min(self.page_size, total_products - self.state.current_page * self.page_size) - 1
        
        if key == curses.KEY_DOWN:
            self.state.selected_index = min(self.state.selected_index + 1, max_select)
        elif key == curses.KEY_UP:
            self.state.selected_index = max(0, self.state.selected_index - 1)
        elif key == curses.KEY_RIGHT:
            if (self.state.current_page + 1) < total_pages:
                self.state.current_page += 1
                self.state.selected_index = 0
        elif key == curses.KEY_LEFT:
            if self.state.current_page > 0:
                self.state.current_page -= 1
                self.state.selected_index = 0
        elif key == curses.KEY_ENTER or key in [10, 13]:
            product_idx = self.state.current_page * self.page_size + self.state.selected_index
            if product_idx < total_products:
                self._show_product_detail(self.state.filtered_products[product_idx])
        elif key == ord('b'):
            self.state.current_mode = UIMode.FILTER
            self.state.selected_index = 0
        return True

    def _handle_search_input(self, key: int) -> bool:
        """Handle input in search mode"""
        results = list(
            p for p in self.products
            if re.search(self.state.search_query, p.get('title', ''), re.IGNORECASE)
        )
        max_select = min(self.page_size, len(results)) - 1
        
        if key == curses.KEY_DOWN:
            self.state.selected_index = min(self.state.selected_index + 1, max_select)
        elif key == curses.KEY_UP:
            self.state.selected_index = max(0, self.state.selected_index - 1)
        elif key == curses.KEY_ENTER or key in [10, 13]:
            if results:
                self.state.filtered_products = results
                self.state.current_mode = UIMode.PRODUCT
                self.state.current_page = 0
                self.state.selected_index = 0
        elif key == 27:  # ESC
            self.state.current_mode = UIMode.CATEGORY
        return True

    def _get_current_category_node(self):
        """Get the current category node from the tree"""
        if not self.state.current_category:
            return self.category_tree.root
        
        def find_node(node):
            if node.name == self.state.current_category:
                return node
            for child in node.children:
                found = find_node(child)
                if found:
                    return found
            return None
        
        return find_node(self.category_tree.root)

    def _get_current_categories(self) -> List:
        """Get categories for current level"""
        if not self.state.current_category:
            return self.category_tree.root.children
        else:
            current_node = self._get_current_category_node()
            return current_node.children if current_node else []

    def _apply_filter(self, filter_type: int):
        """Apply selected filter with efficient product filtering"""
        if not self.state.current_category:
            return
            
        category_filters = self.global_filters.get('by_category', {}).get(self.state.current_category, {})
        
        if not self.state.current_filter:
            self.state.current_filter = ProductFilter()
        
        if filter_type == 1:  # Price
            min_price, max_price = category_filters.get('price_range', [0, 0])
            self._show_interactive_price_filter(min_price, max_price)
        elif filter_type == 2:  # Rating
            available_ratings = category_filters.get('ratings', [])
            if available_ratings:
                self._show_interactive_rating_filter(available_ratings)
        elif filter_type == 3:  # Brand
            available_brands = category_filters.get('brands', [])
            if available_brands:
                self._show_interactive_brand_filter(available_brands)
        elif filter_type == 4:  # Features
            available_features = category_filters.get('features', {})
            if available_features:
                self._show_interactive_feature_filter(available_features)
        
        # Apply filters if they exist
        if self.state.current_filter:
            category_products = category_filters.get('products', [])
            self.state.filtered_products = list(self.state.current_filter.apply(category_products))
            self.state.current_mode = UIMode.PRODUCT
            self.state.current_page = 0
            self.state.selected_index = 0

    # ... (resto de los métodos _show_interactive_* permanecen iguales)

def main():
    """Entry point for the UI"""
    try:
        wrapper(lambda stdscr: AmazonProductUI(stdscr).run())
    except Exception as e:
        logger.error(f"UI crashed: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()