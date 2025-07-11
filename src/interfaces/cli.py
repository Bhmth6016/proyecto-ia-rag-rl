# src/interfaces/cli.py
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum, auto
from textwrap import dedent

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from src.core.utils.logger import get_logger
from src.core.rag.basic.retriever import Retriever
from src.core.rag.advanced.agent import AdvancedRAGAgent
from src.core.category_search.category_tree import CategoryTree
from src.core.data.loader import DataLoader
from src.core.utils.parsers import parse_binary_score, BinaryScore

logger = get_logger(__name__)

class CLIMode(Enum):
    RAG = auto()
    CATEGORY = auto()
    INDEX = auto()

class AmazonRecommendationCLI:
    def __init__(self, products: List[Dict], category_tree: CategoryTree, llm: ChatGoogleGenerativeAI, memory: ConversationBufferMemory):
        self.parser = self._setup_parser()
        self.args = None
        self.products = products
        self.category_tree = category_tree
        self.llm = llm
        self.memory = memory
        self.loader = DataLoader()
        self.agent = None  # Se inicializará en _setup_rag_mode

    def _setup_parser(self) -> argparse.ArgumentParser:
        """Configure the command line argument parser"""
        parser = argparse.ArgumentParser(
            description="Amazon Recommendation System CLI",
            formatter_class=argparse.RawTextHelpFormatter
        )
        
        subparsers = parser.add_subparsers(dest='command', required=True)

        # RAG Mode
        rag_parser = subparsers.add_parser('rag', help='Run in RAG question-answering mode')
        rag_parser.add_argument('-k', '--top-k', type=int, default=5, help='Number of results to return')
        rag_parser.add_argument('--no-feedback', action='store_true', help='Disable feedback collection')

        # Category Mode
        category_parser = subparsers.add_parser('category', help='Browse products by category')
        category_parser.add_argument('-c', '--category', type=str, help='Start with specific category')

        # Index Mode
        index_parser = subparsers.add_parser('index', help='Reindex data')
        index_parser.add_argument('--clear-cache', action='store_true', help='Clear existing cache before indexing')

        # Common arguments
        for p in [rag_parser, category_parser, index_parser]:
            p.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
            p.add_argument('--log-file', type=str, help='Path to log file')

        return parser

    def _setup_rag_mode(self):
        """Initialize components for RAG mode"""
        logger.info("Initializing RAG system...")
        
        retriever = Retriever(
            index_path="data/processed/chroma_db",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            vectorstore_type="chroma"
        )
        
        self.agent = AdvancedRAGAgent(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            feedback_processor=None
        )
        
        logger.info("RAG system ready")

    def _run_rag_mode(self):
        """Interactive RAG question-answering loop"""
        print(dedent("""
        ===========================================
        Amazon Product Recommendation - RAG Mode
        ===========================================
        Ask questions about products and get recommendations
        Type 'exit' to quit
        """))
        
        try:
            while True:
                query = input("\nYour question: ").strip()
                if query.lower() in ('exit', 'quit'):
                    break
                
                if not query:
                    print("Please enter a question")
                    continue
                    
                # Get response from RAG agent using invoke() instead of query()
                result = self.agent.invoke(query)
                print(f"\nResponse: {result['response']}")
                
                # Collect feedback if enabled
                if not self.args.no_feedback:
                    try:
                        feedback = input("\nWas this helpful? (y/n): ").strip().lower()
                        score = parse_binary_score(feedback)
                        if score == BinaryScore.POSITIVE:
                            logger.info("Positive feedback for: %s", query)
                        elif score == BinaryScore.NEGATIVE:
                            logger.warning("Negative feedback for: %s", query)
                    except Exception as e:
                        logger.error("Feedback processing error: %s", str(e))
                    
        except KeyboardInterrupt:
            logger.info("\nExiting RAG mode")
        except Exception as e:
            logger.error(f"Error in RAG mode: {str(e)}")
            print(f"\nAn error occurred: {str(e)}")


    def run(self):
        """Main entry point for the CLI"""
        self.args = self.parser.parse_args()

        # Configure logging
        log_level = logging.DEBUG if self.args.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                *(logging.FileHandler(self.args.log_file,) if self.args.log_file else ())
            ]
        )
        
        try:
            if self.args.command == 'rag':
                self._setup_rag_mode()
                self._run_rag_mode()
            elif self.args.command == 'category':
                self._setup_category_mode()
                self._run_category_mode()
            elif self.args.command == 'index':
                self._run_index_mode()
        except KeyboardInterrupt:
            logger.info("\nOperation cancelled by user")
        except Exception as e:
            logger.error(f"CLI Error: {str(e)}", exc_info=True)
            raise


    def _setup_category_mode(self):
        """Initialize components for category browsing"""
        logger.info("Loading product categories...")
        products = self.loader.load_data()
        self.category_tree = CategoryTree(products)
        self.category_tree.build_tree()
        logger.info(f"Loaded {len(products)} products across categories")

    def _run_category_mode(self):
        """Interactive category browsing loop"""
        print(dedent("""
        ===========================================
        Amazon Product Recommendation - Category Mode
        ===========================================
        Browse products by category hierarchy
        Press Ctrl+C twice to exit
        """))
        
        current_node = self.category_tree.root
        exit_confirmed = False
        
        try:
            while not exit_confirmed:
                try:
                    if not current_node.children:
                        self._show_products(current_node.products)
                        current_node = current_node.parent
                        continue
                    
                    print(f"\nCurrent category: {current_node.name}")
                    print("\nSubcategories:")
                    for i, child in enumerate(current_node.children, 1):
                        print(f"{i}. {child.name} ({len(child.products)} products)")
                    print("\n0. Back to parent" if current_node.parent else "0. Exit")
                    
                    choice = input("\nSelect category (0-{}): ".format(len(current_node.children)))
                    
                    if choice == '0':
                        if current_node.parent:
                            current_node = current_node.parent
                        else:
                            break
                    elif choice.isdigit() and 1 <= int(choice) <= len(current_node.children):
                        current_node = current_node.children[int(choice)-1]
                    else:
                        print("Invalid selection")
                        
                except KeyboardInterrupt:
                    print("\nPress Ctrl+C again to confirm exit or Enter to continue...")
                    try:
                        input()
                        exit_confirmed = False
                    except KeyboardInterrupt:
                        exit_confirmed = True
                    except Exception:
                        continue
                        
        except Exception as e:
            logger.error(f"Fatal error in category mode: {str(e)}")
            print(f"\nA fatal error occurred: {str(e)}")

    def _show_products(self, products: List[Dict[str, Any]]):
        """Display products with filtering options"""
        # Simplified product display - could be enhanced with filters
        print(f"\nShowing {len(products)} products:")
        for i, product in enumerate(products[:20], 1):
            print(f"{i}. {product.get('title', 'Unknown')}")
            print(f"   Price: ${product.get('price', 'N/A')} | Rating: {product.get('average_rating', 'N/A')}")
        
        input("\nPress Enter to continue...")

    def _run_index_mode(self):
        """Handle data reindexing"""
        print("\nReindexing product data...")
        if self.args.clear_cache:
            cleared = self.loader.clear_cache()
            print(f"Cleared {cleared} cache files")
        
        products = self.loader.load_data(use_cache=False)
        print(f"Successfully indexed {len(products)} products")

def main():
    """Entry point for CLI execution"""
    cli = AmazonRecommendationCLI()
    cli.run()

if __name__ == "__main__":
    main()