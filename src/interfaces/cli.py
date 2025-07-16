from __future__ import annotations

# src/interfaces/cli.py

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from src.core.config import settings
from src.core.data.loader import DataLoader
from src.core.data.product import Product
from src.core.rag.basic.retriever import Retriever
from src.core.rag.advanced.agent import RAGAgent
from src.core.category_search.category_tree import CategoryTree
from src.core.utils.logger import configure_root_logger
from src.core.utils.parsers import parse_binary_score
# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------
class AmazonRecommendationCLI:
    def __init__(self, products, category_tree, rag_agent):
        self.products = products
        self.category_tree = category_tree
        self.rag_agent = rag_agent

    def run(self):
        """Main entry point for CLI interface"""
        main()  # Calls your existing main function


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Amazon Product Recommendation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- rag -------------------------------------------------------
    rag = sub.add_parser("rag", help="interactive Q&A")
    rag.add_argument("-k", "--top-k", type=int, default=5)
    rag.add_argument("--no-feedback", action="store_true")
    rag.add_argument("--disable-tqdm", action="store_true", help="disable progress bars")

    # ---- category --------------------------------------------------
    cat = sub.add_parser("category", help="browse by category")
    cat.add_argument("-c", "--category", type=str, help="start category")
    cat.add_argument("--disable-tqdm", action="store_true", help="disable progress bars")

    # ---- index -----------------------------------------------------
    idx = sub.add_parser("index", help="(re)build vector-store")
    idx.add_argument("--clear-cache", action="store_true")
    idx.add_argument("--disable-tqdm", action="store_true", help="disable progress bars")

    # ---- common ----------------------------------------------------
    for cmd in (rag, cat, idx):
        cmd.add_argument("-v", "--verbose", action="store_true")
        cmd.add_argument("--log-file", type=Path)

    args = parser.parse_args(argv)

    # ----------------------------------------------------------------
    # logging
    # ----------------------------------------------------------------
    configure_root_logger(
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=args.log_file,
        module_levels={"urllib3": logging.WARNING, "transformers": logging.WARNING},
    )

    # ----------------------------------------------------------------
    # shared objects
    # ----------------------------------------------------------------
    loader = DataLoader(
        raw_dir=settings.RAW_DIR,
        processed_dir=settings.PROC_DIR,
        cache_enabled=settings.CACHE_ENABLED,
        disable_tqdm=getattr(args, 'disable_tqdm', False),  # Pass the option
    )
    products: List[Product] = loader.load_data(use_cache=True)

    if not products:
        logging.error("No products loaded – aborting.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # dispatch
    # ----------------------------------------------------------------
    if args.command == "index":
        _run_index_mode(loader, clear_cache=args.clear_cache)

    elif args.command == "rag":
        _run_rag_mode(loader, products, k=args.top_k, feedback=not args.no_feedback)

    elif args.command == "category":
        _run_category_mode(products, start=args.category)

    logging.info("Done.")

# ------------------------------------------------------------------
# Mode implementations
# ------------------------------------------------------------------
def _run_rag_mode(loader: DataLoader, products: List[Product], k: int, feedback: bool) -> None:
    """Interactive Q&A loop."""
    logger = logging.getLogger("rag")

    # Check if the index exists
    index_path = settings.VEC_DIR / settings.INDEX_NAME
    if not index_path.exists():
        logger.info("Index not found. Building index...")
        _run_index_mode(loader, clear_cache=False)

    # Build retriever & agent using settings
    retriever = Retriever(
        index_path=settings.VEC_DIR / settings.INDEX_NAME,
        embedding_model=settings.EMBEDDING_MODEL,
        vectorstore_type=settings.VECTOR_BACKEND,
        device=settings.DEVICE
    )
    agent = RAGAgent(
        products=products,
        lora_checkpoint=settings.RLHF_CHECKPOINT,
    )

    print("\n=== Amazon RAG mode ===\nType 'exit' to quit.\n")
    while True:
        query = input("🧑  You: ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue

        answer = agent.ask(query)
        print(f"\n🤖  Bot: {answer}\n")

        if feedback:
            rating = input("Was this helpful? (y/n): ").strip().lower()
            score = parse_binary_score(rating)
            logger.info("Feedback: %s -> %s", query, score.name)

    logger.info("Leaving RAG mode.")

def _run_category_mode(products: List[Product], start: Optional[str]) -> None:
    """Interactive category explorer."""
    tree = CategoryTree(products)
    tree.build_tree()

    node = tree.root
    print("\n=== Category Explorer ===\nPress Ctrl+C twice to exit.\n")

    try:
        while True:
            if node.children:
                print(f"\n📂 {node.name} ({len(node.products)} items)")
                for i, child in enumerate(node.children, 1):
                    print(f"  {i}. {child.name} ({len(child.products)} items)")
                print("  0. Back" if node.parent else "  0. Exit")

                choice = input("Select: ").strip()
                if choice == "0":
                    node = node.parent or tree.root
                    continue
                if choice.isdigit() and 1 <= int(choice) <= len(node.children):
                    node = node.children[int(choice) - 1]
                    continue
                print("Invalid choice.")
            else:
                print(f"\n🛍️  {node.name} – {len(node.products)} products")
                for i, p in enumerate(node.products[:20], 1):
                    print(f"{i:2}. {p.title} – ${p.price}")
                input("\nPress Enter to go back…")
                node = node.parent or tree.root
    except KeyboardInterrupt:
        print("\nLeaving category mode.")

def _run_index_mode(loader: DataLoader, *, clear_cache: bool) -> None:
    """(Re)build vector-store and cache."""
    if clear_cache:
        deleted = loader.clear_cache()
        print(f"🗑️  Cleared {deleted} cache files.")

    # Force re-processing (cache disabled)
    products = loader.load_data(use_cache=False)
    print(f"✅ Re-indexed {len(products)} products.")

    # Build the vector index
    from src.core.rag.basic.retriever import Retriever
    
    # First create the index directory if it doesn't exist
    index_path = settings.VEC_DIR / settings.INDEX_NAME
    index_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize retriever and build index
    retriever = Retriever(
        index_path=index_path,
        embedding_model=settings.EMBEDDING_MODEL,
        vectorstore_type=settings.VECTOR_BACKEND,
        device=settings.DEVICE
    )
    
    # Explicitly build the index with the products
    retriever.build_index(products)
    print("✅ Successfully built vector index.")

# ------------------------------------------------------------------
# Script entry
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()