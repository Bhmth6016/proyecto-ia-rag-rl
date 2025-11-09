from __future__ import annotations
# src/intercaes/cli.py
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from src.core.config import settings
from src.core.rag.basic.retriever import Retriever
from src.core.rag.advanced.rlhf import RAGAgent
from src.core.utils.logger import configure_root_logger
from src.core.utils.parsers import parse_binary_score
from src.core.init import get_system

def main(argv: Optional[List[str]] = None) -> None:
    print("DEBUG - CLI iniciado")  # <-- Agrega esto
    parser = argparse.ArgumentParser(
        description="Amazon Product Recommendation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- Command definitions ----
    rag = sub.add_parser("rag", help="Interactive Q&A mode")
    rag.add_argument("-k", "--top-k", type=int, default=5)
    rag.add_argument("--no-feedback", action="store_true")
    
    index = sub.add_parser("index", help="(Re)build vector index")
    index.add_argument("--clear-cache", action="store_true")
    index.add_argument("--force", action="store_true", help="Force reindexing")

    category = sub.add_parser("category", help="Browse by category")
    category.add_argument("-c", "--category", type=str, help="Starting category")

    # Common arguments
    for p in [rag, index, category]:
        p.add_argument("-v", "--verbose", action="store_true")
        p.add_argument("--log-file", type=Path)

    args = parser.parse_args(argv)

    # Configure logging
    configure_root_logger(
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=args.log_file,
        module_levels={"urllib3": logging.WARNING}
    )

    try:
        system = get_system()
        
        if args.command == "index":
            _handle_index_mode(system, args.clear_cache, args.force)
        elif args.command == "rag":
            _handle_rag_mode(system, args.top_k, not args.no_feedback)
        elif args.command == "category":
            _handle_category_mode(system, args.category)

    except Exception as e:
        logging.error(f"Failed: {str(e)}")
        sys.exit(1)

def _handle_index_mode(system, args):
    """Handle index building with new arguments"""
    if args.clear_cache:
        system.loader.clear_cache()
        print("🗑️ Cleared product cache")
    
    if system.retriever.index_exists() and not args.force:
        print("ℹ️ Index exists. Use --force to rebuild")
        return

    print("🛠️ Building index...")
    try:
        system.retriever.build_index(
            system.products, 
            force_rebuild=args.force,
            batch_size=args.batch_size
        )
        print(f"✅ Index built with {len(system.products)} products")
    except Exception as e:
        print(f"❌ Failed to build index: {str(e)}")
        sys.exit(1)

def _handle_rag_mode(system, top_k: int, feedback: bool) -> None:
    """Handle RAG interaction"""
    agent = RAGAgent(
        products=system.products,
        enable_translation=True
    )

    print("\n=== Amazon RAG ===\nType 'exit' to quit\n")
    while True:
        try:
            query = input("🧑 You: ").strip()
            if query.lower() in {"exit", "quit", "q"}:
                break

            answer = agent.ask(query)
            print(f"\n🤖 {answer}\n")

            if feedback:
                rating = input("Helpful? (y/n): ").strip().lower()
                score = parse_binary_score(rating)
                logging.info(f"Feedback|{query}|{score.name}")

        except KeyboardInterrupt:
            print("\n🛑 Session ended")
            break

def _handle_category_mode(system, start_category: Optional[str]) -> None:
    """Handle category browsing"""
    tree = system.category_tree
    node = tree.find_category(start_category) if start_category else tree.root

    print("\n=== Category Explorer ===\n")
    while node:
        if node.children:
            print(f"\n📂 {node.name} ({len(node.products)} items)")
            for i, child in enumerate(node.children, 1):
                print(f"  {i}. {child.name} ({len(child.products)} items)")
            print("  0. " + ("Back" if node.parent else "Exit"))

            choice = input("Select: ").strip()
            if choice == "0":
                node = node.parent or None
            elif choice.isdigit() and 1 <= int(choice) <= len(node.children):
                node = node.children[int(choice)-1]
        else:
            print(f"\n🛍️ {node.name} ({len(node.products)} products)")
            for i, p in enumerate(node.products[:20], 1):
                print(f"{i:2}. {p.title[:50]} - ${p.price or '?'}")
            input("\nPress Enter to go back...")
            node = node.parent

if __name__ == "__main__":
    main()