from __future__ import annotations
# src/interfaces/cli.py - ACTUALIZADO
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import json

from src.core.config import settings
from src.core.rag.basic.retriever import Retriever
from src.core.rag.advanced.WorkingRAGAgent import WorkingAdvancedRAGAgent, RAGConfig
from src.core.utils.logger import configure_root_logger
from src.core.utils.parsers import parse_binary_score
from src.core.init import get_system
from src.core.data.user_manager import UserManager

def main(argv: Optional[List[str]] = None) -> None:
    print("🚀 CLI SISTEMA HÍBRIDO INICIADO")
    parser = argparse.ArgumentParser(
        description="Amazon Product Recommendation CLI - SISTEMA HÍBRIDO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- Command definitions ACTUALIZADAS ----
    rag = sub.add_parser("rag", help="Interactive Q&A mode (SISTEMA HÍBRIDO)")
    rag.add_argument("-k", "--top-k", type=int, default=5)
    rag.add_argument("--no-feedback", action="store_true")
    rag.add_argument("--user-age", type=int, default=25, help="User age")
    rag.add_argument("--user-gender", type=str, choices=['male', 'female', 'other'], 
                    default='male', help="User gender")
    rag.add_argument("--user-country", type=str, default='Spain', help="User country")
    
    index = sub.add_parser("index", help="(Re)build vector index")
    index.add_argument("--clear-cache", action="store_true")
    index.add_argument("--force", action="store_true", help="Force reindexing")

    # 🔥 NUEVO: Comando de usuarios
    users = sub.add_parser("users", help="User management")
    users.add_argument("--list", action="store_true", help="List all users")
    users.add_argument("--stats", action="store_true", help="Show user statistics")
    users.add_argument("--create", action="store_true", help="Create new user")
    users.add_argument("--age", type=int, help="Age for new user")
    users.add_argument("--gender", type=str, choices=['male', 'female', 'other'], help="Gender for new user")
    users.add_argument("--country", type=str, help="Country for new user")

    # Common arguments
    for p in [rag, index, users]:
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
        user_manager = UserManager()  # 🔥 NUEVO: UserManager
        
        if args.command == "index":
            _handle_index_mode(system, args)
        elif args.command == "rag":
            _handle_rag_mode(system, user_manager, args)
        elif args.command == "users":
            _handle_users_mode(user_manager, args)

    except Exception as e:
        logging.error(f"Failed: {str(e)}")
        sys.exit(1)

def _handle_index_mode(system, args):
    """Handle index building"""
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
            batch_size=getattr(args, 'batch_size', 4000)
        )
        print(f"✅ Index built with {len(system.products)} products")
    except Exception as e:
        print(f"❌ Failed to build index: {str(e)}")
        sys.exit(1)

def _handle_rag_mode(system, user_manager, args):
    """Handle RAG interaction ACTUALIZADO"""
    # 🔥 NUEVO: Configuración del sistema híbrido
    config = RAGConfig(
        max_retrieved=args.top_k * 3,
        max_final=args.top_k,
        domain="amazon"
    )
    
    agent = WorkingAdvancedRAGAgent(config=config)
    
    # 🔥 NUEVO: Gestión de usuario con datos demográficos
    user_id = f"cli_{args.user_age}_{args.user_gender}_{args.user_country}"
    
    try:
        user_profile = user_manager.get_user_profile(user_id)
        if not user_profile:
            user_profile = user_manager.create_user_profile(
                age=args.user_age,
                gender=args.user_gender,
                country=args.user_country
            )
            print(f"👤 Created user: {user_id}")
        else:
            print(f"👤 Loaded user: {user_id}")
    except Exception as e:
        print(f"⚠️ User error: {e}, using default")
        user_id = "default"

    print(f"\n=== Amazon HYBRID RAG ===")
    print(f"User: {user_id} | Weights: 60% Collaborative / 40% RAG")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            query = input("🧑 You: ").strip()
            if query.lower() in {"exit", "quit", "q"}:
                break

            # 🔥 NUEVO: Usar process_query del sistema híbrido
            response = agent.process_query(query, user_id)
            print(f"\n🤖 {response.answer}\n")
            print(f"📊 System: {len(response.products)} products | Quality: {response.quality_score:.2f}")

            if not args.no_feedback:
                rating = input("Helpful? (1-5, skip): ").strip().lower()
                if rating in {'1', '2', '3', '4', '5'}:
                    agent.log_feedback(query, response.answer, int(rating), user_id)
                    print("📝 Feedback saved for hybrid learning")
                elif rating != "skip":
                    print("⚠️ Please enter 1-5 or 'skip'")

        except KeyboardInterrupt:
            print("\n🛑 Session ended")
            break

def _handle_users_mode(user_manager, args):
    """Handle user management"""
    if args.list:
        _list_users(user_manager)
    elif args.stats:
        _show_user_stats(user_manager)
    elif args.create:
        _create_user(user_manager, args)
    else:
        print("ℹ️ Use --list, --stats, or --create with --age/--gender/--country")

def _list_users(user_manager):
    """List all users"""
    print("\n👥 REGISTERED USERS:")
    print("=" * 50)
    
    users_dir = Path("data/users")
    if users_dir.exists():
        user_files = list(users_dir.glob("*.json"))
        for user_file in user_files:
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    user_data = json.load(f)
                print(f"ID: {user_data['user_id']}")
                print(f"  Age: {user_data['age']} | Gender: {user_data['gender']} | Country: {user_data['country']}")
                print(f"  Feedbacks: {len(user_data.get('feedback_history', []))}")
                print("-" * 40)
            except Exception as e:
                print(f"Error reading {user_file}: {e}")
    else:
        print("No users directory")

def _show_user_stats(user_manager):
    """Show user statistics"""
    print("\n📊 USER STATISTICS:")
    print("=" * 50)
    
    stats = user_manager.get_demographic_stats()
    if stats:
        print(f"Total Users: {stats['total_users']}")
        print(f"Age Distribution: {stats['age_distribution']}")
        print(f"Gender Distribution: {stats['gender_distribution']}")
        print(f"Country Distribution: {stats['country_distribution']}")
        print(f"Total Feedbacks: {stats['total_feedbacks']}")
    else:
        print("No statistics available")

def _create_user(user_manager, args):
    """Create new user"""
    if not all([args.age, args.gender, args.country]):
        print("❌ Please provide --age, --gender, and --country")
        return
    
    try:
        user_profile = user_manager.create_user_profile(
            age=args.age,
            gender=args.gender,
            country=args.country
        )
        print(f"✅ User created: {user_profile.user_id}")
        print(f"   Age: {user_profile.age}, Gender: {user_profile.gender.value}, Country: {user_profile.country}")
    except Exception as e:
        print(f"❌ Error creating user: {e}")

if __name__ == "__main__":
    main()