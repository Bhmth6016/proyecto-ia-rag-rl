#!/usr/bin/env python3
# main.py - Amazon Recommendation System Entry Point (ACTUALIZADO)

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

from dotenv import load_dotenv

import google.generativeai as genai

# 🔥 IMPORTACIONES ACTUALIZADAS
from src.core.data.loader import DataLoader
from src.core.rag.advanced.WorkingRAGAgent import WorkingAdvancedRAGAgent, RAGConfig
from src.core.utils.logger import configure_root_logger
from src.core.config import settings
from src.core.data.product import Product
from src.core.init import get_system
from src.core.rag.basic.retriever import Retriever
from src.core.data.user_manager import UserManager

# Cargar variables de entorno
load_dotenv()
if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    print("✅ Gemini API configurada")

# Logger
configure_root_logger(
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE", "logs/system.log")
)
logger = logging.getLogger(__name__)

# =====================================================
#  INIT SYSTEM ACTUALIZADO
# =====================================================
def initialize_system(
    data_dir: Optional[str] = None,
    log_level: Optional[str] = None,
    include_rag_agent: bool = True
):
    """Initialize system components with better error handling."""
    try:
        data_path = Path(data_dir or os.getenv("DATA_DIR") or "./data/raw")
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Created data directory at {data_path}")

        if not any(data_path.glob("*.json")) and not any(data_path.glob("*.jsonl")):
            raise FileNotFoundError(f"No product data found in {data_path}")

        # Load products
        loader = DataLoader(
            raw_dir=data_path,
            processed_dir=settings.PROC_DIR,
            cache_enabled=settings.CACHE_ENABLED
        )

        max_products = int(os.getenv("MAX_PRODUCTS_TO_LOAD", "10000"))
        products = loader.load_data()[:max_products]
        if not products:
            raise RuntimeError("No products loaded from data directory")
        logger.info(f"📦 Loaded {len(products)} products")

        # Retriever
        retriever = Retriever(
            index_path=settings.VECTOR_INDEX_PATH,
            embedding_model=settings.EMBEDDING_MODEL,
            device=settings.DEVICE,
        )

        logger.info("Building vector index...")
        retriever.build_index(products)

        # Base system wrapper
        system = get_system()

        # 🔥 NUEVO: UserManager para gestión de perfiles
        user_manager = UserManager()

        # 🔥 NUEVO: RAG agent actualizado (WorkingAdvancedRAGAgent)
        rag_agent = None
        if include_rag_agent:
            try:
                config = RAGConfig(
                    enable_reranking=True,
                    enable_rlhf=True,
                    max_retrieved=50,
                    max_final=5,
                    domain="amazon",
                    use_advanced_features=True
                )
                rag_agent = WorkingAdvancedRAGAgent(config=config)
                logger.info("🧠 WorkingAdvancedRAGAgent initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize RAG agent: {e}")
                rag_agent = None

        return products, rag_agent, user_manager

    except Exception as e:
        logger.critical(f"🔥 System initialization failed: {e}", exc_info=True)
        raise

# =====================================================
#  PARSER ACTUALIZADO
# =====================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="🔎 Amazon Product Recommendation System - SISTEMA HÍBRIDO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=str, default=None)
    common.add_argument("--log-file", type=Path, default=None)
    common.add_argument("--log-level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO')
    common.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest='command', required=True)

    # index
    sp = sub.add_parser("index", parents=[common], help="(Re)build vector index")
    sp.add_argument("--clear-cache", action="store_true")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--batch-size", type=int, default=4000)

    # RAG - ACTUALIZADO
    sp = sub.add_parser("rag", parents=[common], help="RAG recommendation mode (SISTEMA HÍBRIDO)")
    sp.add_argument("--ui", action="store_true")
    sp.add_argument("-k", "--top-k", type=int, default=5)
    sp.add_argument("--user-age", type=int, default=25, help="User age for personalization")
    sp.add_argument("--user-gender", type=str, choices=['male', 'female', 'other'], 
                   default='male', help="User gender for personalization")
    sp.add_argument("--user-country", type=str, default='Spain', 
                   help="User country for personalization")

    # 🔥 NUEVO: Comando para gestión de usuarios
    sp = sub.add_parser("users", parents=[common], help="User management")
    sp.add_argument("--list", action="store_true", help="List all users")
    sp.add_argument("--stats", action="store_true", help="Show user statistics")

    return parser.parse_args()

# =====================================================
#  RAG LOOP ACTUALIZADO
# =====================================================
def _handle_rag_mode(system, user_manager, args):
    """Manejo actualizado del modo RAG con sistema híbrido"""
    print("🛠️ Preparing HYBRID RAG system...")
    
    # 🔥 NUEVO: Crear o cargar perfil de usuario con datos demográficos
    user_id = f"cli_user_{args.user_age}_{args.user_gender}_{args.user_country}"
    
    try:
        # Intentar cargar usuario existente
        user_profile = user_manager.get_user_profile(user_id)
        if not user_profile:
            # Crear nuevo usuario con datos demográficos
            user_profile = user_manager.create_user_profile(
                age=args.user_age,
                gender=args.user_gender,
                country=args.user_country,
                language="es"
            )
            print(f"👤 Created new user profile: {user_id}")
        else:
            print(f"👤 Loaded existing user: {user_id}")
            
        print(f"   Age: {user_profile.age}, Gender: {user_profile.gender.value}, Country: {user_profile.country}")
        
    except Exception as e:
        logger.error(f"Error creating user profile: {e}")
        # Usar perfil por defecto
        user_id = "default"
        print("⚠️ Using default user profile")

    # 🔥 NUEVO: Usar WorkingAdvancedRAGAgent
    config = RAGConfig(
        max_retrieved=args.top_k * 3,  # Recuperar más para mejor filtrado
        max_final=args.top_k
    )
    agent = WorkingAdvancedRAGAgent(config=config)

    print(f"\n=== Amazon HYBRID RAG System ===")
    print(f"User: {user_id} | Weights: 60% Collaborative / 40% RAG")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            query = input("🧑 You: ").strip()
            if query.lower() in {"exit", "quit", "q"}:
                break

            print("\n🤖 Processing your request with HYBRID system...")
            
            # 🔥 NUEVO: Usar process_query en lugar de ask
            response = agent.process_query(query, user_id)
            
            print(f"\n🤖 {response.answer}\n")
            print(f"📊 System Info: {len(response.products)} products | Quality: {response.quality_score:.2f}")

            # Sistema de feedback mejorado
            while True:
                feedback = input("¿Fue útil esta respuesta? (1-5, 'skip'): ").strip().lower()
                if feedback in {'1', '2', '3', '4', '5'}:
                    rating = int(feedback)
                    agent.log_feedback(query, response.answer, rating, user_id)
                    print("¡Gracias por tu feedback! (Sistema híbrido aprenderá de esto)")
                    break
                elif feedback == "skip":
                    break
                else:
                    print("Por favor ingresa 1-5 o 'skip'")
            
            # 🔥 NUEVO: Verificar reentrenamiento automático
            try:
                agent._check_and_retrain()
            except Exception as e:
                logger.debug(f"Reentrenamiento automático: {e}")
                
        except KeyboardInterrupt:
            print("\n🛑 Session ended")
            break
        except Exception as e:
            logger.error(f"Error in RAG interaction: {e}")
            print("❌ Error procesando tu solicitud. Intenta de nuevo.")

# =====================================================
#  MANEJO DE USUARIOS
# =====================================================
def _handle_users_mode(user_manager, args):
    """Manejo del comando de usuarios"""
    import json  # 🔥 AÑADIR ESTA LÍNEA
    
    if args.list:
        print("\n👥 LISTA DE USUARIOS REGISTRADOS:")
        print("=" * 50)
        
        users_dir = Path("data/users")
        if users_dir.exists():
            user_files = list(users_dir.glob("*.json"))
            if user_files:
                for user_file in user_files:
                    try:
                        with open(user_file, 'r', encoding='utf-8') as f:
                            user_data = json.load(f)  # ✅ Ahora json está definido
                        print(f"ID: {user_data['user_id']}")
                        print(f"  Age: {user_data['age']} | Gender: {user_data['gender']} | Country: {user_data['country']}")
                        print(f"  Created: {user_data['created_at'][:10]}")
                        print(f"  Sessions: {user_data.get('total_sessions', 1)}")
                        print(f"  Feedbacks: {len(user_data.get('feedback_history', []))}")
                        print("-" * 30)
                    except Exception as e:
                        print(f"Error reading {user_file}: {e}")
            else:
                print("No users found.")
        else:
            print("Users directory doesn't exist.")
    
    if args.stats:
        print("\n📊 ESTADÍSTICAS DE USUARIOS:")
        print("=" * 50)
        
        stats = user_manager.get_demographic_stats()
        if stats:
            print(f"Total Users: {stats['total_users']}")
            print(f"Age Distribution: {stats['age_distribution']}")
            print(f"Gender Distribution: {stats['gender_distribution']}")
            print(f"Country Distribution: {stats['country_distribution']}")
            print(f"Average Sessions per User: {stats['avg_sessions_per_user']:.1f}")
            print(f"Total Searches: {stats['total_searches']}")
            print(f"Total Feedbacks: {stats['total_feedbacks']}")
        else:
            print("No statistics available.")

# =====================================================
#  MAIN ACTUALIZADO
# =====================================================
if __name__ == "__main__":
    print("🎯 SISTEMA HÍBRIDO RAG + RL - AMAZON RECOMMENDATION")
    print("=" * 60)
    print("✅ Collaborative Filter: 60% | RAG Traditional: 40%")
    print("✅ User Demographics: Age, Gender, Country")
    print("✅ Automatic RLHF Retraining")
    print("=" * 60)

    # Argumentos
    args = parse_arguments()

    # Logging
    log_level = "DEBUG" if getattr(args, "verbose", False) else args.log_level
    configure_root_logger(level=log_level, log_file=args.log_file)

    try:
        # 🔥 NUEVO: Inicializar con user_manager
        products, rag_agent, user_manager = initialize_system()

        if args.command == "index":
            print("🔨 Index rebuilding not yet fully implemented here.")

        elif args.command == "rag":
            _handle_rag_mode(get_system(), user_manager, args)
            
        elif args.command == "users":
            _handle_users_mode(user_manager, args)

    except Exception as e:
        logging.error(f"System failed: {str(e)}")
        sys.exit(1)