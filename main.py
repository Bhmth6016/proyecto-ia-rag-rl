#!/usr/bin/env python3
# main.py - Amazon Recommendation System Entry Point (VERSIÓN CORREGIDA)

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from src.core.data.user_manager import UserManager
from dotenv import load_dotenv
import google.generativeai as genai
# 🔥 IMPORTACIONES ACTUALIZADAS CON ML Y CLI
from src.core.data.loader import DataLoader
from src.core.rag.advanced.WorkingRAGAgent import WorkingAdvancedRAGAgent, RAGConfig
from src.core.utils.logger import configure_root_logger, get_ml_logger, log_ml_metric
from src.core.config import settings
from src.core.data.product import Product
from src.core.init import get_system
from src.core.data.user_manager import UserManager
from src.core.rag.advanced.feedback_processor import FeedbackProcessor

# 🔥 NUEVO: Importar CLI integrado
from src.interfaces.cli import main as cli_main
# Cargar variables de entorno
load_dotenv()

# Configuración básica de logging primero
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.core.config import settings
    
    if settings.GEMINI_API_KEY:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        print("✅ Gemini API configurada")
    else:
        print("⚠️ GEMINI_API_KEY no configurada")
except Exception as e:
    print(f"⚠️ Error cargando configuración: {e}")

# =====================================================
#  PARSER CORREGIDO
# =====================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="🔎 Amazon Product Recommendation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Subparsers para diferentes modos
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Comando CLI (ejecuta el CLI completo)
    cli_parser = subparsers.add_parser('cli', help='Run the full integrated CLI')
    cli_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    cli_parser.add_argument("--log-file", type=Path, help="Log file path")
    
    # Comando RAG individual
    rag_parser = subparsers.add_parser('rag', help='RAG interactive mode')
    rag_parser.add_argument("-v", "--verbose", action="store_true")
    rag_parser.add_argument("--log-file", type=Path)
    rag_parser.add_argument("-k", "--top-k", type=int, default=5)
    rag_parser.add_argument("--user-age", type=int, default=25)
    rag_parser.add_argument("--user-gender", choices=['male', 'female', 'other'], default='male')
    rag_parser.add_argument("--user-country", type=str, default='Spain')
    rag_parser.add_argument("--ml-enabled", action="store_true", help="Enable ML features")
    
    # Comando index
    index_parser = subparsers.add_parser('index', help='Build/rebuild index')
    index_parser.add_argument("-v", "--verbose", action="store_true")
    index_parser.add_argument("--log-file", type=Path)
    index_parser.add_argument("--force", action="store_true", help="Force rebuild")
    index_parser.add_argument("--clear-cache", action="store_true", help="Clear cache")
    
    # Comando users
    users_parser = subparsers.add_parser('users', help='User management')
    users_parser.add_argument("-v", "--verbose", action="store_true")
    users_parser.add_argument("--log-file", type=Path)
    users_parser.add_argument("--list", action="store_true", help="List users")
    users_parser.add_argument("--stats", action="store_true", help="Show stats")
    
    # Comando ml
    ml_parser = subparsers.add_parser('ml', help='ML operations')
    ml_parser.add_argument("-v", "--verbose", action="store_true")
    ml_parser.add_argument("--log-file", type=Path)
    ml_parser.add_argument("--stats", action="store_true", help="Show ML stats")
    ml_parser.add_argument("--features", nargs="+", default=["category"], 
                          help="ML features to enable")
    
    return parser.parse_args()

# =====================================================
#  FUNCIONES DE MANEJO DE COMANDOS
# =====================================================
def handle_cli_mode(args):
    """Maneja el modo CLI completo"""
    print("🚀 Iniciando CLI completo del sistema...")
    print("=" * 60)
    
    try:
        # Importar y ejecutar el CLI original
        from src.interfaces.cli import main as cli_main
        
        # Preparar argumentos para el CLI original
        cli_args = []
        if args.verbose:
            cli_args.append("-v")
        if args.log_file:
            cli_args.extend(["--log-file", str(args.log_file)])
        
        # Ejecutar el CLI original
        cli_main(cli_args)
        
    except ImportError as e:
        print(f"❌ Error: No se pudo importar el CLI: {e}")
        print("\n🔧 Posibles soluciones:")
        print("1. Verifica que src/interfaces/cli.py exista")
        print("2. Asegúrate de tener todas las dependencias instaladas")
        sys.exit(1)

def handle_rag_mode(args):
    """Maneja el modo RAG individual"""
    print("🎯 Modo RAG interactivo")
    print("=" * 40)
    
    try:
        # Importaciones necesarias
        from src.core.init import get_system
        from src.core.data.user_manager import UserManager
        from src.core.rag.advanced.WorkingRAGAgent import WorkingAdvancedRAGAgent, RAGConfig
        
        # Inicializar sistema
        system = get_system()
        user_manager = UserManager(data_dir="data/users")
        
        # Crear usuario
        user_id = f"rag_user_{args.user_age}_{args.user_gender}_{args.user_country}"
        
        try:
            user_profile = user_manager.get_user_profile(user_id)
            if not user_profile:
                user_profile = user_manager.create_user_profile(
                    age=args.user_age,
                    gender=args.user_gender,
                    country=args.user_country
                )
                print(f"👤 Usuario creado: {user_id}")
            else:
                print(f"👤 Usuario cargado: {user_id}")
        except:
            user_id = "default"
            print("⚠️ Usando usuario por defecto")
        
        # Configurar RAG
        config = RAGConfig(
            max_retrieved=args.top_k * 3,
            max_final=args.top_k,
            use_ml_embeddings=args.ml_enabled
        )
        
        agent = WorkingAdvancedRAGAgent(config=config)
        
        print(f"\n💬 Sistema RAG listo (k={args.top_k})")
        print("Escribe 'exit' para salir\n")
        
        # Bucle interactivo
        while True:
            try:
                query = input("Tú: ").strip()
                if query.lower() in ('exit', 'quit', 'q'):
                    break
                
                response = agent.process_query(query, user_id)
                print(f"\n🤖 {response.answer}\n")
                
                # Feedback simple
                feedback = input("¿Útil? (1-5 o enter para saltar): ").strip()
                if feedback in ('1', '2', '3', '4', '5'):
                    agent.log_feedback(query, response.answer, int(feedback), user_id)
                    print("✅ Feedback guardado")
                    
            except KeyboardInterrupt:
                print("\n🛑 Sesión terminada")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except ImportError as e:
        print(f"❌ Error importando módulos: {e}")
        sys.exit(1)

def handle_index_mode(args):
    """Maneja el modo index"""
    print("🔨 Construyendo índice vectorial...")
    
    try:
        from src.core.init import get_system
        
        system = get_system()
        retriever = system.retriever
        
        if args.clear_cache:
            if hasattr(system.loader, 'clear_cache'):
                system.loader.clear_cache()
                print("🗑️ Cache limpiada")
        
        if retriever.index_exists() and not args.force:
            print("ℹ️ Índice ya existe. Usa --force para reconstruir")
            return
        
        print("🛠️ Construyendo índice...")
        retriever.build_index(system.products, force_rebuild=args.force)
        print(f"✅ Índice construido con {len(system.products)} productos")
        
    except Exception as e:
        print(f"❌ Error construyendo índice: {e}")
        sys.exit(1)

def handle_users_mode(args):
    """Maneja el modo usuarios"""
    try:
        from src.core.data.user_manager import UserManager
        
        user_manager = UserManager()
        
        if args.list:
            print("👥 Usuarios registrados:")
            print("=" * 40)
            
            users_dir = Path("data/users")
            if users_dir.exists():
                user_files = list(users_dir.glob("*.json"))
                for user_file in user_files:
                    try:
                        with open(user_file, 'r', encoding='utf-8') as f:
                            user_data = json.load(f)
                        print(f"ID: {user_data.get('user_id', 'N/A')}")
                        print(f"  Edad: {user_data.get('age', 'N/A')}")
                        print(f"  Género: {user_data.get('gender', 'N/A')}")
                        print(f"  País: {user_data.get('country', 'N/A')}")
                        print("-" * 30)
                    except:
                        continue
                print(f"Total: {len(user_files)} usuarios")
            else:
                print("No hay directorio de usuarios")
                
        if args.stats:
            print("📊 Estadísticas de usuarios:")
            print("=" * 40)
            
            try:
                stats = user_manager.get_demographic_stats()
                if stats:
                    print(f"Total usuarios: {stats.get('total_users', 0)}")
                    print(f"Distribución edad: {stats.get('age_distribution', {})}")
                    print(f"Distribución género: {stats.get('gender_distribution', {})}")
                else:
                    print("No hay estadísticas disponibles")
            except:
                print("No se pudieron obtener estadísticas")
                
    except Exception as e:
        print(f"❌ Error: {e}")

def handle_ml_mode(args):
    """Maneja el modo ML"""
    print("🤖 Operaciones ML")
    print("=" * 40)
    
    try:
        from src.core.init import get_system
        
        system = get_system()
        
        if args.stats:
            print("📊 Estadísticas ML:")
            print(f"  ML habilitado: {system.ml_enabled}")
            print(f"  Features ML: {system.ml_features}")
            
            # Contar productos con ML
            if hasattr(system, '_products') and system._products:
                ml_count = sum(1 for p in system._products if getattr(p, 'ml_processed', False))
                embed_count = sum(1 for p in system._products if hasattr(p, 'embedding') and p.embedding)
                print(f"  Productos con ML: {ml_count}/{len(system._products)}")
                print(f"  Productos con embeddings: {embed_count}/{len(system._products)}")
        
        print(f"🔧 Features solicitadas: {args.features}")
        
        if system.ml_enabled:
            print("✅ Sistema ML activado")
        else:
            print("⚠️ Sistema ML no activado (configura ML_ENABLED=true en .env)")
            
    except Exception as e:
        print(f"❌ Error ML: {e}")

# =====================================================
#  MAIN CORREGIDO
# =====================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎯 SISTEMA DE RECOMENDACIÓN AMAZON")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Configurar logging basado en verbose
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if hasattr(args, 'log_file') and args.log_file:
        # Añadir file handler si se especifica
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
    
    # Ejecutar comando correspondiente
    if args.command == 'cli':
        handle_cli_mode(args)
    elif args.command == 'rag':
        handle_rag_mode(args)
    elif args.command == 'index':
        handle_index_mode(args)
    elif args.command == 'users':
        handle_users_mode(args)
    elif args.command == 'ml':
        handle_ml_mode(args)
    elif args.command is None:
        print("❌ Debes especificar un comando")
        print("\nComandos disponibles:")
        print("  cli    - CLI completo integrado")
        print("  rag    - Modo RAG interactivo")
        print("  index  - Construir índice")
        print("  users  - Gestión de usuarios")
        print("  ml     - Operaciones ML")
        print("\nEjemplo: python main.py cli")
        sys.exit(1)
    else:
        print(f"❌ Comando desconocido: {args.command}")
        sys.exit(1)