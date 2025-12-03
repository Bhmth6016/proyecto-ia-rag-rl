#!/usr/bin/env python3
# main.py - Amazon Recommendation System Entry Point (ACTUALIZADO CON ML)

import argparse
import logging
import os
import sys
import json  # 🔥 AÑADIDO para manejo de usuarios
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv

import google.generativeai as genai

# 🔥 IMPORTACIONES ACTUALIZADAS CON ML
from src.core.data.loader import DataLoader
from src.core.rag.advanced.WorkingRAGAgent import WorkingAdvancedRAGAgent, RAGConfig
from src.core.utils.logger import configure_root_logger
from src.core.config import settings
from src.core.data.product import Product, AutoProductConfig
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
#  INIT SYSTEM ACTUALIZADO CON ML
# =====================================================
def initialize_system(
    data_dir: Optional[str] = None,
    log_level: Optional[str] = None,
    include_rag_agent: bool = True,
    # 🔥 NUEVO: Parámetros ML
    ml_enabled: bool = False,
    ml_features: Optional[List[str]] = None,
    ml_batch_size: int = 32,
    # 🔥 NUEVO: Configuración Chroma con ML
    use_product_embeddings: bool = False,
    chroma_ml_logging: bool = False
):
    """Initialize system components with ML support."""
    try:
        # 🔥 NUEVO: Configurar ML globalmente
        _configure_ml_system(ml_enabled, ml_features, ml_batch_size)
        
        data_path = Path(data_dir or os.getenv("DATA_DIR") or "./data/raw")
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Created data directory at {data_path}")

        if not any(data_path.glob("*.json")) and not any(data_path.glob("*.jsonl")):
            raise FileNotFoundError(f"No product data found in {data_path}")

        # 🔥 NUEVO: Loader con configuración ML
        loader = DataLoader(
            raw_dir=data_path,
            processed_dir=settings.PROC_DIR,
            cache_enabled=settings.CACHE_ENABLED,
            # 🔥 NUEVO: Configuración ML para loader
            ml_enabled=ml_enabled,
            ml_features=ml_features,
            ml_batch_size=ml_batch_size
        )

        max_products = int(os.getenv("MAX_PRODUCTS_TO_LOAD", "10000"))
        products = loader.load_data()[:max_products]
        
        if not products:
            raise RuntimeError("No products loaded from data directory")
        
        # 🔥 NUEVO: Estadísticas ML
        ml_products_count = sum(1 for p in products if getattr(p, 'ml_processed', False))
        ml_embeddings_count = sum(1 for p in products if getattr(p, 'embedding', None))
        
        logger.info(f"📦 Loaded {len(products)} products")
        if ml_enabled:
            logger.info(f"🤖 ML Stats: {ml_products_count} products with ML | {ml_embeddings_count} with embeddings")

        # 🔥 NUEVO: Retriever con configuración ML
        retriever = Retriever(
            index_path=settings.VECTOR_INDEX_PATH,
            embedding_model=settings.EMBEDDING_MODEL,
            device=settings.DEVICE,
            # 🔥 NUEVO: Habilitar embeddings ML en retriever
            use_product_embeddings=use_product_embeddings
        )

        logger.info("Building vector index...")
        retriever.build_index(products)

        # Base system wrapper
        system = get_system()

        # UserManager para gestión de perfiles
        user_manager = UserManager()

        # 🔥 NUEVO: RAG agent con configuración ML mejorada
        rag_agent = None
        if include_rag_agent:
            try:
                config = RAGConfig(
                    enable_reranking=True,
                    enable_rlhf=True,
                    max_retrieved=50,
                    max_final=5,
                    domain="amazon",
                    use_advanced_features=True,
                    # 🔥 NUEVO: Configuración ML para RAG agent
                    use_ml_embeddings=use_product_embeddings,
                    ml_embedding_weight=0.3 if use_product_embeddings else 0.0
                )
                rag_agent = WorkingAdvancedRAGAgent(config=config)
                logger.info(f"🧠 WorkingAdvancedRAGAgent initialized with ML: {use_product_embeddings}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize RAG agent: {e}")
                rag_agent = None

        return products, rag_agent, user_manager, {
            'ml_enabled': ml_enabled,
            'ml_features': ml_features,
            'ml_stats': {
                'total_products': len(products),
                'ml_processed': ml_products_count,
                'with_embeddings': ml_embeddings_count,
                'use_product_embeddings': use_product_embeddings
            }
        }

    except Exception as e:
        logger.critical(f"🔥 System initialization failed: {e}", exc_info=True)
        raise

def _configure_ml_system(
    ml_enabled: bool,
    ml_features: Optional[List[str]],
    ml_batch_size: int
):
    """Configura el sistema ML globalmente"""
    # Configurar Product para usar ML
    Product.configure_ml(
        enabled=ml_enabled,
        features=ml_features or ["category", "entities"],
        categories=AutoProductConfig.DEFAULT_CATEGORIES
    )
    
    if ml_enabled:
        logger.info(f"🤖 ML System configured: {ml_features}")
        logger.info(f"📦 ML Batch size: {ml_batch_size}")
        
        # Verificar dependencias ML
        try:
            from src.core.data.ml_processor import ProductDataPreprocessor
            logger.info("✅ ML dependencies available")
        except ImportError as e:
            logger.warning(f"⚠️ ML dependencies not fully available: {e}")
            logger.warning("ML features may be limited")
    else:
        logger.info("🤖 ML processing disabled")

# =====================================================
#  PARSER ACTUALIZADO CON ML
# =====================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="🔎 Amazon Product Recommendation System - SISTEMA HÍBRIDO CON ML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=str, default=None)
    common.add_argument("--log-file", type=Path, default=None)
    common.add_argument("--log-level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO')
    common.add_argument("-v", "--verbose", action="store_true")
    
    # 🔥 NUEVO: Argumentos ML
    common.add_argument("--ml-enabled", action="store_true", 
                       help="Enable ML processing (categories, NER, embeddings)")
    common.add_argument("--ml-features", nargs="+", 
                       default=["category", "entities"],
                       help="ML features to enable: category, entities, tags, embedding")
    common.add_argument("--ml-batch-size", type=int, default=32,
                       help="Batch size for ML processing")
    common.add_argument("--use-product-embeddings", action="store_true",
                       help="Use product's own embeddings when available")
    common.add_argument("--chroma-ml-logging", action="store_true",
                       help="Enable ML logging for Chroma builder")

    sub = parser.add_subparsers(dest='command', required=True)

    # index
    sp = sub.add_parser("index", parents=[common], help="(Re)build vector index")
    sp.add_argument("--clear-cache", action="store_true")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--batch-size", type=int, default=4000)

    # RAG - ACTUALIZADO CON ML
    sp = sub.add_parser("rag", parents=[common], 
                       help="RAG recommendation mode (SISTEMA HÍBRIDO CON ML)")
    sp.add_argument("--ui", action="store_true")
    sp.add_argument("-k", "--top-k", type=int, default=5)
    sp.add_argument("--user-age", type=int, default=25, 
                   help="User age for personalization")
    sp.add_argument("--user-gender", type=str, 
                   choices=['male', 'female', 'other'], 
                   default='male', 
                   help="User gender for personalization")
    sp.add_argument("--user-country", type=str, 
                   default='Spain', 
                   help="User country for personalization")

    # 🔥 NUEVO: Comando ML específico
    sp = sub.add_parser("ml", parents=[common], help="ML operations and diagnostics")
    sp.add_argument("--stats", action="store_true", help="Show ML statistics")
    sp.add_argument("--metrics", action="store_true", help="Show ML metrics")
    sp.add_argument("--fit-tfidf", action="store_true", 
                   help="Fit TF-IDF model with sample data")
    sp.add_argument("--enrich-sample", type=int, default=10,
                   help="Enrich sample products with ML")

    # Comando para gestión de usuarios
    sp = sub.add_parser("users", parents=[common], help="User management")
    sp.add_argument("--list", action="store_true", help="List all users")
    sp.add_argument("--stats", action="store_true", help="Show user statistics")

    return parser.parse_args()

# =====================================================
#  RAG LOOP ACTUALIZADO CON ML
# =====================================================
def _handle_rag_mode(system, user_manager, args, ml_config: Dict[str, Any] = None):
    """Manejo actualizado del modo RAG con sistema híbrido y ML"""
    print("🛠️ Preparing HYBRID RAG system with ML...")
    
    # 🔥 NUEVO: Mostrar configuración ML
    if ml_config and ml_config.get('ml_enabled'):
        print(f"🤖 ML Features: {', '.join(ml_config.get('ml_features', []))}")
        print(f"📊 ML Stats: {ml_config.get('ml_stats', {}).get('ml_processed', 0)} products processed")
    
    # Crear o cargar perfil de usuario con datos demográficos
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

    # 🔥 NUEVO: RAG agent con configuración ML
    config = RAGConfig(
        max_retrieved=args.top_k * 3,
        max_final=args.top_k,
        use_ml_embeddings=args.use_product_embeddings,
        ml_embedding_weight=0.3 if args.use_product_embeddings else 0.0
    )
    agent = WorkingAdvancedRAGAgent(config=config)

    print(f"\n=== Amazon HYBRID RAG System ===")
    print(f"User: {user_id}")
    print(f"Mode: {'ML Enhanced' if args.ml_enabled else 'Standard'}")
    print(f"Embeddings: {'Product Embeddings' if args.use_product_embeddings else 'Chroma Embeddings'}")
    print("Type 'exit' to quit\n")
    
    while True:
        try:
            query = input("🧑 You: ").strip()
            if query.lower() in {"exit", "quit", "q"}:
                break

            print("\n🤖 Processing with HYBRID ML system...")
            
            # Procesar query
            response = agent.process_query(query, user_id)
            
            print(f"\n🤖 {response.answer}\n")
            
            # 🔥 NUEVO: Mostrar información ML en la respuesta
            if hasattr(response, 'ml_embeddings_used'):
                print(f"📊 ML Info: {response.ml_embeddings_used} embeddings used | Method: {response.ml_scoring_method}")
            
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
            
            # Verificar reentrenamiento automático
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
#  MODO ML - OPERACIONES Y DIAGNÓSTICOS
# =====================================================
def _handle_ml_mode(args):
    """Manejo del comando ML - diagnósticos y operaciones"""
    print("\n🤖 ML SYSTEM DIAGNOSTICS")
    print("=" * 50)
    
    try:
        from src.core.data.product import MLProductEnricher
        
        # Obtener métricas del sistema ML
        metrics = MLProductEnricher.get_metrics()
        
        print(f"✅ ML System Status: {'Enabled' if metrics.get('ml_enabled', False) else 'Disabled'}")
        print(f"📊 Preprocessor Loaded: {metrics.get('preprocessor_loaded', False)}")
        
        if metrics.get('ml_enabled'):
            print(f"\n📈 ML MODELS:")
            models_info = metrics.get('models_loaded', {})
            for model_name, status in models_info.items():
                print(f"  • {model_name}: {'✅ Loaded' if status else '❌ Not loaded'}")
            
            print(f"\n🔧 ML FEATURES:")
            print(f"  • Embedding Cache: {metrics.get('embedding_cache_size', 0)} entries")
            print(f"  • TF-IDF Fitted: {metrics.get('tfidf_fitted', False)}")
            
            # 🔥 NUEVO: Mostrar configuración actual
            from src.core.data.product import Product
            ml_config = Product._ml_config
            print(f"\n⚙️ CURRENT ML CONFIGURATION:")
            print(f"  • Enabled: {ml_config.get('enabled', False)}")
            print(f"  • Features: {', '.join(ml_config.get('features', []))}")
            print(f"  • Categories: {len(ml_config.get('categories', []))} categories")
        
        # 🔥 NUEVO: Operaciones ML
        if args.fit_tfidf:
            print("\n🔧 Fitting TF-IDF model...")
            try:
                # Cargar datos de muestra
                from src.core.data.loader import DataLoader
                loader = DataLoader()
                sample_products = loader.load_data()[:100]  # 100 productos para entrenamiento
                
                # Extraer descripciones
                descriptions = []
                for product in sample_products:
                    if hasattr(product, 'description') and product.description:
                        descriptions.append(product.description)
                
                if descriptions:
                    success = MLProductEnricher.fit_tfidf(descriptions)
                    if success:
                        print("✅ TF-IDF model fitted successfully")
                    else:
                        print("❌ Failed to fit TF-IDF model")
                else:
                    print("⚠️ No descriptions available for TF-IDF training")
                    
            except Exception as e:
                print(f"❌ Error fitting TF-IDF: {e}")
        
        if args.enrich_sample > 0:
            print(f"\n🔧 Enriching {args.enrich_sample} sample products with ML...")
            try:
                from src.core.data.loader import DataLoader
                loader = DataLoader(ml_enabled=True, ml_features=args.ml_features)
                sample_products = loader.load_data()[:args.enrich_sample]
                
                enriched_count = 0
                for product in sample_products:
                    if hasattr(product, 'ml_processed') and product.ml_processed:
                        enriched_count += 1
                
                print(f"✅ {enriched_count}/{args.enrich_sample} products enriched with ML")
                
                # Mostrar ejemplo
                if sample_products and len(sample_products) > 0:
                    sample = sample_products[0]
                    print(f"\n📋 SAMPLE ENRICHED PRODUCT:")
                    print(f"  • Title: {getattr(sample, 'title', 'N/A')}")
                    print(f"  • ML Processed: {getattr(sample, 'ml_processed', False)}")
                    if hasattr(sample, 'predicted_category'):
                        print(f"  • Predicted Category: {sample.predicted_category}")
                    if hasattr(sample, 'ml_tags') and sample.ml_tags:
                        print(f"  • ML Tags: {', '.join(sample.ml_tags[:3])}")
                    if hasattr(sample, 'embedding') and sample.embedding:
                        print(f"  • Embedding: {len(sample.embedding)} dimensions")
                        
            except Exception as e:
                print(f"❌ Error enriching sample: {e}")
    
    except ImportError as e:
        print(f"❌ ML system not available: {e}")
    except Exception as e:
        print(f"❌ Error getting ML diagnostics: {e}")

# =====================================================
#  MANEJO DE USUARIOS
# =====================================================
def _handle_users_mode(user_manager, args):
    """Manejo del comando de usuarios"""
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
                            user_data = json.load(f)
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
#  MAIN ACTUALIZADO CON ML
# =====================================================
if __name__ == "__main__":
    print("🎯 SISTEMA HÍBRIDO RAG + RL + ML - AMAZON RECOMMENDATION")
    print("=" * 60)
    print("✅ Collaborative Filter: 60% | RAG Traditional: 40%")
    print("✅ User Demographics: Age, Gender, Country")
    print("✅ Automatic RLHF Retraining")
    print("✅ ML Features: Categories, NER, Embeddings")
    print("=" * 60)

    # Argumentos
    args = parse_arguments()

    # Logging
    log_level = "DEBUG" if getattr(args, "verbose", False) else args.log_level
    configure_root_logger(level=log_level, log_file=args.log_file)

    try:
        # 🔥 NUEVO: Inicializar con configuración ML
        products, rag_agent, user_manager, ml_config = initialize_system(
            data_dir=args.data_dir,
            ml_enabled=args.ml_enabled,
            ml_features=args.ml_features,
            ml_batch_size=args.ml_batch_size,
            use_product_embeddings=args.use_product_embeddings,
            chroma_ml_logging=args.chroma_ml_logging
        )

        if args.command == "index":
            print("🔨 Index rebuilding not yet fully implemented here.")

        elif args.command == "rag":
            _handle_rag_mode(get_system(), user_manager, args, ml_config)
            
        elif args.command == "ml":
            _handle_ml_mode(args)
            
        elif args.command == "users":
            _handle_users_mode(user_manager, args)

    except Exception as e:
        logging.error(f"System failed: {str(e)}")
        sys.exit(1)