#!/usr/bin/env python3
# main.py - Amazon Recommendation System Entry Point (ML COMPLETAMENTE INTEGRADO)

import argparse
import logging
import os
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from dotenv import load_dotenv
import google.generativeai as genai

# 🔥 IMPORTACIONES ML COMPLETAS
from src.core.data.loader import DataLoader
from src.core.rag.advanced.WorkingRAGAgent import WorkingAdvancedRAGAgent, RAGConfig
from src.core.utils.logger import configure_root_logger, get_ml_logger, log_ml_metric, log_ml_event
from src.core.config import settings
from src.core.data.product import Product, AutoProductConfig
from src.core.init import get_system
from src.core.rag.basic.retriever import Retriever
from src.core.data.user_manager import UserManager
from src.core.data.product_reference import ProductReference, create_ml_enhanced_reference
from src.core.rag.advanced.feedback_processor import FeedbackProcessor

# 🔥 NUEVO: Importaciones ML condicionales
try:
    from src.core.data.ml_processor import ProductDataPreprocessor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML processor not available. ML features will be limited.")

# Cargar variables de entorno
load_dotenv()
if settings.GEMINI_API_KEY:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    print("✅ Gemini API configurada")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Logger ML específico
ml_logger = get_ml_logger("main")

# =====================================================
#  INIT SYSTEM ML COMPLETO
# =====================================================
def initialize_system(
    data_dir: Optional[str] = None,
    log_level: Optional[str] = None,
    include_rag_agent: bool = True,
    # 🔥 PARÁMETROS ML MEJORADOS
    ml_enabled: bool = False,
    ml_features: Optional[List[str]] = None,
    ml_batch_size: int = 32,
    use_product_embeddings: bool = False,
    chroma_ml_logging: bool = False,
    track_ml_metrics: bool = True,
    # 🔥 NUEVO: Añadir args como parámetro opcional
    args: Optional[argparse.Namespace] = None
) -> Tuple[List[Product], Optional[WorkingAdvancedRAGAgent], UserManager, Dict[str, Any]]:
    """Initialize system components with complete ML support."""
    
    # 🔥 CORREGIDO: Actualizar settings con argumentos de línea de comandos
    from src.core.config import settings
    if ml_enabled:
        settings.update_ml_settings(
            ml_enabled=True,
            ml_features=ml_features or ["category", "entities"]
        )
    
    # 🔥 NUEVO: Loggear configuración actualizada
    logger.info(f"✅ Configuración ML actualizada:")
    logger.info(f"   - ML_ENABLED: {settings.ML_ENABLED}")
    logger.info(f"   - ML_FEATURES: {settings.ML_FEATURES}")
    # 🔥 NUEVO: Registrar evento ML de inicialización
    log_ml_event(
        "system_initialization_start",
        {
            "ml_enabled": ml_enabled,
            "ml_features": ml_features,
            "use_product_embeddings": use_product_embeddings,
            "timestamp": datetime.now().isoformat()
        }
    )
    
    try:
        start_time = datetime.now()
        
        # 🔥 MEJORADO: Configurar ML globalmente
        ml_config = _configure_ml_system(
            ml_enabled, 
            ml_features, 
            ml_batch_size, 
            use_product_embeddings,
            track_ml_metrics
        )
        
        data_path = Path(data_dir or os.getenv("DATA_DIR") or "./data/raw")
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Created data directory at {data_path}")

        if not any(data_path.glob("*.json")) and not any(data_path.glob("*.jsonl")):
            raise FileNotFoundError(f"No product data found in {data_path}")

        # 🔥 NUEVO: Inicializar FeedbackProcessor con tracking ML
        feedback_processor = None
        if track_ml_metrics:
            try:
                feedback_processor = FeedbackProcessor(
                    feedback_dir="data/feedback",
                    track_ml_metrics=True
                )
                ml_logger.info("✅ FeedbackProcessor with ML tracking initialized")
            except Exception as e:
                ml_logger.warning(f"Could not initialize FeedbackProcessor: {e}")

        # 🔥 MEJORADO: Loader con soporte ML avanzado
        loader = DataLoader(
            raw_dir=data_path,
            processed_dir=settings.PROC_DIR,
            cache_enabled=settings.CACHE_ENABLED,
            ml_enabled=ml_enabled,
            ml_features=ml_features,
            ml_batch_size=ml_batch_size,
        )

        max_products = int(os.getenv("MAX_PRODUCTS_TO_LOAD", "10000"))
        
        # 🔥 NUEVO: Loggear métrica de carga
        log_ml_metric(
            "product_loading_start",
            max_products,
            {"timestamp": datetime.now().isoformat()}
        )
        
        products = loader.load_data()[:max_products]
        
        if not products:
            raise RuntimeError("No products loaded from data directory")
        
        # 🔥 MEJORADO: Estadísticas ML detalladas
        ml_stats = _calculate_ml_statistics(products)
        
        ml_logger.info(f"📦 Loaded {len(products)} products")
        if ml_enabled:
            ml_logger.info(f"🤖 ML Stats: {ml_stats}")
            
            # 🔥 NUEVO: Registrar métricas ML
            log_ml_metric(
                "products_loaded",
                len(products),
                ml_stats
            )

        # 🔥 NUEVO: Retriever con soporte ML mejorado
        retriever = Retriever(
            index_path=settings.VECTOR_INDEX_PATH,
            embedding_model=settings.EMBEDDING_MODEL,
            device=settings.DEVICE,
            use_product_embeddings=use_product_embeddings,
        )

        logger.info("Building vector index...")
        retriever.build_index(products)
        
        # 🔥 NUEVO: Loggear métrica de indexación
        log_ml_metric(
            "index_built",
            (datetime.now() - start_time).total_seconds(),
            {"product_count": len(products), "ml_products": ml_stats.get('ml_processed', 0)}
        )

        # Base system wrapper
        system = get_system()
        
        # 🔥 NUEVO: Actualizar configuración ML del sistema
        if ml_enabled:
            system.update_ml_config({
                'ml_enabled': True,
                'ml_features': ml_features,
                'collaborative_ml_config': {
                    'use_ml_features': True,
                    'ml_weight': settings.ML_WEIGHT,
                    'min_similar_users': settings.MIN_SIMILAR_USERS
                }
            })

        # UserManager para gestión de perfiles
        user_manager = UserManager()

        # 🔥 NUEVO: RAG agent con configuración ML avanzada
        rag_agent = None
        if include_rag_agent:
            try:
                # 🔥 CORREGIDO: Pasar args a _create_rag_config_with_ml
                config = _create_rag_config_with_ml(args if args else type('Args', (), {
                    'ml_features': ml_features or ["category", "entities"]
                })(), use_product_embeddings)
                
                rag_agent = WorkingAdvancedRAGAgent(config=config)
                
                # 🔥 NUEVO: Inyectar dependencias ML
                if hasattr(rag_agent, '_collaborative_filter') and ml_enabled:
                    from src.core.recommendation.collaborative_filter import CollaborativeFilter
                    rag_agent._collaborative_filter = CollaborativeFilter(
                        user_manager=user_manager,
                        use_ml_features=ml_enabled,
                        min_similarity=0.6
                    )
                    ml_logger.info("✅ CollaborativeFilter with ML initialized")
                
                ml_logger.info(f"🧠 WorkingAdvancedRAGAgent initialized with ML: {use_product_embeddings}")
                
                # 🔥 NUEVO: Registrar evento de inicialización exitosa
                log_ml_event(
                    "rag_agent_initialized",
                    {
                        "ml_enabled": ml_enabled,
                        "use_product_embeddings": use_product_embeddings,
                        "config": config.__dict__
                    }
                )
            except Exception as e:
                logger.error(f"❌ Failed to initialize RAG agent: {e}")
                rag_agent = None

        # 🔥 NUEVO: Loggear métrica de inicialización completa
        initialization_time = (datetime.now() - start_time).total_seconds()
        log_ml_metric(
            "system_initialization_complete",
            initialization_time,
            {
                "product_count": len(products),
                "ml_enabled": ml_enabled,
                "ml_features": ml_features,
                "rag_agent_initialized": rag_agent is not None
            }
        )
        
        ml_logger.info(f"🚀 System initialization completed in {initialization_time:.2f}s")

        return products, rag_agent, user_manager, {
            'ml_enabled': ml_enabled,
            'ml_features': ml_features,
            'ml_stats': ml_stats,
            'use_product_embeddings': use_product_embeddings,
            'feedback_processor': feedback_processor,
            'initialization_time': initialization_time
        }

    except Exception as e:
        # 🔥 NUEVO: Loggear error con  el traceback
        import traceback
        error_details = traceback.format_exc()
        
        logger.critical(f"🔥 System initialization failed: {e}")
        logger.critical(f"📋 Error details:\n{error_details}")
        
        # 🔥 NUEVO: Registrar evento de error con detalles completos
        log_ml_event(
            "system_initialization_error",
            {
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": error_details,
                "ml_enabled": ml_enabled,
                "ml_features": ml_features,
                "timestamp": datetime.now().isoformat()
            }
        )
        raise

def _configure_ml_system(
    ml_enabled: bool,
    ml_features: Optional[List[str]],
    ml_batch_size: int,
    use_product_embeddings: bool,
    track_ml_metrics: bool
) -> Dict[str, Any]:
    """Configura el sistema ML globalmente con opciones avanzadas"""
    
    ml_config = {
        'ml_enabled': ml_enabled,
        'ml_features': ml_features or ["category", "entities"],
        'ml_batch_size': ml_batch_size,
        'use_product_embeddings': use_product_embeddings,
        'track_ml_metrics': track_ml_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # 🔥 NUEVO: Configurar logging ML específico
    if ml_enabled:
        from src.core.utils.logger import configure_root_logger
        configure_root_logger(
            level=logging.INFO,
            log_file="logs/app.log",
            enable_ml_logger=True,
            ml_log_file="logs/ml_system.log"
        )
        
        ml_logger.info(f"🤖 ML System configured: {ml_features}")
        ml_logger.info(f"📦 ML Batch size: {ml_batch_size}")
        ml_logger.info(f"🔤 Product embeddings: {'Enabled' if use_product_embeddings else 'Disabled'}")
        ml_logger.info(f"📊 ML Metrics tracking: {'Enabled' if track_ml_metrics else 'Disabled'}")
        
        # Verificar dependencias ML
        if ML_AVAILABLE:
            try:
                from src.core.data.ml_processor import ProductDataPreprocessor
                preprocessor = ProductDataPreprocessor(verbose=True)
                deps = preprocessor.check_dependencies()
                ml_logger.info(f"✅ ML dependencies: {deps}")
                ml_config['ml_dependencies'] = deps
            except Exception as e:
                ml_logger.warning(f"⚠️ ML dependencies check failed: {e}")
                ml_config['ml_dependencies_error'] = str(e)
        else:
            ml_logger.warning("⚠️ ML processor not available. Install: pip install transformers sentence-transformers scikit-learn")
            ml_config['ml_dependencies'] = {'available': False}
    else:
        ml_logger.info("🤖 ML processing disabled - running in basic mode")
    
    return ml_config

def _calculate_ml_statistics(products: List[Product]) -> Dict[str, Any]:
    """Calcula estadísticas ML detalladas de los productos"""
    stats = {
        'total_products': len(products),
        'ml_processed': 0,
        'with_embeddings': 0,
        'with_categories': 0,
        'with_entities': 0,
        'embedding_dimensions': []
    }
    
    for product in products:
        if getattr(product, 'ml_processed', False):
            stats['ml_processed'] += 1
            
        if getattr(product, 'embedding', None):
            stats['with_embeddings'] += 1
            if isinstance(product.embedding, list):
                stats['embedding_dimensions'].append(len(product.embedding))
        
        if getattr(product, 'predicted_category', None):
            stats['with_categories'] += 1
            
        if getattr(product, 'extracted_entities', None):
            stats['with_entities'] += 1
    
    # Calcular estadísticas agregadas
    if stats['embedding_dimensions']:
        stats['avg_embedding_dim'] = sum(stats['embedding_dimensions']) / len(stats['embedding_dimensions'])
        stats['min_embedding_dim'] = min(stats['embedding_dimensions'])
        stats['max_embedding_dim'] = max(stats['embedding_dimensions'])
    
    return stats

def _create_rag_config_with_ml(args, use_product_embeddings: bool) -> RAGConfig:
    """Crea configuración RAG con parámetros ML"""
    # Versión simplificada sin parámetros ML que no existen
    return RAGConfig(
        enable_reranking=True,
        enable_rlhf=True,
        max_retrieved=50,
        max_final=5,
        domain="amazon",
        use_advanced_features=True
    )
# =====================================================
#  PARSER MEJORADO CON ML
# =====================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="🔎 Amazon Product Recommendation System - SISTEMA HÍBRIDO CON ML AVANZADO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🤖 ML FEATURES:
  category     - Zero-shot category classification
  entities     - Named Entity Recognition (brands, models)
  embeddings   - Semantic embeddings with sentence-transformers
  similarity   - Similarity matching with ML
  all          - Enable all ML features

📊 EXAMPLES:
  %(prog)s rag --ml-enabled --ml-features embeddings similarity
  %(prog)s ml --stats --enrich-sample 50
  %(prog)s evaluate --ml-metrics --compare-methods
        """
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=str, default=None,
                       help="Directory containing product data")
    common.add_argument("--log-file", type=Path, default=None,
                       help="Log file path")
    common.add_argument("--log-level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO',
                        help="Logging level")
    common.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose output")
    
    # 🔥 MEJORADO: Argumentos ML
    ml_group = common.add_argument_group('ML Configuration')
    ml_group.add_argument("--ml-enabled", action="store_true", 
                         help="Enable ML processing (categories, NER, embeddings)")
    ml_group.add_argument("--ml-features", nargs="+", 
                         default=["category", "entities"],
                         choices=["category", "entities", "embeddings", "similarity", "all"],
                         help="ML features to enable")
    ml_group.add_argument("--ml-batch-size", type=int, default=32,
                         help="Batch size for ML processing")
    ml_group.add_argument("--ml-weight", type=float, default=0.3,
                         help="Weight for ML scores in hybrid system (0.0-1.0)")
    ml_group.add_argument("--use-product-embeddings", action="store_true",
                         help="Use product's own embeddings when available")
    ml_group.add_argument("--no-ml-tracking", action="store_false", dest="track_ml_metrics",
                         help="Disable ML metrics tracking")
    ml_group.add_argument("--ml-log-file", type=Path, default="logs/ml_system.log",
                         help="ML-specific log file")

    sub = parser.add_subparsers(dest='command', required=True, 
                               title='Available commands',
                               description='Select a command to run')

    # index
    sp = sub.add_parser("index", parents=[common], 
                       help="(Re)build vector index")
    sp.add_argument("--clear-cache", action="store_true",
                   help="Clear cache before indexing")
    sp.add_argument("--force", action="store_true",
                   help="Force reindexing even if index exists")
    sp.add_argument("--batch-size", type=int, default=4000,
                   help="Batch size for indexing")

    # RAG - ACTUALIZADO CON ML
    sp = sub.add_parser("rag", parents=[common], 
                       help="RAG recommendation mode (SISTEMA HÍBRIDO CON ML)")
    sp.add_argument("--ui", action="store_true",
                   help="Enable web UI (if available)")
    sp.add_argument("-k", "--top-k", type=int, default=5,
                   help="Number of recommendations to return")
    sp.add_argument("--user-age", type=int, default=25, 
                   help="User age for personalization")
    sp.add_argument("--user-gender", type=str, 
                   choices=['male', 'female', 'other'], 
                   default='male', 
                   help="User gender for personalization")
    sp.add_argument("--user-country", type=str, 
                   default='Spain', 
                   help="User country for personalization")
    sp.add_argument("--user-id", type=str,
                   help="Specific user ID (overrides auto-generated)")
    sp.add_argument("--show-ml-info", action="store_true",
                   help="Show ML information in responses")

    # 🔥 NUEVO: Comando ML específico mejorado
    sp = sub.add_parser("ml", parents=[common], 
                       help="ML operations and diagnostics")
    ml_sub = sp.add_subparsers(dest='ml_command', 
                              title='ML subcommands',
                              required=True)
    
    # ML stats
    ml_stats = ml_sub.add_parser("stats", help="Show ML statistics")
    ml_stats.add_argument("--detailed", action="store_true",
                         help="Show detailed ML statistics")
    ml_stats.add_argument("--export", type=Path,
                         help="Export statistics to JSON file")
    
    # ML process
    ml_process = ml_sub.add_parser("process", help="Process products with ML")
    ml_process.add_argument("--count", type=int, default=100,
                           help="Number of products to process")
    ml_process.add_argument("--save", type=Path,
                           help="Save processed products to file")
    ml_process.add_argument("--features", nargs="+",
                           default=["category", "entities", "embeddings"],
                           help="Features to apply")
    
    # ML evaluate
    ml_eval = ml_sub.add_parser("evaluate", help="Evaluate ML models")
    ml_eval.add_argument("--test-size", type=float, default=0.2,
                        help="Test set size ratio")
    ml_eval.add_argument("--compare-methods", action="store_true",
                        help="Compare different ML methods")
    ml_eval.add_argument("--output-file", type=Path,
                        help="Output evaluation results to file")
    
    # ML cache
    ml_cache = ml_sub.add_parser("cache", help="Manage ML cache")
    ml_cache.add_argument("--clear", action="store_true",
                         help="Clear ML cache")
    ml_cache.add_argument("--stats", action="store_true",
                         help="Show cache statistics")

    # Comando para gestión de usuarios
    sp = sub.add_parser("users", parents=[common], 
                       help="User management")
    sp.add_argument("--list", action="store_true", 
                   help="List all users")
    sp.add_argument("--stats", action="store_true", 
                   help="Show user statistics")
    sp.add_argument("--export", type=Path,
                   help="Export users to JSON file")

    # 🔥 NUEVO: Comando para evaluar sistema
    sp = sub.add_parser("evaluate", parents=[common],
                       help="System evaluation")
    sp.add_argument("--queries-file", type=Path,
                   help="File with test queries")
    sp.add_argument("--ml-metrics", action="store_true",
                   help="Calculate ML-specific metrics")
    sp.add_argument("--compare", nargs="+",
                   choices=["rag", "collaborative", "hybrid", "ml"],
                   default=["hybrid"],
                   help="Compare different methods")
    sp.add_argument("--output", type=Path,
                   help="Output evaluation results")

    return parser.parse_args()

# =====================================================
#  RAG LOOP MEJORADO CON ML
# =====================================================
def _handle_rag_mode(system, user_manager, args, ml_config: Dict[str, Any] = None):
    """Manejo actualizado del modo RAG con sistema híbrido y ML avanzado"""
    
    # 🔥 NUEVO: Header mejorado con información ML
    print("\n" + "="*60)
    print("🎯 AMAZON HYBRID RECOMMENDATION SYSTEM WITH ML")
    print("="*60)
    
    if ml_config and ml_config.get('ml_enabled'):
        ml_stats = ml_config.get('ml_stats', {})
        print(f"🤖 ML MODE: ENABLED")
        print(f"📊 Features: {', '.join(ml_config.get('ml_features', []))}")
        print(f"📈 Products with ML: {ml_stats.get('ml_processed', 0)}/{ml_stats.get('total_products', 0)}")
        print(f"🔤 Embeddings: {ml_stats.get('with_embeddings', 0)} products")
        if ml_config.get('use_product_embeddings'):
            print(f"⚖️  ML Weight: {settings.ML_WEIGHT}")
    else:
        print(f"🤖 ML MODE: DISABLED (Basic RAG + Collaborative)")
    
    print(f"👤 Personalization: Age, Gender, Country")
    print(f"🔄 Auto-retraining: ENABLED")
    print("="*60 + "\n")
    
    # Crear o cargar perfil de usuario
    user_id = args.user_id or f"cli_user_{args.user_age}_{args.user_gender}_{args.user_country}"
    
    try:
        user_profile = user_manager.get_user_profile(user_id)
        if not user_profile:
            user_profile = user_manager.create_user_profile(
                user_id=user_id,
                age=args.user_age,
                gender=args.user_gender,
                country=args.user_country,
                language="es"
            )
            ml_logger.info(f"👤 Created new user profile: {user_id}")
            log_ml_event("user_profile_created", {
                "user_id": user_id,
                "age": args.user_age,
                "gender": args.user_gender,
                "country": args.user_country
            })
        else:
            ml_logger.info(f"👤 Loaded existing user: {user_id}")
            
        print(f"👤 User: {user_id} (Age: {user_profile.age}, Gender: {user_profile.gender.value}, Country: {user_profile.country})")
        
    except Exception as e:
        logger.error(f"Error creating user profile: {e}")
        user_id = "default"
        print("⚠️ Using default user profile")

    # RAG agent con configuración ML
    config = _create_rag_config_with_ml(args, ml_config.get('use_product_embeddings', False) if ml_config else False)
    agent = WorkingAdvancedRAGAgent(config=config)
    
    # 🔥 NUEVO: Inicializar feedback processor si está disponible
    feedback_processor = ml_config.get('feedback_processor') if ml_config else None

    print(f"\n💡 Type 'exit' to quit | 'stats' for ML stats | 'help' for commands\n")
    
    session_queries = 0
    session_start = datetime.now()
    
    while True:
        try:
            query = input("🧑 You: ").strip()
            
            # 🔥 NUEVO: Comandos especiales
            if query.lower() in {"exit", "quit", "q"}:
                break
            elif query.lower() == "stats":
                _show_session_stats(session_queries, session_start, agent, ml_config)
                continue
            elif query.lower() == "help":
                _show_help_commands()
                continue
            elif query.lower() == "mlinfo":
                _show_ml_info(agent, ml_config)
                continue
            elif not query:
                continue

            session_queries += 1
            
            # 🔥 NUEVO: Registrar evento de query
            log_ml_event("user_query", {
                "user_id": user_id,
                "query": query,
                "session_queries": session_queries,
                "ml_enabled": ml_config.get('ml_enabled', False) if ml_config else False
            })

            print(f"\n{'🚀' if ml_config and ml_config.get('ml_enabled') else '🤖'} Processing with {'ML-enhanced ' if ml_config and ml_config.get('ml_enabled') else ''}HYBRID system...")
            
            # Procesar query con timing
            start_time = datetime.now()
            response = agent.process_query(query, user_id)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 🔥 NUEVO: Loggear métrica de procesamiento
            log_ml_metric(
                "query_processing_time",
                processing_time,
                {
                    "query_length": len(query),
                    "user_id": user_id,
                    "ml_enabled": ml_config.get('ml_enabled', False) if ml_config else False,
                    "products_returned": len(response.products) if hasattr(response, 'products') else 0
                }
            )
            
            print(f"\n🤖 {response.answer}\n")
            
            # 🔥 MEJORADO: Mostrar información ML mejorada
            if args.show_ml_info and hasattr(response, 'products'):
                _show_ml_response_info(response)
            
            print(f"📊 System Info: {len(response.products)} products | "
                  f"Quality: {getattr(response, 'quality_score', 0):.2f} | "
                  f"Time: {processing_time:.2f}s")

            # 🔥 MEJORADO: Sistema de feedback con ML tracking
            _handle_user_feedback(
                query, response, user_id, agent, feedback_processor,
                ml_config.get('ml_enabled', False) if ml_config else False
            )
            
            # 🔥 NUEVO: Verificar reentrenamiento automático con logging ML
            try:
                retrain_info = agent._check_and_retrain()
                if retrain_info and retrain_info.get('retrained', False):
                    ml_logger.info(f"🔄 Auto-retraining completed: {retrain_info}")
                    log_ml_event("auto_retraining_completed", retrain_info)
            except Exception as e:
                ml_logger.debug(f"Auto-retraining check: {e}")
                
        except KeyboardInterrupt:
            print("\n🛑 Session ended")
            break
        except Exception as e:
            logger.error(f"Error in RAG interaction: {e}")
            print("❌ Error processing your request. Please try again.")
            
            # 🔥 NUEVO: Loggear error con contexto ML
            log_ml_event("rag_interaction_error", {
                "error": str(e),
                "user_id": user_id,
                "ml_enabled": ml_config.get('ml_enabled', False) if ml_config else False,
                "query": query if 'query' in locals() else "unknown"
            })

    # 🔥 NUEVO: Loggear estadísticas de sesión
    session_duration = (datetime.now() - session_start).total_seconds()
    log_ml_metric(
        "session_summary",
        session_duration,
        {
            "user_id": user_id,
            "queries_count": session_queries,
            "ml_enabled": ml_config.get('ml_enabled', False) if ml_config else False,
            "avg_time_per_query": session_duration / session_queries if session_queries > 0 else 0
        }
    )

def _show_session_stats(session_queries, session_start, agent, ml_config):
    """Muestra estadísticas de la sesión actual"""
    session_duration = (datetime.now() - session_start).total_seconds()
    
    print(f"\n📈 SESSION STATISTICS:")
    print(f"   Queries: {session_queries}")
    print(f"   Duration: {session_duration:.1f}s")
    print(f"   Avg time per query: {session_duration/session_queries if session_queries > 0 else 0:.1f}s")
    
    if ml_config and ml_config.get('ml_enabled'):
        print(f"\n🤖 ML STATISTICS:")
        print(f"   ML Features: {', '.join(ml_config.get('ml_features', []))}")
        print(f"   ML Products: {ml_config.get('ml_stats', {}).get('ml_processed', 0)}")
        print(f"   ML Embeddings: {ml_config.get('ml_stats', {}).get('with_embeddings', 0)}")
    
    if hasattr(agent, '_collaborative_filter'):
        try:
            cf_stats = agent._collaborative_filter.get_stats()
            print(f"\n🤝 COLLABORATIVE FILTER:")
            print(f"   Similarity checks: {cf_stats.get('similarity_checks', 0)}")
            print(f"   ML enabled: {cf_stats.get('ml_enabled', False)}")
        except:
            pass

def _show_help_commands():
    """Muestra comandos disponibles"""
    print("\n💡 AVAILABLE COMMANDS:")
    print("   'exit', 'quit', 'q' - End session")
    print("   'stats' - Show session statistics")
    print("   'mlinfo' - Show ML system information")
    print("   'help' - Show this help")

def _show_ml_info(agent, ml_config):
    """Muestra información detallada del sistema ML"""
    print("\n🤖 ML SYSTEM INFORMATION:")
    print("="*50)
    
    if ml_config and ml_config.get('ml_enabled'):
        print(f"✅ ML Status: ENABLED")
        print(f"📊 Features: {', '.join(ml_config.get('ml_features', []))}")
        
        stats = ml_config.get('ml_stats', {})
        print(f"\n📈 PRODUCT STATISTICS:")
        print(f"   Total products: {stats.get('total_products', 0)}")
        print(f"   ML processed: {stats.get('ml_processed', 0)} ({stats.get('ml_processed', 0)/stats.get('total_products', 1)*100:.1f}%)")
        print(f"   With embeddings: {stats.get('with_embeddings', 0)}")
        print(f"   With categories: {stats.get('with_categories', 0)}")
        
        if 'avg_embedding_dim' in stats:
            print(f"   Avg embedding dim: {stats['avg_embedding_dim']:.1f}")
        
        # 🔥 NUEVO: Mostrar configuración del sistema
        try:
            system = get_system()
            ml_sys_config = system.get_ml_config()
            print(f"\n⚙️ SYSTEM CONFIGURATION:")
            print(f"   Collaborative ML: {ml_sys_config.get('collaborative_ml_config', {}).get('use_ml_features', False)}")
            print(f"   ML Weight: {ml_sys_config.get('collaborative_ml_config', {}).get('ml_weight', 0.0)}")
        except:
            pass
    else:
        print("❌ ML Status: DISABLED")
        print("💡 Enable with: --ml-enabled --ml-features category entities embeddings")

def _show_ml_response_info(response):
    """Muestra información ML de la respuesta"""
    if hasattr(response, 'products') and response.products:
        print(f"\n🔍 ML ANALYSIS OF TOP PRODUCTS:")
        for i, product in enumerate(response.products[:3], 1):
            if hasattr(product, 'ml_processed') and product.ml_processed:
                print(f"  {i}. {getattr(product, 'title', 'Unknown')}")
                if hasattr(product, 'predicted_category'):
                    print(f"     Category: {product.predicted_category}")
                if hasattr(product, 'ml_confidence'):
                    print(f"     ML Confidence: {product.ml_confidence:.2f}")
                if hasattr(product, 'similarity_score'):
                    print(f"     Similarity: {product.similarity_score:.2f}")
                print()

def _handle_user_feedback(query, response, user_id, agent, feedback_processor, ml_enabled):
    """Maneja el feedback del usuario con tracking ML"""
    while True:
        feedback = input("¿Fue útil esta respuesta? (1-5, 'skip', 'ml'): ").strip().lower()
        
        if feedback in {'1', '2', '3', '4', '5'}:
            rating = int(feedback)
            
            # 🔥 NUEVO: Loggear feedback con contexto ML
            log_ml_event("user_feedback", {
                "user_id": user_id,
                "rating": rating,
                "query": query,
                "ml_enabled": ml_enabled,
                "products_returned": len(response.products) if hasattr(response, 'products') else 0
            })
            
            # Loggear en el agente
            agent.log_feedback(query, response.answer, rating, user_id)
            
            # 🔥 NUEVO: Loggear en feedback processor con métricas ML
            if feedback_processor:
                try:
                    # Extraer métricas ML de la respuesta
                    ml_metrics = {}
                    if hasattr(response, 'ml_scoring_method'):
                        ml_metrics['ml_method'] = response.ml_scoring_method
                    if hasattr(response, 'ml_embeddings_used'):
                        ml_metrics['ml_embeddings_count'] = response.ml_embeddings_used
                    if hasattr(response, 'ml_confidence_score'):
                        ml_metrics['ml_confidence'] = response.ml_confidence_score
                    
                    feedback_processor.save_feedback(
                        query=query,
                        answer=response.answer,
                        rating=rating,
                        extra_meta={
                            'user_id': user_id,
                            'ml_enabled': ml_enabled,
                            'ml_metrics': ml_metrics if ml_metrics else None
                        }
                    )
                except Exception as e:
                    ml_logger.warning(f"Could not save feedback with ML metrics: {e}")
            
            print(f"✅ ¡Gracias por tu feedback! ({'ML system' if ml_enabled else 'System'} aprenderá de esto)")
            break
            
        elif feedback == "skip":
            break
            
        elif feedback == "ml":
            # 🔥 NUEVO: Comando especial para feedback ML
            if ml_enabled:
                print("\n🤖 ML-SPECIFIC FEEDBACK:")
                print("  1 - ML categorization was accurate")
                print("  2 - ML embeddings improved results")
                print("  3 - ML similarity was helpful")
                print("  4 - ML features were not useful")
                print("  5 - Skip ML feedback")
                
                ml_feedback = input("Select (1-5): ").strip()
                if ml_feedback in {'1', '2', '3', '4'}:
                    ml_logger.info(f"User provided ML-specific feedback: {ml_feedback}")
                    log_ml_event("ml_specific_feedback", {
                        "user_id": user_id,
                        "rating": int(ml_feedback),
                        "query": query
                    })
                print("¡Gracias por tu feedback ML!")
            else:
                print("⚠️ ML is not enabled in this session")
            break
            
        else:
            print("❌ Please enter 1-5, 'skip', or 'ml' for ML-specific feedback")

# =====================================================
#  MODO ML MEJORADO
# =====================================================
def _handle_ml_mode(args):
    """Manejo mejorado del comando ML"""
    
    print("\n🤖 ADVANCED ML SYSTEM OPERATIONS")
    print("="*60)
    
    try:
        system = get_system()
        
        if args.ml_command == "stats":
            _handle_ml_stats(args, system)
            
        elif args.ml_command == "process":
            _handle_ml_process(args, system)
            
        elif args.ml_command == "evaluate":
            _handle_ml_evaluate(args, system)
            
        elif args.ml_command == "cache":
            _handle_ml_cache(args, system)
            
    except Exception as e:
        print(f"❌ Error in ML operations: {e}")
        logger.error(f"ML mode error: {e}", exc_info=True)

def _handle_ml_stats(args, system):
    """Maneja estadísticas ML"""
    print("\n📊 ML SYSTEM STATISTICS")
    print("-"*40)
    
    # Obtener configuración ML
    ml_config = system.get_ml_config()
    
    print(f"✅ ML System Status: {'ENABLED' if ml_config.get('ml_enabled', False) else 'DISABLED'}")
    print(f"📊 ML Features: {', '.join(ml_config.get('ml_features', []))}")
    
    # 🔥 NUEVO: Mostrar configuración colaborativa ML
    collab_config = ml_config.get('collaborative_ml_config', {})
    print(f"🤝 Collaborative ML: {'ENABLED' if collab_config.get('use_ml_features', False) else 'DISABLED'}")
    if collab_config.get('use_ml_features'):
        print(f"   • ML Weight: {collab_config.get('ml_weight', 0.0)}")
        print(f"   • Min Similar Users: {collab_config.get('min_similar_users', 3)}")
    
    # 🔥 NUEVO: Mostrar configuración de embeddings
    embed_config = ml_config.get('embedding_config', {})
    print(f"🔤 Embedding Configuration:")
    print(f"   • Sentence Transformers: {'ENABLED' if embed_config.get('use_sentence_transformers', False) else 'DISABLED'}")
    print(f"   • Cache: {'ENABLED' if embed_config.get('cache_embeddings', False) else 'DISABLED'}")
    print(f"   • Model: {embed_config.get('embedding_model', 'N/A')}")
    
    # 🔥 NUEVO: Estadísticas de productos ML
    try:
        products = system.products
        if products:
            ml_stats = _calculate_ml_statistics(products)
            print(f"\n📈 PRODUCT ML STATISTICS:")
            print(f"   • Total products: {ml_stats['total_products']}")
            print(f"   • ML processed: {ml_stats['ml_processed']} ({ml_stats['ml_processed']/ml_stats['total_products']*100:.1f}%)")
            print(f"   • With embeddings: {ml_stats['with_embeddings']}")
            print(f"   • With categories: {ml_stats['with_categories']}")
            print(f"   • With entities: {ml_stats['with_entities']}")
            
            if ml_stats.get('avg_embedding_dim'):
                print(f"   • Avg embedding dimension: {ml_stats['avg_embedding_dim']:.1f}")
            
            # 🔥 NUEVO: Exportar estadísticas
            if args.export:
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'ml_config': ml_config,
                    'ml_stats': ml_stats
                }
                with open(args.export, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                print(f"✅ Statistics exported to {args.export}")
    except Exception as e:
        print(f"⚠️ Could not calculate product statistics: {e}")
    
    # 🔥 NUEVO: Estadísticas detalladas si se solicita
    if args.detailed:
        print(f"\n🔍 DETAILED ML STATISTICS:")
        print(json.dumps(ml_config, indent=2, default=str))

def _handle_ml_process(args, system):
    """Procesa productos con ML"""
    print(f"\n🔧 PROCESSING PRODUCTS WITH ML")
    print("-"*40)
    
    if not ML_AVAILABLE:
        print("❌ ML processor not available. Install: pip install transformers sentence-transformers scikit-learn")
        return
    
    try:
        from src.core.data.ml_processor import ProductDataPreprocessor
        
        # Inicializar preprocesador
        preprocessor = ProductDataPreprocessor(
            verbose=True,
            use_gpu=False,
            embedding_model='sentence-transformers/all-MiniLM-L6-v2'
        )
        
        print(f"✅ ML Preprocessor initialized")
        
        # Cargar productos
        products = system.products[:args.count] if hasattr(system, 'products') else []
        if not products:
            print("❌ No products available to process")
            return
        
        print(f"📥 Processing {len(products)} products with features: {args.features}")
        
        # Convertir a dicts
        product_dicts = []
        for product in products:
            product_dict = {
                'id': getattr(product, 'id', 'unknown'),
                'title': getattr(product, 'title', ''),
                'description': getattr(product, 'description', ''),
                'price': getattr(product, 'price', 0.0),
                'brand': getattr(product, 'brand', ''),
                'categories': getattr(product, 'categories', [])
            }
            product_dicts.append(product_dict)
        
        # Procesar con ML
        processed_dicts = preprocessor.preprocess_batch(product_dicts)
        
        # 🔥 NUEVO: Mostrar resultados
        print(f"\n✅ PROCESSING COMPLETED")
        print(f"📊 Results for {len(processed_dicts)} products:")
        
        # Analizar resultados
        stats = {
            'with_embedding': 0,
            'with_category': 0,
            'with_entities': 0,
            'with_tags': 0
        }
        
        for pd in processed_dicts[:10]:  # Mostrar primeros 10 como ejemplo
            if pd.get('embedding'):
                stats['with_embedding'] += 1
            if pd.get('predicted_category'):
                stats['with_category'] += 1
            if pd.get('extracted_entities'):
                stats['with_entities'] += 1
            if pd.get('tags'):
                stats['with_tags'] += 1
        
        print(f"   • With embeddings: {stats['with_embedding']}")
        print(f"   • With predicted category: {stats['with_category']}")
        print(f"   • With extracted entities: {stats['with_entities']}")
        print(f"   • With ML tags: {stats['with_tags']}")
        
        # 🔥 NUEVO: Mostrar ejemplo
        if processed_dicts:
            example = processed_dicts[0]
            print(f"\n📋 SAMPLE PROCESSED PRODUCT:")
            print(f"   • ID: {example.get('id')}")
            print(f"   • Title: {example.get('title', '')[:50]}...")
            if 'predicted_category' in example:
                print(f"   • Predicted Category: {example['predicted_category']}")
            if 'embedding' in example and example['embedding']:
                print(f"   • Embedding: {len(example['embedding'])} dimensions")
            if 'extracted_entities' in example:
                entities = example['extracted_entities']
                if entities:
                    print(f"   • Extracted Entities: {len(entities)} groups")
        
        # 🔥 NUEVO: Guardar resultados
        if args.save:
            with open(args.save, 'w', encoding='utf-8') as f:
                json.dump(processed_dicts, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved to {args.save}")
            
    except Exception as e:
        print(f"❌ Error processing products: {e}")
        logger.error(f"ML processing error: {e}", exc_info=True)

def _handle_ml_evaluate(args, system):
    """Evalúa modelos ML"""
    print("\n📈 ML MODEL EVALUATION")
    print("-"*40)
    
    print("⚠️ ML evaluation feature coming soon!")
    print("Planned features:")
    print("  • Zero-shot classification accuracy")
    print("  • NER extraction F1 score")
    print("  • Embedding quality metrics")
    print("  • Comparative analysis between methods")
    
    # Placeholder para implementación futura
    if args.output_file:
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'test_size': args.test_size,
            'compare_methods': args.compare_methods,
            'status': 'not_implemented_yet'
        }
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"✅ Evaluation placeholder saved to {args.output_file}")

def _handle_ml_cache(args, system):
    """Maneja cache ML"""
    print("\n🗄️ ML CACHE MANAGEMENT")
    print("-"*40)
    
    if args.clear:
        try:
            # Limpiar caché de embeddings
            system.clear_embedding_cache()
            print("✅ ML cache cleared")
            log_ml_event("ml_cache_cleared", {"timestamp": datetime.now().isoformat()})
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
    
    if args.stats:
        try:
            cache_stats = system.get_embedding_cache_stats()
            print(f"📊 Cache Statistics:")
            print(f"   • Size: {cache_stats.get('size', 0)} entries")
            print(f"   • Memory usage: {cache_stats.get('memory_mb', 0):.1f} MB")
            print(f"   • Hit rate: {cache_stats.get('hit_rate', 0):.1f}%")
        except Exception as e:
            print(f"⚠️ Could not get cache stats: {e}")

# =====================================================
#  MANEJO DE USUARIOS MEJORADO
# =====================================================
def _handle_users_mode(user_manager, args):
    """Manejo mejorado del comando de usuarios"""
    if args.list:
        _list_users(user_manager)
    
    if args.stats:
        _show_user_stats(user_manager)
    
    if args.export:
        _export_users(user_manager, args.export)

def _list_users(user_manager):
    """Lista usuarios"""
    print("\n👥 REGISTERED USERS:")
    print("="*50)
    
    try:
        users_data = user_manager.get_all_users()
        if users_data:
            for user_id, user_data in users_data.items():
                print(f"\n🆔 ID: {user_id}")
                print(f"   📅 Created: {user_data.get('created_at', 'unknown')}")
                print(f"   👤 Demographics: Age {user_data.get('age', '?')}, "
                      f"{user_data.get('gender', 'unknown')}, {user_data.get('country', 'unknown')}")
                print(f"   📊 Activity: {user_data.get('total_sessions', 0)} sessions, "
                      f"{len(user_data.get('feedback_history', []))} feedbacks")
                print(f"   🏷️  Preferences: {', '.join(user_data.get('preferred_categories', ['none']))}")
                print("-" * 30)
        else:
            print("No users found in database.")
    except Exception as e:
        print(f"❌ Error listing users: {e}")

def _show_user_stats(user_manager):
    """Muestra estadísticas de usuarios"""
    print("\n📊 USER STATISTICS:")
    print("="*50)
    
    try:
        stats = user_manager.get_demographic_stats()
        if stats:
            print(f"👥 Total Users: {stats.get('total_users', 0)}")
            print(f"\n📈 AGE DISTRIBUTION:")
            for age_range, count in stats.get('age_distribution', {}).items():
                print(f"   • {age_range}: {count} users")
            
            print(f"\n🚻 GENDER DISTRIBUTION:")
            for gender, count in stats.get('gender_distribution', {}).items():
                print(f"   • {gender}: {count} users")
            
            print(f"\n🌍 COUNTRY DISTRIBUTION (top 5):")
            countries = sorted(stats.get('country_distribution', {}).items(), 
                             key=lambda x: x[1], reverse=True)[:5]
            for country, count in countries:
                print(f"   • {country}: {count} users")
            
            print(f"\n📊 ACTIVITY STATISTICS:")
            print(f"   • Avg sessions per user: {stats.get('avg_sessions_per_user', 0):.1f}")
            print(f"   • Total searches: {stats.get('total_searches', 0)}")
            print(f"   • Total feedbacks: {stats.get('total_feedbacks', 0)}")
            print(f"   • Avg feedback rating: {stats.get('avg_feedback_rating', 0):.1f}/5.0")
        else:
            print("No statistics available.")
    except Exception as e:
        print(f"❌ Error getting user statistics: {e}")

def _export_users(user_manager, export_path):
    """Exporta usuarios a archivo"""
    try:
        users_data = user_manager.get_all_users()
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(users_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Users exported to {export_path} ({len(users_data)} users)")
    except Exception as e:
        print(f"❌ Error exporting users: {e}")

# =====================================================
#  MODO EVALUACIÓN
# =====================================================
def _handle_evaluate_mode(args):
    """Maneja el modo de evaluación"""
    print("\n📊 SYSTEM EVALUATION MODE")
    print("="*60)
    
    # Esta función sería implementada completamente
    # con métricas ML y comparativas
    
    print("⚠️ System evaluation feature coming soon!")
    print("\nPlanned evaluation metrics:")
    print("  • RAG precision and recall")
    print("  • Collaborative filter accuracy")
    print("  • Hybrid system performance")
    print("  • ML-enhanced vs traditional methods")
    print("  • User satisfaction metrics")
    print("  • Response time analysis")
    
    if args.output:
        placeholder_results = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_type': 'placeholder',
            'ml_metrics_enabled': args.ml_metrics,
            'methods_to_compare': args.compare,
            'status': 'not_implemented_yet'
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(placeholder_results, f, indent=2)
        print(f"\n✅ Evaluation placeholder saved to {args.output}")

# =====================================================
#  MAIN MEJORADO
# =====================================================
if __name__ == "__main__":
    # 🔥 NUEVO: Banner de inicio con información ML
    print("╔" + "═"*58 + "╗")
    print("║" + " "*58 + "║")
    print("║  🎯 AMAZON HYBRID RECOMMENDATION SYSTEM WITH ML  ║")
    print("║" + " "*58 + "║")
    print("╠" + "═"*58 + "╣")
    print("║ 🤖 ML Features: Categories, NER, Embeddings, Similarity  ║")
    print("║ 🤝 Hybrid System: RAG 40% + Collaborative 60% + ML       ║")
    print("║ 👤 Personalization: Age, Gender, Country, Preferences     ║")
    print("║ 🔄 Auto-retraining with RLHF Feedback                    ║")
    print("║ 📊 ML Metrics Tracking & Performance Analysis             ║")
    print("╚" + "═"*58 + "╝")
    print()

    # Argumentos
    args = parse_arguments()

    # Logging mejorado
    log_level = "DEBUG" if getattr(args, "verbose", False) else args.log_level
    configure_root_logger(
        level=log_level, 
        log_file=args.log_file,
        enable_ml_logger=True,
        ml_log_file=getattr(args, "ml_log_file", "logs/ml_system.log")
    )

    # 🔥 NUEVO: Registrar inicio del sistema
    log_ml_event("system_start", {
        "command": args.command,
        "ml_enabled": args.ml_enabled,
        "ml_features": args.ml_features,
        "timestamp": datetime.now().isoformat()
    })

    try:
        # 🔥 CORREGIDO: Pasar args a initialize_system
        products, rag_agent, user_manager, ml_config = initialize_system(
            data_dir=args.data_dir,
            ml_enabled=args.ml_enabled,
            ml_features=args.ml_features,
            ml_batch_size=args.ml_batch_size,
            use_product_embeddings=args.use_product_embeddings,
            chroma_ml_logging=getattr(args, 'chroma_ml_logging', False),
            track_ml_metrics=getattr(args, 'track_ml_metrics', True),
            args=args  # 🔥 NUEVO: Pasar args
        )

        if args.command == "index":
            print("🔨 Index building completed during initialization.")
            print(f"✅ Index contains {len(products)} products")
            if ml_config.get('ml_enabled'):
                print(f"🤖 {ml_config.get('ml_stats', {}).get('ml_processed', 0)} products processed with ML")

        elif args.command == "rag":
            _handle_rag_mode(get_system(), user_manager, args, ml_config)
            
        elif args.command == "ml":
            _handle_ml_mode(args)
            
        elif args.command == "users":
            _handle_users_mode(user_manager, args)
            
        elif args.command == "evaluate":
            _handle_evaluate_mode(args)

    except Exception as e:
        logger.error(f"System failed: {str(e)}", exc_info=True)
        
        # 🔥 NUEVO: Registrar error del sistema
        log_ml_event("system_error", {
            "error": str(e),
            "command": args.command,
            "ml_enabled": getattr(args, 'ml_enabled', False),
            "timestamp": datetime.now().isoformat()
        })
        
        sys.exit(1)
    
    # 🔥 NUEVO: Registrar finalización exitosa
    log_ml_event("system_shutdown", {
        "command": args.command,
        "exit_status": "success",
        "timestamp": datetime.now().isoformat()
    })