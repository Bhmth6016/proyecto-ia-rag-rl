#!/usr/bin/env python3
# main.py - Sistema de Recomendación E-Commerce (VERSIÓN MEJORADA)

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================
#  CONFIGURACIÓN INICIAL CRÍTICA
# =====================================================
try:
    # 🔥 MANTENER: Configuración de ProductReference
    from src.core.initialization.product_setup import setup_product_reference, check_product_reference_setup
    
    print("🔧 Configurando ProductReference...")
    if not setup_product_reference():
        logger.error("❌ No se pudo configurar ProductReference")
        print("⚠️  ProductReference no configurado - algunas funcionalidades pueden fallar")
    else:
        print("✅ ProductReference configurado correctamente")
        
except ImportError as e:
    logger.error(f"❌ Error importando configuración ProductReference: {e}")
    print("⚠️  Asegúrate de que src.core.initialization.product_setup.py existe")
except Exception as e:
    logger.error(f"❌ Error configurando ProductReference: {e}")

# 🔥 AHORA IMPORTAR CONFIGURACIÓN CENTRALIZADA
from src.core.config import settings

# =====================================================
#  BANNER ACTUALIZADO
# =====================================================
def show_banner():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     🎯 Sistema de Recomendación E-Commerce - ADVANCED RAG       ║")
    print("║     🤖 Con procesamiento ML 100% Local                          ║")
    print("║     🔥 Multi-categoría: Electrónicos, Ropa, Hogar...            ║")
    print("║     📦 ProductReference + WorkingAdvancedRAGAgent               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

def show_config():
    """Mostrar configuración actual del sistema."""
    from src.core.config import settings
    
    print("\n🔧 CONFIGURACIÓN ACTUAL:")
    print(f"   • Modo: {settings.CURRENT_MODE}")
    
    if settings.ML_ENABLED:
        print(f"   • ML: ✅ HABILITADO - Predicción de categorías, NLP, embeddings ML")
        print(f"   • Características: {', '.join(settings.ML_FEATURES)}")
    else:
        print(f"   • ML: ❌ DESHABILITADO - Solo búsqueda semántica básica")
    
    print(f"   • NLP: {'✅ HABILITADO' if settings.NLP_ENABLED else '❌ DESHABILITADO'}")
    print(f"   • LLM Local: {'✅ HABILITADO' if settings.LOCAL_LLM_ENABLED else '❌ DESHABILITADO'}")
    
    # 🔥 MANTENER: Estado de ProductReference
    try:
        from src.core.initialization.product_setup import check_product_reference_setup
        if check_product_reference_setup():
            print(f"   • ProductReference: ✅ CONFIGURADO")
        else:
            print(f"   • ProductReference: ⚠️  PARCIALMENTE CONFIGURADO")
    except:
        print(f"   • ProductReference: ❌ NO CONFIGURADO")
    
    print()

# =====================================================
#  PARSER DE ARGUMENTOS - MEJORADO
# =====================================================
def parse_arguments():
    """Parse arguments mejorado."""
    parser = argparse.ArgumentParser(
        description="Sistema de Recomendación E-Commerce - ML Local con RAG Avanzado",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s rag --mode enhanced        # ML completo con NLP y RLHF
  %(prog)s rag --mode basic           # Solo búsqueda básica
  %(prog)s rag --mode balanced        # ML básico sin NLP
  
  %(prog)s index                      # Construir índice
  %(prog)s ml                         # Ver estadísticas ML
  %(prog)s ml repair                  # Reparar embeddings ML
  %(prog)s test product-ref           # Test ProductReference
  %(prog)s test rag-agent             # Test WorkingAdvancedRAGAgent
  %(prog)s verify                     # Verificar sistema completo
        """
    )
    
    parser.add_argument(
        'command',
        choices=['rag', 'index', 'ml', 'train', 'test', 'verify', 'interactive'],
        help='Comando a ejecutar'
    )
    
    parser.add_argument('--mode', 
                       choices=['basic', 'enhanced', 'balanced', 'llm_enhanced'],
                       default='enhanced',
                       help='Modo de operación del sistema')
    
    parser.add_argument(
        'subcommand',
        nargs='?',
        default='',
        help='Subcomando (stats, repair, test, rlhf, collab)'
    )
    
    # Argumentos opcionales
    parser.add_argument('--data-dir', help='Directorio de datos')
    parser.add_argument('--verbose', '-v', action='store_true', help='Modo verbose')
    
    # 🔥 Opción ML explícita
    parser.add_argument('--ml', action='store_true', help='Habilitar ML')
    parser.add_argument('--no-ml', action='store_false', dest='ml', help='Deshabilitar ML')
    
    # 🔥 Opciones específicas para RAG Avanzado
    parser.add_argument('--max-results', type=int, default=5,
                       help='Número máximo de resultados a mostrar')
    parser.add_argument('--user-id', help='ID de usuario para personalización')
    parser.add_argument('--rag-debug', action='store_true', 
                       help='Modo debug para RAG avanzado')
    parser.add_argument('--no-collaborative', action='store_true',
                       help='Deshabilitar filtro colaborativo')
    parser.add_argument('--no-rlhf', action='store_true',
                       help='Deshabilitar RLHF')
    
    # 🔥 Opciones específicas para ProductReference
    parser.add_argument('--product-ref-debug', action='store_true', 
                       help='Modo debug para ProductReference')
    
    return parser.parse_args()

# =====================================================
#  FUNCIONES CRÍTICAS MANTENIDAS
# =====================================================
def run_index(data_dir: Optional[str] = None, verbose: bool = False):
    """Construir índice vectorial - Versión mejorada"""
    print("\n🔨 CONSTRUYENDO ÍNDICE VECTORIAL")
    print("="*50)
    
    try:
        from src.core.data.loader import DataLoader
        
        loader = DataLoader(
            raw_dir=Path(data_dir) if data_dir else settings.RAW_DIR,
            processed_dir=settings.PROC_DIR
        )
        
        products = loader.load_data()
        
        if not products:
            print("❌ No se pudieron cargar productos")
            return
        
        print(f"📦 Productos cargados: {len(products)}")
        
        # Estadísticas mejoradas
        if settings.ML_ENABLED:
            ml_count = sum(1 for p in products if getattr(p, 'ml_processed', False))
            embed_count = sum(1 for p in products if getattr(p, 'embedding', None))
            cat_count = sum(1 for p in products if getattr(p, 'predicted_category', None))
            
            print(f"📊 ESTADÍSTICAS ML:")
            print(f"   • Con ML procesado: {ml_count} ({ml_count/len(products)*100:.1f}%)")
            print(f"   • Con embeddings: {embed_count}")
            print(f"   • Con categorías predichas: {cat_count}")
        
        # 🔥 Construir índice con ChromaBuilder mejorado si está disponible
        try:
            from src.core.data.chroma_builder import OptimizedChromaBuilder
            
            print("🔧 Usando OptimizedChromaBuilder...")
            
            builder = OptimizedChromaBuilder(
                processed_json_path=settings.PROC_DIR / "products.json",
                chroma_db_path=Path(settings.CHROMA_DB_PATH),
                embedding_model=settings.ML_EMBEDDING_MODEL,
                device=settings.DEVICE,
                use_product_embeddings=settings.ML_ENABLED,
                ml_logging=verbose
            )
            
            index = builder.build_index(persist=True)
            
            # Estadísticas del índice
            stats = builder.get_index_stats()
            print(f"✅ Índice construido:")
            print(f"   • Documentos: {stats.get('document_count', 'N/A')}")
            print(f"   • ML habilitado: {stats.get('ml_enabled', 'N/A')}")
            
            # Información adicional si está disponible
            if 'ml_info' in stats:
                ml_info = stats['ml_info']
                print(f"   • Muestras con ML: {ml_info.get('samples_with_ml', 0)}/10")
                print(f"   • Muestras con embedding: {ml_info.get('samples_with_embedding', 0)}/10")
            
            builder.cleanup()
            
        except ImportError:
            # Fallback a retriever original
            from src.core.rag.basic.retriever import Retriever
            print("⚠️  Usando Retriever original...")
            
            retriever = Retriever(
                index_path=settings.VECTOR_INDEX_PATH,
                embedding_model=settings.EMBEDDING_MODEL,
                device=settings.DEVICE
            )
            
            retriever.build_index(products)
            print(f"✅ Índice construido con {len(products)} productos")
        
    except Exception as e:
        print(f"❌ Error construyendo índice: {e}")
        if verbose:
            import traceback
            traceback.print_exc()

def run_ml_stats():
    """Estadísticas ML mejoradas."""
    print("\n🤖 ESTADÍSTICAS ML")
    print("="*50)
    
    print(f"📊 CONFIGURACIÓN ML:")
    print(f"   • Estado: {'✅ HABILITADO' if settings.ML_ENABLED else '❌ DESHABILITADO'}")
    
    if settings.ML_ENABLED:
        print(f"   • Características: {', '.join(settings.ML_FEATURES)}")
        print(f"   • Modelo embeddings: {settings.ML_EMBEDDING_MODEL}")
        print(f"   • Peso ML: {settings.ML_WEIGHT}")
        print(f"   • Categorías ML: {', '.join(settings.ML_CATEGORIES[:5])}...")
    
    # 🔥 Verificar ProductReference
    try:
        from src.core.data.product_reference import ProductClassHolder
        if ProductClassHolder.is_available():
            print(f"   • ProductReference: ✅ CONFIGURADO")
        else:
            print(f"   • ProductReference: ⚠️  NO CONFIGURADO")
    except Exception as e:
        print(f"   • ProductReference: ❌ ERROR: {e}")
    
    # Cargar productos para estadísticas detalladas
    try:
        from src.core.data.loader import DataLoader
        
        loader = DataLoader(
            raw_dir=settings.RAW_DIR,
            processed_dir=settings.PROC_DIR
        )
        
        products = loader.load_data()[:100]  # Primeros 100 para estadísticas
        
        if products:
            # Contar productos con ML
            ml_count = sum(1 for p in products if getattr(p, 'ml_processed', False))
            embed_count = sum(1 for p in products if getattr(p, 'embedding', None))
            cat_count = sum(1 for p in products if getattr(p, 'predicted_category', None))
            
            print(f"\n📈 ESTADÍSTICAS PRODUCTOS (muestra de {len(products)}):")
            print(f"   • Procesados con ML: {ml_count} ({ml_count/len(products)*100:.1f}%)")
            print(f"   • Con embeddings: {embed_count}")
            print(f"   • Con categorías predichas: {cat_count}")
            
            # Distribución de categorías
            print(f"\n🏷️ DISTRIBUCIÓN DE CATEGORÍAS:")
            categories = {}
            for p in products:
                cat = getattr(p, 'main_category', 'Unknown') or 'Unknown'
                cat = getattr(p, 'predicted_category', cat) or cat
                categories[cat] = categories.get(cat, 0) + 1
            
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   • {cat}: {count} productos")
        
    except Exception as e:
        print(f"⚠️ Error cargando productos: {e}")

def run_train(args):
    """Comando para entrenar modelos ML - versión mejorada"""
    print("\n🤖 ENTRENAMIENTO DE MODELOS ML")
    print("="*50)
    
    if args.subcommand == "rlhf":
        try:
            from src.core.rag.advanced.train_pipeline import RLHFTrainingPipeline
            
            pipeline = RLHFTrainingPipeline()
            result = pipeline.train_from_feedback(min_samples=10)
            
            if result:
                print(f"✅ RLHF entrenado exitosamente")
                print(f"   • Muestras: {result.get('samples', 0)}")
                print(f"   • Pérdida: {result.get('train_loss', 0):.4f}")
                print(f"   • Tiempo: {result.get('training_time', 0):.2f}s")
                print(f"   • Guardado en: data/models/rlhf_model/")
            else:
                print("⚠️ No se pudo entrenar RLHF (datos insuficientes)")
        except Exception as e:
            print(f"❌ Error entrenando RLHF: {e}")
            import traceback
            traceback.print_exc()
        
    elif args.subcommand == "collab":
        try:
            from scripts.maintenance import update_collaborative_embeddings
            update_collaborative_embeddings()
            print("✅ Embeddings colaborativos actualizados")
        except Exception as e:
            print(f"❌ Error actualizando embeddings: {e}")
    
    else:
        print("ℹ️ Subcomandos disponibles:")
        print("   • train rlhf     - Entrenar modelo RLHF desde feedback")
        print("   • train collab   - Actualizar embeddings colaborativos")

def run_test_command(args):
    """Comandos de testing mejorados."""
    print("\n🧪 COMANDOS DE TEST")
    print("="*50)
    
    if args.subcommand == "product-ref":
        print("\n🔍 TEST DE ProductReference")
        print("-"*30)
        
        try:
            from src.core.data.product_reference import ProductReference
            
            # Crear un producto de prueba simple
            class MockProduct:
                def __init__(self):
                    self.id = "test_123"
                    self.title = "Nintendo Switch OLED - Consola de Videojuegos"
                    self.price = 349.99
                    self.description = "Consola Nintendo Switch con pantalla OLED de 7 pulgadas"
                    self.main_category = "Electronics"
                    self.ml_processed = True
                    self.embedding = [0.1] * 384
                    self.predicted_category = "Video Games"
                
                def to_metadata(self):
                    return {
                        "title": self.title,
                        "price": self.price,
                        "main_category": self.main_category,
                        "ml_processed": self.ml_processed,
                        "description": self.description
                    }
            
            test_product = MockProduct()
            ref = ProductReference.from_product(test_product, source="test")
            
            print(f"✅ ProductReference creado: {ref}")
            print(f"   • ID: {ref.id}")
            print(f"   • Title: {ref.title}")
            print(f"   • Source: {ref.source}")
            print(f"   • ML procesado: {ref.is_ml_processed}")
            print(f"   • Categoría: {ref.metadata.get('main_category', 'N/A')}")
            
            # Test de serialización
            ref_dict = ref.to_dict()
            print(f"✅ Convertido a dict: {len(ref_dict)} campos")
            
            # Test de reconstrucción
            ref2 = ProductReference.from_dict(ref_dict)
            print(f"✅ Reconstruido desde dict: {ref2.id}")
            
        except Exception as e:
            print(f"❌ Error en test ProductReference: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.subcommand == "rag-agent":
        print("\n🔍 TEST DE WorkingAdvancedRAGAgent")
        print("-"*30)
        
        try:
            from src.core.rag.advanced.WorkingRAGAgent import (
                create_rag_agent,
                test_rag_pipeline
            )
            
            # Test básico del pipeline
            print("🧪 Probando pipeline RAG...")
            test_result = test_rag_pipeline(query="smartphone barato")
            
            print(f"✅ Test completado:")
            print(f"   • Éxito: {test_result.get('success', False)}")
            print(f"   • Productos encontrados: {test_result.get('products_found', 0)}")
            print(f"   • Tiempo: {test_result.get('processing_time', 0):.2f}s")
            
            if test_result.get('success'):
                print("\n📋 Configuración del agente:")
                config_summary = test_result.get('config_summary', {})
                rag_config = config_summary.get('rag_config', {})
                print(f"   • Modo: {rag_config.get('mode', 'N/A')}")
                print(f"   • ML: {'✅' if rag_config.get('ml_enabled') else '❌'}")
                print(f"   • LLM: {'✅' if rag_config.get('local_llm_enabled') else '❌'}")
                
                components = config_summary.get('components', {})
                print(f"   • RLHF: {'✅' if components.get('rlhf_pipeline') else '❌'}")
                print(f"   • Collaborative Filter: {'✅' if components.get('collaborative_filter') else '❌'}")
            
        except Exception as e:
            print(f"❌ Error en test RAG Agent: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("ℹ️ Subcomandos de test disponibles:")
        print("   • test product-ref     - Test de ProductReference")
        print("   • test rag-agent       - Test de WorkingAdvancedRAGAgent")

# =====================================================
#  RUN_RAG - VERSIÓN MEJORADA CON WORKINGADVANCEDRAGAGENT
# =====================================================
def run_rag(data_dir: Optional[str] = None, 
           mode: str = "enhanced",
           verbose: bool = False,
           ml_enabled: Optional[bool] = None,
           max_results: int = 5,
           user_id: Optional[str] = None,
           rag_debug: bool = False,
           no_collaborative: bool = False,
           no_rlhf: bool = False,
           product_ref_debug: bool = False):
    
    print(f"\n🧠 MODO RAG: {mode.upper()}")
    print("="*50)
    
    # 🔥 MEJORA: Verificar y forzar configuración ML si se especifica
    if ml_enabled is not None:
        settings.ML_ENABLED = ml_enabled
        if ml_enabled:
            # Forzar características ML básicas
            if not settings.ML_FEATURES:
                settings.ML_FEATURES = {'category', 'embedding', 'similarity'}
            logger.info("🔥 ML forzado manualmente: ✅ HABILITADO")
    
    # 🔥 MEJORA: Mostrar configuración real
    print(f"\n📋 CONFIGURACIÓN REAL:")
    print(f"   • Modo: {settings.CURRENT_MODE}")
    print(f"   • ML: {'✅ HABILITADO' if settings.ML_ENABLED else '❌ DESHABILITADO'}")
    print(f"   • Características ML: {list(settings.ML_FEATURES)}")
    print(f"   • NLP: {'✅ HABILITADO' if settings.NLP_ENABLED else '❌ DESHABILITADO'}")
    print(f"   • LLM: {'🧠 ON' if settings.LOCAL_LLM_ENABLED else 'OFF'}")
    
    # 🔥 Configurar debug si se solicita
    if rag_debug:
        print("🔍 Modo debug de RAG activado")
        logging.getLogger('src.core.rag.advanced').setLevel(logging.DEBUG)
    
    if product_ref_debug:
        print("🔍 Modo debug de ProductReference activado")
        logging.getLogger('src.core.data.product_reference').setLevel(logging.DEBUG)
    
    # 🔥 Manejo del argumento ml_enabled
    if ml_enabled is not None:
        print(f"🔥 ML especificado explícitamente: {'✅ HABILITADO' if ml_enabled else '❌ DESHABILITADO'}")
        settings.ML_ENABLED = ml_enabled
        if not ml_enabled:
            settings.NLP_ENABLED = False
    
    try:
        # Cargar productos
        from src.core.data.loader import DataLoader
        from src.core.data.user_manager import UserManager
        
        # Definir directorio de datos
        if data_dir:
            data_path = Path(data_dir)
        else:
            data_path = settings.RAW_DIR
        
        print(f"\n📂 Cargando datos desde: {data_path}")
        
        loader = DataLoader(
            raw_dir=data_path,
            processed_dir=settings.PROC_DIR
        )
        
        products = loader.load_data()
        
        if not products:
            print("❌ No se pudieron cargar productos")
            return
        
        print(f"📦 Productos cargados: {len(products)}")
        
        # Gestor de usuarios
        user_manager = UserManager()
        
        # Crear o usar usuario especificado
        if not user_id:
            user_profile = user_manager.create_user_profile(
                age=25,
                gender="male",
                country="Spain",
                language="es"
            )
            user_id = user_profile.user_id
            print(f"👤 Usuario creado: {user_id}")
        else:
            user_profile = user_manager.get_user(user_id)
            if user_profile:
                print(f"👤 Usuario existente: {user_id}")
            else:
                user_profile = user_manager.create_user_profile(
                    user_id=user_id,
                    age=30,
                    gender="female",
                    country="Spain",
                    language="es"
                )
                print(f"👤 Usuario registrado: {user_id}")
        
        # 🔥 INICIALIZAR WORKINGADVANCEDRAGAGENT
        print("\n🚀 Inicializando WorkingAdvancedRAGAgent...")
        
        try:
            from src.core.rag.advanced.WorkingRAGAgent import (
                WorkingAdvancedRAGAgent,
                RAGConfig,
                RAGMode
            )
            
            # Configurar modo RAG basado en el modo del sistema
            rag_mode_map = {
                'basic': RAGMode.BASIC,
                'balanced': RAGMode.HYBRID,
                'enhanced': RAGMode.ML_ENHANCED,
                'llm_enhanced': RAGMode.LLM_ENHANCED
            }
            
            rag_mode = rag_mode_map.get(mode, RAGMode.HYBRID)
            
            # Crear configuración RAG
            rag_config = RAGConfig(
                mode=rag_mode,
                ml_enabled=settings.ML_ENABLED,
                local_llm_enabled=settings.LOCAL_LLM_ENABLED,
                max_final=max_results,
                enable_reranking=(not no_rlhf),  # Deshabilitar RLHF si se solicita
                ml_features=list(settings.ML_FEATURES),
                use_ml_embeddings=settings.ML_ENABLED and 'embedding' in settings.ML_FEATURES,
                ml_embedding_weight=settings.ML_WEIGHT
            )
            
            # Crear agente
            rag_agent = WorkingAdvancedRAGAgent(config=rag_config)
            
            # Deshabilitar componentes si se solicita
            if no_collaborative:
                rag_agent._collaborative_filter = None
                print("🤝 Collaborative Filter: ❌ DESHABILITADO")
            
            if no_rlhf:
                rag_agent.rlhf_model = None
                rag_agent.rlhf_pipeline = None
                print("🧠 RLHF: ❌ DESHABILITADO")
            
            # Mostrar configuración del agente
            config_summary = rag_agent.get_config_summary()
            print(f"\n📡 CONFIGURACIÓN RAG AGENT:")
            print(f"   • Modo: {config_summary['rag_config']['mode']}")
            print(f"   • ML: {'✅' if config_summary['rag_config']['ml_enabled'] else '❌'}")
            print(f"   • LLM: {'✅' if config_summary['rag_config']['local_llm_enabled'] else '❌'}")
            print(f"   • RLHF: {'✅' if config_summary['components']['rlhf_pipeline'] else '❌'}")
            print(f"   • Collaborative Filter: {'✅' if config_summary['components']['collaborative_filter'] else '❌'}")
            
            # Verificar que el índice existe
            if not rag_agent.retriever.index_exists():
                print("\n🔧 Índice no encontrado, construyendo...")
                rag_agent.retriever.build_index(products)
                print(f"✅ Índice construido con {len(products)} productos")
            
        except ImportError as e:
            print(f"❌ No se pudo importar WorkingAdvancedRAGAgent: {e}")
            print("⚠️  Fallback a RAG simple...")
            
            # Fallback a RAG simple
            from src.core.rag.basic.retriever import Retriever
            from src.core.rag.basic.RAG import SimpleRAG
            
            retriever = Retriever(
                index_path=settings.VECTOR_INDEX_PATH,
                embedding_model=settings.EMBEDDING_MODEL,
                device=settings.DEVICE
            )
            
            if not retriever.index_exists():
                print("🔧 Construyendo índice...")
                retriever.build_index(products)
            
            rag_agent = SimpleRAG(retriever=retriever)
            print("🧠 Agente RAG simple inicializado")
        
        # Loop interactivo mejorado
        print(f"\n💡 Escribe 'exit' para salir, 'help' para comandos")
        print("="*50)
        
        while True:
            try:
                query = input("\n🔍 Tu consulta: ").strip()
                
                if query.lower() == 'exit':
                    print("👋 ¡Hasta luego!")
                    break
                
                if query.lower() == 'help':
                    print("\n📋 COMANDOS DISPONIBLES:")
                    print("   • exit - Salir del programa")
                    print("   • stats - Mostrar estadísticas")
                    print("   • config - Mostrar configuración")
                    print("   • user - Mostrar información del usuario")
                    print("   • clear - Limpiar caché")
                    continue
                
                if query.lower() == 'stats':
                    print(f"\n📊 ESTADÍSTICAS:")
                    print(f"   • Productos totales: {len(products)}")
                    print(f"   • ML habilitado: {settings.ML_ENABLED}")
                    print(f"   • LLM habilitado: {settings.LOCAL_LLM_ENABLED}")
                    
                    # Estadísticas del agente RAG si está disponible
                    if hasattr(rag_agent, 'get_config_summary'):
                        config = rag_agent.get_config_summary()
                        print(f"   • Modo RAG: {config['rag_config']['mode']}")
                        print(f"   • RLHF: {'✅' if config['components']['rlhf_pipeline'] else '❌'}")
                        print(f"   • Collaborative Filter: {'✅' if config['components']['collaborative_filter'] else '❌'}")
                    
                    continue
                
                if query.lower() == 'config':
                    show_config()
                    continue
                
                if query.lower() == 'user':
                    print(f"\n👤 INFORMACIÓN DEL USUARIO:")
                    print(f"   • ID: {user_id}")
                    if hasattr(user_profile, 'age'):
                        print(f"   • Edad: {user_profile.age}")
                    if hasattr(user_profile, 'gender'):
                        print(f"   • Género: {user_profile.gender}")
                    if hasattr(user_profile, 'country'):
                        print(f"   • País: {user_profile.country}")
                    continue
                
                if query.lower() == 'clear':
                    if hasattr(rag_agent, 'clear_cache'):
                        rag_agent.clear_cache()
                        print("🗑️  Cache limpiado")
                    else:
                        print("⚠️  El agente no tiene función clear_cache")
                    continue
                
                if not query:
                    continue
                
                print(f"\n🔍 Buscando: '{query}'...")
                
                # 🔥 USAR WORKINGADVANCEDRAGAGENT
                if hasattr(rag_agent, 'process_query'):
                    # Procesar con RAG avanzado
                    response = rag_agent.process_query(query, user_id)
                    
                    if isinstance(response, dict):
                        answer = response.get('answer', 'Sin respuesta')
                        products_result = response.get('products', [])
                        stats = response.get('stats', {})
                        
                        # Mostrar estadísticas si está en modo verbose
                        if verbose:
                            print(f"\n📊 ESTADÍSTICAS PROCESAMIENTO:")
                            print(f"   • Tiempo: {stats.get('processing_time', 0):.2f}s")
                            print(f"   • Resultados iniciales: {stats.get('initial_results', 0)}")
                            print(f"   • Resultados finales: {stats.get('final_results', 0)}")
                            print(f"   • ML mejorado: {stats.get('ml_enhanced', False)}")
                            print(f"   • Re-ranking: {stats.get('reranking_enabled', False)}")
                    else:
                        answer = str(response)
                        products_result = []
                else:
                    # Fallback a RAG simple
                    products_result = rag_agent.search(query, top_k=max_results)
                    answer = f"Encontré {len(products_result)} productos"
                
                # Mostrar respuesta
                print(f"\n🤖 {answer}")
                
                # Mostrar resultados
                if products_result:
                    print(f"\n📦 RESULTADOS ({len(products_result)} encontrados):")
                    
                    for i, product in enumerate(products_result[:max_results], 1):
                        # 🔥 Manejar tanto ProductReference como productos normales
                        try:
                            from src.core.data.product_reference import ProductReference
                            
                            if isinstance(product, ProductReference):
                                # Es un ProductReference
                                title = product.title[:80]
                                price = product.price
                                category = product.metadata.get('main_category', 'General')
                                category = product.ml_features.get('predicted_category', category)
                                score = product.score
                                source = product.source
                                
                                # Emoji basado en categoría
                                emoji = "📱" if "phone" in title.lower() or "smartphone" in title.lower() else \
                                        "🎮" if "nintendo" in title.lower() or "game" in title.lower() else \
                                        "💻" if "laptop" in title.lower() or "computer" in title.lower() else \
                                        "📦"
                            else:
                                # Producto normal
                                title = getattr(product, 'title', str(product))[:80]
                                price = getattr(product, 'price', 0.0)
                                category = getattr(product, 'main_category', 'General')
                                category = getattr(product, 'predicted_category', category)
                                score = getattr(product, 'score', 0.0)
                                source = "simple_rag"
                                
                                emoji = "📦"
                            
                            # Mostrar producto
                            print(f"  {emoji} {i}. {title}")
                            print(f"     💰 ${price:.2f} | 🏷️ {category}")
                            
                            if verbose:
                                print(f"     ⭐ Score: {score:.3f} | 📍 Source: {source}")
                            
                            # Línea separadora
                            if i < min(len(products_result), max_results):
                                print("     " + "-" * 40)
                                
                        except Exception as e:
                            print(f"  {i}. Error mostrando producto: {e}")
                
                # Feedback mejorado
                try:
                    print(f"\n💬 ¿Fue útil esta respuesta?")
                    feedback = input("   (s) Sí | (n) No | (skip) Saltar: ").strip().lower()
                    
                    if feedback == 's':
                        print("     ✅ ¡Gracias por tu feedback positivo!")
                        # Guardar feedback positivo
                        try:
                            user_manager.add_feedback(user_id, query, "positive")
                        except:
                            pass
                    elif feedback == 'n':
                        print("     ⚠️  Lo sentimos, mejoraremos")
                        # Guardar feedback negativo
                        try:
                            user_manager.add_feedback(user_id, query, "negative")
                        except:
                            pass
                    else:
                        print("     ℹ️  Feedback omitido")
                        
                except (KeyboardInterrupt, EOFError):
                    print("\n⚠️  Feedback interrumpido")
                    pass
                
            except KeyboardInterrupt:
                print("\n\n🛑 Sesión interrumpida")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                
    except Exception as e:
        print(f"❌ Error inicializando RAG: {e}")
        import traceback
        traceback.print_exc()

def run_interactive_mode():
    """Modo interactivo para explorar el sistema."""
    print("\n🎮 MODO INTERACTIVO")
    print("="*50)
    
    print("\n📋 COMANDOS DISPONIBLES:")
    print("   1. test-rag - Probar WorkingAdvancedRAGAgent")
    print("   2. test-product-ref - Probar ProductReference")
    print("   3. test-ml - Probar procesamiento ML")
    print("   4. verify - Verificar sistema completo")
    print("   5. exit - Salir")
    
    while True:
        try:
            choice = input("\n🔍 Elige una opción (1-5): ").strip()
            
            if choice == '1':
                from src.core.rag.advanced.WorkingRAGAgent import test_rag_pipeline
                result = test_rag_pipeline("smartphone barato")
                print(f"✅ Test RAG completado: {result.get('products_found', 0)} productos")
                
            elif choice == '2':
                # Test ProductReference
                try:
                    from src.core.data.product_reference import ProductReference
                    
                    class TestProduct:
                        def __init__(self):
                            self.id = "test_interactive"
                            self.title = "Producto de prueba interactivo"
                            self.price = 49.99
                            self.main_category = "Electronics"
                        
                        def to_metadata(self):
                            return {"title": self.title, "price": self.price}
                    
                    product = TestProduct()
                    ref = ProductReference.from_product(product)
                    print(f"✅ ProductReference creado: {ref.title}")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
            elif choice == '3':
                print("🧪 Procesamiento ML - En desarrollo...")
                run_ml_stats()
                
            elif choice == '4':
                try:
                    from scripts.verify_system import main as verify_main
                    verify_main()
                except ImportError:
                    print("❌ Script verify_system.py no encontrado")
                    
            elif choice == '5' or choice.lower() == 'exit':
                print("👋 ¡Hasta luego!")
                break
                
            else:
                print("❌ Opción no válida. Intenta de nuevo.")
                
        except KeyboardInterrupt:
            print("\n🛑 Modo interactivo interrumpido")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# =====================================================
#  MAIN COMPLETO
# =====================================================
if __name__ == "__main__":
    # Mostrar banner
    show_banner()
    
    # Parsear argumentos
    args = parse_arguments()
    
    # Configurar logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Mostrar configuración
    show_config()
    
    # Ejecutar comando
    try:
        if args.command == "index":
            run_index(data_dir=args.data_dir, verbose=args.verbose)
            
        elif args.command == "rag":
            # Manejar argumento --ml/--no-ml
            ml_enabled = None
            if hasattr(args, 'ml') and args.ml is not None:
                ml_enabled = args.ml
            
            run_rag(
                data_dir=args.data_dir,
                mode=args.mode,
                ml_enabled=ml_enabled,
                verbose=args.verbose,
                max_results=args.max_results,
                user_id=args.user_id,
                rag_debug=args.rag_debug,
                no_collaborative=args.no_collaborative,
                no_rlhf=args.no_rlhf,
                product_ref_debug=args.product_ref_debug
            )
            
        elif args.command == "ml":
            if args.subcommand == "repair":
                print("🔧 Ejecutando reparación de embeddings ML...")
                try:
                    from scripts.repair_ml_embeddings import repair_ml_embeddings
                    repair_ml_embeddings()
                except ImportError:
                    print("⚠️  Script repair_ml_embeddings.py no encontrado")
                    print("💡 Use la versión completa para esta funcionalidad")
            else:
                run_ml_stats()
            
        elif args.command == "train":
            run_train(args)
            
        elif args.command == "test":
            run_test_command(args)
            
        elif args.command == "verify":
            print("🔍 Verificando sistema completo...")
            try:
                from scripts.verify_system import main as verify_main
                verify_main()
            except ImportError:
                print("⚠️  Script verify_system.py no encontrado")
                print("💡 Use la versión completa para esta funcionalidad")
                
        elif args.command == "interactive":
            run_interactive_mode()
            
        else:
            print(f"❌ Comando no reconocido: {args.command}")
            sys.exit(1)
        
        print("\n✅ Ejecución completada")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Ejecución interrumpida")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)