#!/usr/bin/env python3
# main.py - Amazon Recommendation System - VERSIÓN COMPLETA CON ProductReference

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# =====================================================
#  🔥 CRÍTICO: CONFIGURAR ProductReference AL INICIO
# =====================================================

# Configurar logging PRIMERO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 🔥 CONFIGURACIÓN ProductReference ANTES de cualquier import
try:
    from src.core.initialization.product_setup import setup_product_reference, check_product_reference_setup
    
    print("🔧 Configurando ProductReference...")
    if not setup_product_reference():
        logger.error("❌ No se pudo configurar ProductReference")
        # Podrías decidir si continuar o salir
        print("⚠️  ProductReference no configurado - algunas funcionalidades pueden fallar")
    else:
        print("✅ ProductReference configurado correctamente")
        
    # Verificar configuración
    if not check_product_reference_setup():
        logger.warning("⚠️  ProductReference no está completamente configurado")
    
except ImportError as e:
    logger.error(f"❌ Error importando configuración ProductReference: {e}")
    print("⚠️  Asegúrate de que src.core.initialization.product_setup.py existe")
except Exception as e:
    logger.error(f"❌ Error configurando ProductReference: {e}")

# 🔥 AHORA IMPORTAR CONFIGURACIÓN CENTRALIZADA
from src.core.config import settings

# =====================================================
#  BANNER Y CONFIGURACIÓN
# =====================================================
def show_banner():
    print("╔══════════════════════════════════════════════════╗")
    print("║     🎯 Sistema de Recomendación Amazon           ║")
    print("║     🤖 Con procesamiento ML 100% Local           ║")
    print("║     🔥 ProductReference Configurado              ║")
    print("╚══════════════════════════════════════════════════╝")

def show_config():
    """Mostrar configuración actual del sistema."""
    print("\n🔧 CONFIGURACIÓN ACTUAL:")
    print(f"   • ML: {'✅ HABILITADO' if settings.ML_ENABLED else '❌ DESHABILITADO'}")
    if settings.ML_ENABLED:
        print(f"   • Características: {', '.join(settings.ML_FEATURES)}")
    print(f"   • LLM Local: {'✅ HABILITADO' if settings.LOCAL_LLM_ENABLED else '❌ DESHABILITADO'}")
    if settings.LOCAL_LLM_ENABLED:
        print(f"   • Modelo: {settings.LOCAL_LLM_MODEL}")
    
    # 🔥 Mostrar estado de ProductReference
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
#  PARSER DE ARGUMENTOS
# =====================================================
def parse_arguments():
    """Parse arguments super simple."""
    parser = argparse.ArgumentParser(
        description="Sistema de Recomendación Amazon - ML Local",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos con modos:
  %(prog)s rag --mode basic          # Modo básico sin ML
  %(prog)s rag --mode enhanced       # Modo completo con NER y Zero-Shot
  %(prog)s rag --mode balanced       # ML básico sin NLP
  
  %(prog)s verify                    # Verificar sistema completo
  %(prog)s test nlp                  # Probar componentes NLP
        """
    )
    # Solo un argumento de comando
    parser.add_argument(
        'command',
        choices=['rag', 'index', 'ml', 'train', 'test'],
        help='Comando a ejecutar (rag, index, ml, train, test)'
    )
    parser.add_argument('--mode', 
                       choices=['basic', 'enhanced', 'balanced'],
                       default='enhanced',
                       help='Modo de operación del sistema')
    # Subcomando para ml y train
    parser.add_argument(
        'subcommand',
        nargs='?',
        default='',
        help='Subcomando (stats, repair, test, rlhf, collab)'
    )
    
    # Argumentos opcionales simples
    parser.add_argument('--data-dir', help='Directorio de datos')
    parser.add_argument('--verbose', '-v', action='store_true', help='Modo verbose')
    
    # 🔥 Opciones ML
    parser.add_argument('--ml', action='store_true', help='Habilitar ML')
    parser.add_argument('--no-ml', action='store_false', dest='ml', help='Deshabilitar ML')
    
    # 🔥 Opciones específicas para ProductReference
    parser.add_argument('--product-ref-debug', action='store_true', 
                       help='Modo debug para ProductReference')
    
    return parser.parse_args()

# =====================================================
#  COMANDO INDEX
# =====================================================
# COMANDO INDEX - VERSIÓN CORREGIDA
def run_index(data_dir: Optional[str] = None, verbose: bool = False):
    """Construir índice vectorial."""
    print("\n🔨 CONSTRUYENDO ÍNDICE VECTORIAL")
    print("="*50)
    
    try:
        # 🔥 CORRECCIÓN: FastDataLoader simplificado no acepta parámetros ML
        try:
            from src.core.data.loader import FastDataLoader
            print("🚀 Usando FastDataLoader optimizado...")
            
            loader = FastDataLoader(
                use_progress_bar=True,
                # 🔥 ELIMINAR estos parámetros que ya no existen:
                # ml_enabled=settings.ML_ENABLED,
                # ml_features=list(settings.ML_FEATURES)
            )
            
            # Ruta para JSON procesado
            processed_json = settings.PROC_DIR / "products.json"
            products = loader.load_data(processed_json)
            
        except ImportError as e:
            print(f"⚠️  Error importando FastDataLoader: {e}")
            # Fallback a DataLoader original
            from src.core.data.loader import DataLoader
            print("⚠️  Usando DataLoader original...")
            
            loader = DataLoader(
                raw_dir=Path(data_dir) if data_dir else settings.RAW_DIR,
                processed_dir=settings.PROC_DIR
            )
            products = loader.load_data()
        
        if not products:
            print("❌ No se pudieron cargar productos")
            return
        
        print(f"📦 Productos cargados: {len(products)}")
        
        # 🔥 Estadísticas ML si está habilitado
        if settings.ML_ENABLED:
            ml_count = sum(1 for p in products if getattr(p, 'ml_processed', False))
            embed_count = sum(1 for p in products if getattr(p, 'embedding', None))
            print(f"   • Con ML procesado: {ml_count}")
            print(f"   • Con embeddings: {embed_count}")
        
        # Construir índice con ChromaBuilder mejorado
        try:
            from src.core.data.chroma_builder import OptimizedChromaBuilder
            
            builder = OptimizedChromaBuilder(
                processed_json_path=settings.PROC_DIR / "products.json",
                chroma_db_path=Path(settings.CHROMA_DB_PATH),
                embedding_model=settings.ML_EMBEDDING_MODEL,
                device=settings.DEVICE,
                use_product_embeddings=settings.ML_ENABLED,  # 🔥 Esto usa ML_ENABLED correctamente
                ml_logging=verbose
            )
            
            print("🔧 Construyendo índice Chroma...")
            index = builder.build_index(persist=True)
            
            # Estadísticas del índice
            stats = builder.get_index_stats()
            print(f"✅ Índice construido:")
            print(f"   • Documentos: {stats.get('document_count', 'N/A')}")
            print(f"   • ML habilitado: {stats.get('ml_enabled', 'N/A')}")
            
            if verbose and 'ml_info' in stats:
                ml_info = stats['ml_info']
                print(f"   • Muestras con ML: {ml_info.get('samples_with_ml', 0)}/10")
                print(f"   • Muestras con embedding: {ml_info.get('samples_with_embedding', 0)}/10")
            
            # Limpiar memoria
            builder.cleanup()
            
        except ImportError as e:
            print(f"⚠️  OptimizedChromaBuilder no disponible: {e}")
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

# =====================================================
#  COMANDO RAG
# =====================================================
def run_rag(data_dir: Optional[str] = None, 
           mode: str = "enhanced",   # 🔥 Ahora el modo controla todo
           verbose: bool = False,
           ml_enabled: Optional[bool] = None,
           product_ref_debug: bool = False):
    
    print(f"\n🧠 MODO RAG: {mode.upper()}")
    print("="*50)
    
    # 🔥 NUEVO — Configuración automática del sistema
    from src.core.config import apply_system_mode
    apply_system_mode(mode)

    # Cargar settings después de aplicar modo
    from src.core.config import settings
    
    print(f"\n📋 CONFIGURACIÓN APLICADA:")
    print(f"   • Modo: {settings.CURRENT_MODE}")
    print(f"   • ML: {'✅ HABILITADO' if settings.ML_ENABLED else '❌ DESHABILITADO'}")
    print(f"   • NLP: {'✅ HABILITADO' if settings.NLP_ENABLED else '❌ DESHABILITADO'}")
    print(f"   • LLM: {'🧠 ON' if settings.LOCAL_LLM_ENABLED else 'OFF'}")
    print(f"   • Ref. Productos: {'📦 ON' if settings.PRODUCT_REF_ENABLED else 'OFF'}")
    
    
    if mode == "basic":
        # Deshabilitar todo ML/NLP
        settings.update_ml_settings(ml_enabled=False)
        settings.CURRENT_MODE = "basic"
        if hasattr(settings, 'NLP_ENABLED'):
            settings.NLP_ENABLED = False
        print("🔧 Modo BÁSICO activado: Solo búsqueda semántica")
        
    elif mode == "enhanced":
        # Habilitar todo
        settings.update_ml_settings(ml_enabled=True)
        settings.CURRENT_MODE = "enhanced"
        
        # Asegurar que NLP esté habilitado
        if hasattr(settings, 'NLP_ENABLED'):
            settings.NLP_ENABLED = True
            
        print("🔧 Modo ENHANCED activado: NER + Zero-Shot + ML completo")
        
    elif mode == "balanced":
        # ML básico sin NLP
        settings.update_ml_settings(ml_enabled=True)
        settings.CURRENT_MODE = "balanced"
        
        if hasattr(settings, 'NLP_ENABLED'):
            settings.NLP_ENABLED = False
            
        print("🔧 Modo BALANCED activado: ML básico sin NLP")
    
    # Si ml_enabled no se especifica, usar configuración global
    if ml_enabled is None:
        ml_enabled = settings.ML_ENABLED
    
    print(f"🤖 ML habilitado para esta sesión: {'✅' if ml_enabled else '❌'}")
    
    # 🔥 ACTUALIZAR CONFIGURACIÓN GLOBAL
    if ml_enabled != settings.ML_ENABLED:
        settings.update_ml_settings(ml_enabled=ml_enabled)
        print(f"📡 Configuración ML actualizada globalmente: {ml_enabled}")
    
    # 🔥 Configurar debug de ProductReference si se solicita
    if product_ref_debug:
        print("🔍 Modo debug de ProductReference activado")
        logging.getLogger('src.core.data.product_reference').setLevel(logging.DEBUG)
    
    try:
        # Cargar productos - CORRECCIÓN APLICADA
        from src.core.data.loader import DataLoader
        from src.core.data.user_manager import UserManager
        
        # Definir directorio de datos - usar el parámetro data_dir si se proporciona,
        # de lo contrario usar RAW_DIR de settings
        if data_dir:
            data_path = Path(data_dir)
        else:
            data_path = settings.RAW_DIR
        
        print(f"📂 Cargando datos desde: {data_path}")
        
        loader = DataLoader(
            raw_dir=data_path,
            processed_dir=settings.PROC_DIR
        )
        
        products = loader.load_data()
        
        if not products:
            print("❌ No se pudieron cargar productos")
            print("   Asegúrate de que el directorio contiene archivos JSON de productos")
            print(f"   Directorio verificado: {data_path}")
            return
        
        print(f"📦 Productos cargados: {len(products)}")
        
        # 🔥 Test de ProductReference si está en modo debug
        if product_ref_debug:
            print("\n🧪 TEST DE ProductReference:")
            try:
                from src.core.data.product_reference import ProductReference, create_ml_enhanced_reference
                
                # Probar con un producto
                test_product = products[0] if products else None
                if test_product:
                    ref = ProductReference.from_product(test_product, source="ml_enhanced")
                    print(f"   • ProductReference creado: {ref}")
                    print(f"   • ID: {ref.id}")
                    print(f"   • Título: {ref.title}")
                    print(f"   • Source: {ref.source}")
                    print(f"   • ML procesado: {ref.is_ml_processed}")
                    print(f"   • Tiene embedding: {ref.has_embedding}")
                    
                    # Test de conversión a diccionario
                    ref_dict = ref.to_dict()
                    print(f"   • Convertido a dict: {len(ref_dict)} campos")
                    
            except Exception as e:
                print(f"   ⚠️ Error en test ProductReference: {e}")
        
        # Inicializar RAG (intentar avanzado, luego simple)
        rag_agent = None
        
        # Intentar RAG avanzado
        try:
            from src.core.rag.advanced.WorkingRAGAgent import WorkingAdvancedRAGAgent, RAGConfig
            
            rag_config = RAGConfig(
                ml_enabled=settings.ML_ENABLED,
                local_llm_enabled=settings.LOCAL_LLM_ENABLED,
                local_llm_model=settings.LOCAL_LLM_MODEL # 🔥 Habilitar uso de ProductReference
            )
            
            rag_agent = WorkingAdvancedRAGAgent(config=rag_config)
            print("🧠 Agente RAG avanzado inicializado")
            
        except ImportError as e:
            if verbose:
                print(f"⚠️ RAG avanzado no disponible: {e}")
            print("⚠️ RAG avanzado no disponible, usando simple...")
            try:
                from src.core.rag.basic.retriever import Retriever
                from src.core.rag.basic.RAG import SimpleRAG
                
                retriever = Retriever(
                    index_path=settings.VECTOR_INDEX_PATH,
                    embedding_model=settings.EMBEDDING_MODEL,
                    device=settings.DEVICE
                )
                
                # Construir índice si no existe
                if not retriever.index_exists():
                    print("🔧 Construyendo índice...")
                    retriever.build_index(products)
                
                rag_agent = SimpleRAG(retriever=retriever)
                print("🧠 Agente RAG simple inicializado")
                
            except ImportError as e:
                print(f"❌ RAG simple no disponible: {e}")
                return
        
        # Gestor de usuarios
        user_manager = UserManager()
        user_profile = user_manager.create_user_profile(
            age=25,
            gender="male",
            country="Spain",
            language="es"
        )
        print(f"👤 Usuario: {user_profile.user_id}")
        
        # Loop interactivo
        print("\n💡 Escribe 'exit' para salir, 'stats' para estadísticas")
        print("="*50)
        
        while True:
            try:
                query = input("\n🔍 Tu consulta: ").strip()
                
                if query.lower() == 'exit':
                    print("👋 ¡Hasta luego!")
                    break
                
                if query.lower() == 'stats':
                    print(f"\n📊 ESTADÍSTICAS:")
                    print(f"   • Productos totales: {len(products)}")
                    print(f"   • ML habilitado: {settings.ML_ENABLED}")
                    print(f"   • LLM habilitado: {settings.LOCAL_LLM_ENABLED}")
                    
                    # 🔥 Estadísticas de ProductReference
                    try:
                        from src.core.data.product_reference import ProductClassHolder
                        if ProductClassHolder.is_available():
                            print(f"   • ProductReference: ✅ CONFIGURADO")
                        else:
                            print(f"   • ProductReference: ⚠️  NO CONFIGURADO")
                    except:
                        print(f"   • ProductReference: ❌ ERROR")
                    continue
                
                if not query:
                    continue
                
                print(f"\n🔍 Buscando: '{query}'...")
                
                # Procesar consulta
                if hasattr(rag_agent, 'process_query'):
                    # RAG avanzado
                    response = rag_agent.process_query(query, user_profile.user_id)
                    
                    if isinstance(response, dict):
                        answer = response.get('answer', 'Sin respuesta')
                        products_result = response.get('products', [])
                    else:
                        answer = str(response)
                        products_result = []
                else:
                    # RAG simple
                    products_result = rag_agent.search(query, top_k=5)
                    answer = f"Encontré {len(products_result)} productos"
                
                # Mostrar resultados
                print(f"\n🤖 {answer}")
                
                if products_result:
                    print(f"\n📦 Resultados:")
                    for i, product in enumerate(products_result[:5], 1):
                        # 🔥 Manejar ProductReference si está disponible
                        try:
                            from src.core.data.product_reference import ProductReference
                            if isinstance(product, ProductReference):
                                # Es un ProductReference
                                title = product.title
                                price = product.price
                                category = product.metadata.get('main_category', 'General')
                                source = product.source
                                ml_indicator = "🔥" if product.is_ml_processed else ""
                            else:
                                # Producto normal
                                if hasattr(product, 'title'):
                                    title = product.title
                                    price = getattr(product, 'price', 0.0)
                                    category = getattr(product, 'main_category', 'General')
                                    source = "rag"
                                    ml_indicator = "🔥" if getattr(product, 'ml_processed', False) else ""
                                elif isinstance(product, dict):
                                    title = product.get('title', 'Sin título')
                                    price = product.get('price', 0.0)
                                    category = product.get('main_category', 'General')
                                    source = product.get('source', 'rag')
                                    ml_indicator = "🔥" if product.get('ml_processed', False) else ""
                                else:
                                    title = str(product)[:50]
                                    price = 0.0
                                    category = 'General'
                                    source = 'unknown'
                                    ml_indicator = ""
                        except:
                            # Fallback simple
                            title = str(product)[:50] if hasattr(product, '__str__') else str(product)[:50]
                            price = 0.0
                            category = 'General'
                            source = 'unknown'
                            ml_indicator = ""
                        
                        print(f"  {i}. {title[:60]}{ml_indicator}")
                        if price:
                            print(f"     💰 ${price:.2f}")
                        if category:
                            print(f"     🏷️ {category}")
                        if verbose:
                            print(f"     📍 Source: {source}")
                
                # Feedback simple
                try:
                    feedback = input("\n¿Fue útil? (s/n/skip): ").strip().lower()
                    if feedback == 's':
                        print("✅ ¡Gracias!")
                    elif feedback == 'n':
                        print("⚠️ Lo sentimos, mejoraremos")
                except:
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

# =====================================================
#  COMANDO TRAIN
# =====================================================
def run_train(args):
    """Comando para entrenar modelos ML"""
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

# =====================================================
#  COMANDO ML
# =====================================================
def run_ml_stats():
    """Mostrar estadísticas ML."""
    print("\n🤖 ESTADÍSTICAS ML")
    print("="*50)
    
    print(f"📊 CONFIGURACIÓN ML:")
    print(f"   • Estado: {'✅ HABILITADO' if settings.ML_ENABLED else '❌ DESHABILITADO'}")
    
    if settings.ML_ENABLED:
        print(f"   • Características: {', '.join(settings.ML_FEATURES)}")
        print(f"   • Modelo embeddings: {settings.ML_EMBEDDING_MODEL}")
        print(f"   • Peso ML: {settings.ML_WEIGHT}")
        print(f"   • Categorías: {', '.join(settings.ML_CATEGORIES[:3])}...")
    
    # 🔥 Verificar ProductReference
    try:
        from src.core.data.product_reference import ProductClassHolder
        if ProductClassHolder.is_available():
            print(f"   • ProductReference: ✅ CONFIGURADO")
        else:
            print(f"   • ProductReference: ⚠️  NO CONFIGURADO")
    except Exception as e:
        print(f"   • ProductReference: ❌ ERROR: {e}")
    
    # Cargar algunos productos para estadísticas
    try:
        from src.core.data.loader import DataLoader
        
        loader = DataLoader(
            raw_dir=settings.RAW_DIR,
            processed_dir=settings.PROC_DIR
        )
        
        products = loader.load_data()[:50]  # Primeros 50
        
        if products:
            # Contar productos con ML
            ml_count = sum(1 for p in products if getattr(p, 'ml_processed', False))
            embed_count = sum(1 for p in products if getattr(p, 'embedding', None))
            cat_count = sum(1 for p in products if getattr(p, 'predicted_category', None))
            
            print(f"\n📈 ESTADÍSTICAS PRODUCTOS (muestra de {len(products)}):")
            print(f"   • Procesados con ML: {ml_count} ({ml_count/len(products)*100:.1f}%)")
            print(f"   • Con embeddings: {embed_count}")
            print(f"   • Con categorías predichas: {cat_count}")
            
            # 🔥 Probar ProductReference en algunos productos
            try:
                from src.core.data.product_reference import ProductReference
                ref_count = 0
                for product in products[:5]:
                    try:
                        ref = ProductReference.from_product(product)
                        ref_count += 1
                    except:
                        pass
                print(f"   • Compatible con ProductReference: {ref_count}/5")
            except:
                print(f"   • Compatible con ProductReference: ❌ NO DISPONIBLE")
        
    except Exception as e:
        print(f"⚠️ Error cargando productos: {e}")

def run_ml_fix_categories():
    """Reparar categorías de productos automáticamente."""
    print("\n🔧 REPARANDO CATEGORÍAS DE PRODUCTOS")
    print("="*50)
    
    try:
        from scripts.fix_categories import fix_products_categories
        from src.core.config import settings
        
        products_file = settings.PROC_DIR / "products.json"
        fix_products_categories(products_file)
        
        print("✅ Categorías reparadas")
        print("\n💡 Recomendación: Ejecuta 'python main.py index' para reconstruir el índice")
        
    except ImportError:
        print("❌ Script fix_categories.py no encontrado")
    except Exception as e:
        print(f"❌ Error: {e}")

# =====================================================
#  COMANDO TEST
# =====================================================
def run_test_command(args):
    """Comandos de testing."""
    print("\n🧪 COMANDOS DE TEST")
    print("="*50)
    
    if args.subcommand == "product-ref":
        print("\n🔍 TEST DE ProductReference")
        print("-"*30)
        
        try:
            # Test básico de ProductReference
            from src.core.data.product_reference import (
                ProductReference, 
                ProductClassHolder,
                create_ml_enhanced_reference
            )
            
            print(f"✅ ProductClassHolder disponible: {ProductClassHolder.is_available()}")
            
            # Crear un producto de prueba
            class MockProduct:
                def __init__(self):
                    self.id = "test_123"
                    self.title = "Producto de prueba"
                    self.price = 99.99
                    self.description = "Descripción de prueba"
                    self.main_category = "Electronics"
                    self.ml_processed = True
                    self.embedding = [0.1] * 384
                    self.predicted_category = "Electronics"
                
                def to_metadata(self):
                    return {
                        "title": self.title,
                        "price": self.price,
                        "main_category": self.main_category,
                        "ml_processed": self.ml_processed
                    }
            
            # Test 1: Crear ProductReference
            test_product = MockProduct()
            ref = ProductReference.from_product(test_product, source="ml_enhanced")
            print(f"✅ ProductReference creado: {ref}")
            print(f"   • ID: {ref.id}")
            print(f"   • Title: {ref.title}")
            print(f"   • Source: {ref.source}")
            print(f"   • ML procesado: {ref.is_ml_processed}")
            print(f"   • Tiene embedding: {ref.has_embedding}")
            
            # Test 2: Convertir a dict
            ref_dict = ref.to_dict()
            print(f"✅ Convertido a dict: {len(ref_dict)} campos")
            
            # Test 3: Crear desde dict
            ref2 = ProductReference.from_dict(ref_dict)
            print(f"✅ Reconstruido desde dict: {ref2}")
            
            # Test 4: Test ML enhanced
            ml_ref = create_ml_enhanced_reference(
                test_product, 
                ml_score=0.9,
                ml_data={"confidence": 0.95, "similarity_score": 0.87}
            )
            print(f"✅ ML enhanced reference: {ml_ref}")
            print(f"   • ML confidence: {ml_ref.ml_confidence}")
            
            print("\n🎉 Todos los tests de ProductReference PASADOS")
            
        except Exception as e:
            print(f"❌ Error en test ProductReference: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.subcommand == "serialization":
        print("\n🔍 TEST DE SERIALIZACIÓN")
        print("-"*30)
        
        try:
            from src.core.utils.serialization_utils import EmbeddingSerializer
            
            # Test de serialización
            test_embedding = [0.1 * i for i in range(384)]
            
            for method in ["b64pickle", "b64json", "json", "compressed"]:
                serialized = EmbeddingSerializer.serialize_embedding(test_embedding, method)
                deserialized = EmbeddingSerializer.deserialize_embedding(serialized)
                valid = EmbeddingSerializer.validate_embedding(deserialized)
                
                print(f"   • {method}: {'✅' if valid else '❌'} "
                      f"(len: {len(serialized)}, valid: {valid})")
            
            print("✅ Test de serialización completado")
            
        except Exception as e:
            print(f"❌ Error en test de serialización: {e}")
    
    elif args.subcommand == "ml-processor":
        print("\n🔍 TEST DE ML PROCESSOR")
        print("-"*30)
        
        try:
            # Importar después de arreglar el problema circular
            from src.core.data.ml_processor import (
                get_ml_preprocessor,
                create_ml_preprocessor_with_context,
                process_with_memory_management
            )
            
            print("✅ Módulo ml_processor importado correctamente")
            
            # Test 1: Preprocesador básico
            print("\n🧪 Test 1: Preprocesador básico")
            preprocessor = get_ml_preprocessor(verbose=True)
            
            test_product = {
                "id": "test_ml_1",
                "title": "Laptop gaming ASUS ROG",
                "description": "Laptop para juegos con RTX 4080, 32GB RAM, 1TB SSD",
                "brand": "ASUS",
                "price": 1999.99
            }
            
            result = preprocessor.preprocess_product(test_product, enable_ml=True)
            print(f"✅ Producto procesado: {result.get('title')}")
            print(f"   • Categoría predicha: {result.get('predicted_category', 'N/A')}")
            print(f"   • Tiene embedding: {'embedding' in result}")
            
            # Estadísticas de memoria
            stats = preprocessor.get_cache_stats()
            print(f"   • Memoria usada: {stats.get('memory_usage_peak_mb', 0):.1f}MB")
            
            # Limpiar memoria
            preprocessor.cleanup_memory()
            print("✅ Memoria liberada")
            
            # Test 2: Context manager
            print("\n🧪 Test 2: Context manager")
            with create_ml_preprocessor_with_context(verbose=True) as preprocessor2:
                result2 = preprocessor2.preprocess_product(test_product, enable_ml=True)
                print(f"✅ Procesado con context manager: {result2.get('title')}")
                print(f"   • ML procesado: {result2.get('ml_processed', False)}")
            
            print("✅ Context manager completado (memoria liberada automáticamente)")
            
            # Test 3: Procesamiento por lotes
            print("\n🧪 Test 3: Procesamiento por lotes")
            test_products = [
                {"id": f"test_{i}", "title": f"Producto {i}", "description": f"Descripción {i}"}
                for i in range(5)
            ]
            
            results = process_with_memory_management(
                test_products,
                batch_size=2,
                verbose=False
            )
            print(f"✅ {len(results)} productos procesados en lote")
            
            print("\n🎉 Todos los tests de ML Processor PASADOS")
            
        except Exception as e:
            print(f"❌ Error en test ML Processor: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("ℹ️ Subcomandos de test disponibles:")
        print("   • test product-ref     - Test de ProductReference")
        print("   • test serialization   - Test de serialización")
        print("   • test ml-processor    - Test de ML Processor")

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
            # Manejar argumentos ML
            ml_enabled = None
            if hasattr(args, 'ml'):
                if args.ml is True:
                    ml_enabled = True
                elif args.ml is False:
                    ml_enabled = False
            
            run_rag(
                data_dir=args.data_dir, 
                ml_enabled=ml_enabled,
                verbose=args.verbose,
                product_ref_debug=args.product_ref_debug
            )
        elif args.command == "verify":
            try:
                from scripts.verify_system import main as verify_main
                verify_main()
            except ImportError:
                print("❌ Script verify_system.py no encontrado")
                print("⚠️ Ejecuta: python scripts/verify_system.py directamente")    
        elif args.command == "train":
            run_train(args)
            
        elif args.command == "ml":
            if args.subcommand == "repair":
                try:
                    from scripts.repair_ml_embeddings import repair_ml_embeddings
                    repair_ml_embeddings()
                except ImportError:
                    print("❌ Script repair_ml_embeddings.py no encontrado")
            elif args.subcommand == "test":
                try:
                    from scripts.repair_ml_embeddings import test_rag_with_ml
                    test_rag_with_ml()
                except ImportError:
                    print("❌ No se pudo ejecutar test de RAG con ML")
            else:
                run_ml_stats()
        
        elif args.command == "test":
            run_test_command(args)
            
        else:
            print(f"❌ Comando no reconocido: {args.command}")
            sys.exit(1)
        
        print("\n✅ Ejecución completada")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Ejecución interrumpida")
        sys.exit(0)
    except ImportError as e:
        print(f"\n❌ Error importando módulo: {e}")
        print("⚠️ Verifica que todos los módulos estén instalados")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)