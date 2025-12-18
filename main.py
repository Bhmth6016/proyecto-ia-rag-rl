#!/usr/bin/env python3
# main.py - Sistema de Recomendación E-Commerce (VERSIÓN MEJORADA)

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================
#  INICIALIZACIÓN DEL SISTEMA
# =====================================================
def init_system():
    """Inicializar sistema de feedback y directorios."""
    try:
        from scripts.init_feedback_system import init_feedback_system
        init_feedback_system()
        print("✅ Sistema de feedback inicializado")
    except ImportError:
        print("⚠️  Script de inicialización no encontrado - creando directorios básicos...")
        # Crear directorios mínimos
        import os
        os.makedirs("data/processed/historial", exist_ok=True)
        os.makedirs("data/feedback", exist_ok=True)
        os.makedirs("data/processed/user_profiles", exist_ok=True)
        print("📁 Directorios básicos creados")
    except Exception as e:
        print(f"⚠️  Error inicializando sistema: {e}")

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
    print("║     💾 Sistema de Historial de Conversaciones                   ║")
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

# =====================================================
#  FUNCIONES DE HISTORIAL DE CONVERSACIONES
# =====================================================

def _save_conversation_to_historial(
    query: str, 
    answer: str, 
    feedback_rating: Optional[int], 
    products_shown: List[str],
    user_id: str
):
    """Guarda la conversación en el historial."""
    try:
        from datetime import datetime
        import json
        from pathlib import Path
        
        # Crear directorio de historial
        historial_dir = Path("data/processed/historial")
        historial_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear entrada de conversación
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": f"session_{datetime.now().strftime('%Y%m%d')}",
            "user_id": user_id,
            "query": query,
            "response": answer,
            "feedback": feedback_rating if feedback_rating else None,
            "products_shown": products_shown[:3],  # Guardar primeros 3 productos
            "source": "rag_interactive",
            "processed": False
        }
        
        # Nombre del archivo basado en fecha
        date_str = datetime.now().strftime("%Y-%m-%d")
        file_path = historial_dir / f"conversation_{date_str}.json"
        
        # Cargar conversaciones existentes o crear nueva lista
        conversations = []
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    conversations = json.load(f)
                    if not isinstance(conversations, list):
                        conversations = []
            except:
                conversations = []
        
        # Agregar nueva conversación
        conversations.append(conversation_entry)
        
        # Guardar (limitar a últimas 100 conversaciones)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversations[-100:], f, indent=2, ensure_ascii=False)
        
        logger.debug(f"💾 Conversación guardada en historial ({len(conversations)} entradas)")
        
    except Exception as e:
        logger.error(f"⚠️  Error guardando historial: {e}")

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
  %(prog)s test historial             # Ver historial de conversaciones
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
    
    elif args.subcommand == "historial":
        print("\n📚 VERIFICANDO HISTORIAL DE CONVERSACIONES")
        print("-"*40)
        
        historial_dir = Path("data/processed/historial")
        
        if not historial_dir.exists():
            print("❌ Directorio de historial no existe")
            return
        
        total_conversations = 0
        for file in historial_dir.glob("conversation_*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    conversations = json.load(f)
                    count = len(conversations)
                    total_conversations += count
                    print(f"📄 {file.name}: {count} conversaciones")
                    
                    if count > 0:
                        last_convo = conversations[-1]
                        print(f"   Última: '{last_convo.get('query', '')[:50]}...'")
                        print(f"   Feedback: {last_convo.get('feedback', 'N/A')}")
            except Exception as e:
                print(f"⚠️  Error leyendo {file}: {e}")
        
        print(f"\n📊 TOTAL: {total_conversations} conversaciones en historial")
    
    else:
        print("ℹ️ Subcomandos de test disponibles:")
        print("   • test product-ref     - Test de ProductReference")
        print("   • test rag-agent       - Test de WorkingAdvancedRAGAgent")
        print("   • test historial       - Ver historial de conversaciones")

# =====================================================
#  FUNCIONES AUXILIARES MEJORADAS
# =====================================================

def _extract_product_info(item: Any) -> Dict[str, Any]:
    """
    Extrae información de un producto o ProductReference de forma segura.
    
    Args:
        item: Puede ser Product, ProductReference, o cualquier objeto
        
    Returns:
        Diccionario con información extraída
    """
    try:
        item_type = type(item).__name__
        
        if "ProductReference" in item_type:
            # Es un ProductReference
            title = item.title[:80] if hasattr(item, 'title') else str(item)[:80]
            price = item.price if hasattr(item, 'price') else 0.0
            category = "General"
            
            # Obtener categoría de múltiples fuentes
            if hasattr(item, 'ml_features') and item.ml_features:
                category = item.ml_features.get('predicted_category', 'General')
            elif hasattr(item, 'metadata') and item.metadata:
                category = item.metadata.get('main_category', 'General')
            
            score = item.score if hasattr(item, 'score') else 0.0
            source = item.source if hasattr(item, 'source') else "ProductReference"
            
            return {
                "type": "ProductReference",
                "title": title,
                "price": price,
                "category": category,
                "score": score,
                "source": source,
                "object": item
            }
            
        elif "Product" in item_type:
            # Es un objeto Product
            title = getattr(item, 'title', str(item))[:80]
            price = getattr(item, 'price', 0.0)
            category = getattr(item, 'main_category', 'General')
            predicted_category = getattr(item, 'predicted_category', None)
            category = predicted_category if predicted_category else category
            
            # Product no tiene score o source nativo
            ml_processed = getattr(item, 'ml_processed', False)
            
            return {
                "type": "Product",
                "title": title,
                "price": price,
                "category": category,
                "score": 0.0,  # Product no tiene score
                "source": "ml" if ml_processed else "basic",
                "object": item
            }
            
        else:
            # Tipo desconocido
            title = str(item)[:80]
            return {
                "type": "Unknown",
                "title": title,
                "price": 0.0,
                "category": "Unknown",
                "score": 0.0,
                "source": "unknown",
                "object": item
            }
            
    except Exception as e:
        logger.debug(f"Error extrayendo información del producto: {e}")
        return {
            "type": "Error",
            "title": f"Error: {str(e)[:50]}",
            "price": 0.0,
            "category": "Error",
            "score": 0.0,
            "source": "error",
            "object": item
        }

def _extract_product_ids_safely(products_result: List[Any], max_items: int = 3) -> List[str]:
    """
    Extrae IDs de productos de forma segura, manejando objetos None.
    
    Args:
        products_result: Lista de productos o referencias
        max_items: Número máximo de IDs a extraer
    
    Returns:
        Lista de IDs de productos como strings
    """
    product_ids = []
    
    for i, item in enumerate(products_result[:max_items]):
        try:
            product_info = _extract_product_info(item)
            obj = product_info.get("object")
            
            if obj is None:
                product_ids.append(f"unknown_{i}")
                continue
            
            # Intentar obtener ID de múltiples formas
            product_id = None
            
            # 1. Buscar atributo 'id'
            if hasattr(obj, 'id') and obj.id is not None:
                product_id = str(obj.id)
            # 2. Buscar en metadata
            elif hasattr(obj, 'metadata') and isinstance(obj.metadata, dict):
                product_id = obj.metadata.get('id')
            # 3. Buscar en diccionario
            elif isinstance(obj, dict):
                product_id = obj.get('id')
            
            if product_id:
                product_ids.append(product_id)
            else:
                product_ids.append(f"item_{i}")
                
        except Exception as e:
            logger.debug(f"Error extrayendo ID de producto {i}: {e}")
            product_ids.append(f"error_{i}")
    
    return product_ids
def _get_category_emoji(category: str) -> str:
    """Devuelve emoji apropiado para la categoría."""
    emoji_map = {
        'Electronics': '📱',
        'Books': '📚',
        'Clothing': '👕',
        'Home & Kitchen': '🏠',
        'Sports & Outdoors': '⚽',
        'Beauty': '💄',
        'Toys & Games': '🧸',
        'Automotive': '🚗',
        'Office Products': '💼',
        'Video Games': '🎮',
        'General': '📦'
    }
    
    for key, emoji in emoji_map.items():
        if key.lower() in category.lower():
            return emoji
    
    return '📦'  # Emoji por defecto


# =====================================================
#  FUNCIÓN DE INICIALIZACIÓN DE FEEDBACK PROCESSOR
# =====================================================
def init_feedback_processor():
    """Inicializar FeedbackProcessor global."""
    try:
        from src.core.rag.advanced.feedback_processor import FeedbackProcessor
        return FeedbackProcessor()
    except ImportError:
        logger.debug("⚠️  FeedbackProcessor no disponible")
        return None
    except Exception as e:
        logger.debug(f"⚠️  Error inicializando FeedbackProcessor: {e}")
        return None


# =====================================================
#  RUN_RAG - VERSIÓN REFACTORIZADA Y MEJORADA
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
    
    # 🔥 Inicializar FeedbackProcessor global
    feedback_processor = init_feedback_processor()
    if feedback_processor:
        print("📊 FeedbackProcessor global inicializado")
    
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
            user_profile = user_manager.get_user_profile(user_id)
            if user_profile:
                print(f"👤 Usuario existente: {user_id}")
            else:
                user_profile = user_manager.create_user_profile(
                    age=30,
                    gender="female",
                    country="Spain",
                    language="es"
                )
                user_id = user_profile.user_id
                print(f"👤 Usuario registrado: {user_id}")
        
        # 🔥 INICIALIZAR AGENTE
        print("\n🚀 Inicializando agente...")
        
        working_rag_agent = None
        basic_retriever = None
        
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
                enable_reranking=(not no_rlhf),
                ml_features=list(settings.ML_FEATURES),
                use_ml_embeddings=settings.ML_ENABLED and 'embedding' in settings.ML_FEATURES,
                ml_embedding_weight=settings.ML_WEIGHT
            )
            
            # Crear agente avanzado
            working_rag_agent = WorkingAdvancedRAGAgent(config=rag_config)
            
            # Deshabilitar componentes si se solicita
            if no_collaborative:
                working_rag_agent._collaborative_filter = None
                print("🤝 Collaborative Filter: ❌ DESHABILITADO")
            
            if no_rlhf:
                working_rag_agent.rlhf_model = None
                working_rag_agent.rlhf_pipeline = None
                print("🧠 RLHF: ❌ DESHABILITADO")
            
            # Mostrar configuración del agente
            print(f"\n📡 CONFIGURACIÓN RAG AGENT:")
            print(f"   • Tipo: WorkingAdvancedRAGAgent")
            print(f"   • Modo: {rag_mode.value}")
            print(f"   • ML: {'✅' if rag_config.ml_enabled else '❌'}")
            print(f"   • LLM: {'✅' if rag_config.local_llm_enabled else '❌'}")
            print(f"   • RLHF: {'✅' if not no_rlhf else '❌'}")
            print(f"   • Collaborative Filter: {'✅' if not no_collaborative else '❌'}")
            
            # Verificar que el índice existe
            if not working_rag_agent.retriever.index_exists():
                print("\n🔧 Índice no encontrado, construyendo...")
                working_rag_agent.retriever.build_index(products)
                print(f"✅ Índice construido con {len(products)} productos")
            
        except ImportError as e:
            print(f"⚠️  No se pudo importar WorkingAdvancedRAGAgent: {e}")
            print("🔧 Fallback a Retriever básico...")
            
            # Fallback a Retriever básico
            from src.core.rag.basic.retriever import Retriever
            
            basic_retriever = Retriever(
                index_path=settings.VECTOR_INDEX_PATH,
                embedding_model=settings.EMBEDDING_MODEL,
                device=settings.DEVICE
            )
            
            if not basic_retriever.index_exists():
                print("🔧 Construyendo índice...")
                basic_retriever.build_index(products)
                print(f"✅ Índice construido con {len(products)} productos")
            
            print(f"\n📡 CONFIGURACIÓN RAG AGENT:")
            print(f"   • Tipo: Retriever Básico")
            print(f"   • ML: {'✅' if settings.ML_ENABLED else '❌'}")
            print(f"   • Modelo embeddings: {settings.EMBEDDING_MODEL}")
            
        except Exception as e:
            print(f"❌ Error inicializando agente: {e}")
            raise
        
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
                    print("   • historial - Mostrar estadísticas de historial")
                    if working_rag_agent is not None:
                        print("   • clear - Limpiar caché (solo WorkingAdvancedRAGAgent)")
                    continue
                
                if query.lower() == 'stats':
                    print(f"\n📊 ESTADÍSTICAS:")
                    print(f"   • Productos totales: {len(products)}")
                    print(f"   • ML habilitado: {settings.ML_ENABLED}")
                    print(f"   • LLM habilitado: {settings.LOCAL_LLM_ENABLED}")
                    print(f"   • Tipo de agente: {'WorkingAdvancedRAGAgent' if working_rag_agent else 'Retriever Básico'}")
                    
                    if working_rag_agent:
                        print(f"   • Modo RAG: {mode}")
                        print(f"   • RLHF: {'✅' if not no_rlhf else '❌'}")
                        print(f"   • Collaborative Filter: {'✅' if not no_collaborative else '❌'}")
                    
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
                
                if query.lower() == 'historial':
                    historial_dir = Path("data/processed/historial")
                    if historial_dir.exists():
                        total_files = len(list(historial_dir.glob("*.json")))
                        print(f"📚 Historial: {total_files} archivos de conversación")
                    else:
                        print("📚 Historial: Directorio no existe aún")
                    continue
                
                if query.lower() == 'clear':
                    if working_rag_agent is not None:
                        working_rag_agent.clear_cache()
                        print("🗑️  Cache limpiado")
                    else:
                        print("⚠️  Comando 'clear' solo disponible para WorkingAdvancedRAGAgent")
                    continue
                
                if not query:
                    continue
                
                print(f"\n🔍 Buscando: '{query}'...")
                
                # 🔥 PROCESAR CONSULTA SEGÚN TIPO DE AGENTE
                products_result = []
                answer = ""
                
                if working_rag_agent is not None:
                    # Usar WorkingAdvancedRAGAgent
                    try:
                        response = working_rag_agent.process_query(query, user_id)
                        
                        if isinstance(response, dict):
                            answer = response.get('answer', 'Sin respuesta')
                            products_result = response.get('products', [])
                            
                            if verbose:
                                stats = response.get('stats', {})
                                print(f"\n📊 ESTADÍSTICAS PROCESAMIENTO:")
                                print(f"   • Tiempo: {stats.get('processing_time', 0):.2f}s")
                                print(f"   • Resultados iniciales: {stats.get('initial_results', 0)}")
                                print(f"   • Resultados finales: {stats.get('final_results', 0)}")
                                print(f"   • ML mejorado: {stats.get('ml_enhanced', False)}")
                                print(f"   • Re-ranking: {stats.get('reranking_enabled', False)}")
                        else:
                            answer = str(response)
                            
                    except Exception as e:
                        print(f"❌ Error en process_query: {e}")
                        answer = "Error procesando consulta con WorkingAdvancedRAGAgent"
                        
                elif basic_retriever is not None:
                    # Usar Retriever básico
                    try:
                        # El Retriever tiene el método retrieve()
                        products_result = basic_retriever.retrieve(query=query, k=max_results)
                        
                        # Crear una respuesta simple
                        if products_result:
                            product_titles = []
                            for p in products_result[:3]:
                                if hasattr(p, 'title'):
                                    product_titles.append(p.title[:50])
                                else:
                                    product_titles.append(str(p)[:50])
                            
                            if product_titles:
                                answer = f"Encontré {len(products_result)} productos para '{query}'. Los más relevantes: {', '.join(product_titles)}"
                            else:
                                answer = f"Encontré {len(products_result)} productos para '{query}'"
                        else:
                            answer = f"No encontré productos para '{query}'"
                            
                    except Exception as e:
                        print(f"❌ Error en búsqueda básica: {e}")
                        answer = "Error en búsqueda básica"
                else:
                    print("❌ No hay agente configurado")
                    continue
                
                # Mostrar respuesta
                print(f"\n🤖 {answer}")
                
                # Mostrar resultados
                product_ids = []
                if products_result:
                    print(f"\n📦 RESULTADOS ({len(products_result)} encontrados):")
                    
                    for i, item in enumerate(products_result[:max_results], 1):
                        try:
                            # Extraer información usando función auxiliar
                            product_info = _extract_product_info(item)
                            
                            title = product_info["title"]
                            price = product_info["price"]
                            category = product_info["category"]
                            score = product_info["score"]
                            source = product_info["source"]
                            item_type = product_info["type"]
                            
                            # Obtener emoji para categoría
                            emoji = _get_category_emoji(category)
                            
                            # Guardar ID del producto para historial
                            if hasattr(product_info["object"], 'id'):
                                product_ids.append(str(product_info["object"].id))
                            else:
                                product_ids.append(f"item_{i}")
                            
                            # Mostrar producto
                            print(f"  {emoji} {i}. {title}")
                            print(f"     💰 ${price:.2f} | 🏷️ {category}")
                            
                            if verbose:
                                print(f"     ⭐ Score: {score:.3f} | 📍 Source: {source} ({item_type})")
                            
                            # Línea separadora
                            if i < min(len(products_result), max_results):
                                print("     " + "-" * 40)
                                
                        except Exception as e:
                            print(f"  {i}. Error mostrando producto: {e}")
                            product_ids.append(f"error_{i}")
                
                # 🔥 SECCIÓN DE FEEDBACK MEJORADA
                feedback_rating = None
                try:
                    print(f"\n💬 ¿Fue útil esta respuesta?")
                    feedback_input = input("   (s) Sí | (n) No | (skip) Saltar: ").strip().lower()
                    
                    if feedback_input == 's':
                        feedback_rating = 5
                        print("     ✅ ¡Gracias por tu feedback positivo!")
                    elif feedback_input == 'n':
                        feedback_rating = 1
                        print("     ⚠️  Lo sentimos, mejoraremos")
                    else:
                        print("     ℹ️  Feedback omitido")
                        feedback_rating = None
                    
                    # 🔥 GUARDAR FEEDBACK EN MÚLTIPLES LUGARES
                    
                    # 1. Guardar feedback en agentes específicos
                    if feedback_rating:
                        if working_rag_agent is not None:
                            # Usar el método de feedback del agente avanzado
                            working_rag_agent.log_feedback(
                                query=query,
                                answer=answer,
                                rating=feedback_rating,
                                user_id=user_id
                            )
                        
                        # 2. Extraer IDs de productos para historial (CORRECCIÓN: manejar None)
                        feedback_product_ids = []
                        for i, item in enumerate(products_result[:3]):
                            product_info = _extract_product_info(item)
                            obj = product_info.get("object")
                            
                            # 🔥 CORRECCIÓN: Manejar objeto None de forma segura
                            if obj and hasattr(obj, 'id') and obj.id is not None:
                                feedback_product_ids.append(str(obj.id))
                            else:
                                # Usar un ID generado basado en el índice
                                feedback_product_ids.append(f"item_{i}")

                        # 3. Guardar en perfil de usuario de forma segura
                        try:
                            # 🔥 CORRECCIÓN: Usar el método add_feedback_event con los parámetros CORRECTOS
                            # según la definición en user_models.py
                            user_profile.add_feedback_event(
                                query=query,                    # str
                                response=answer,                # str  
                                rating=feedback_rating,         # int
                                products_shown=feedback_product_ids[:3] if feedback_product_ids else [],  # List[str]
                                selected_product=feedback_product_ids[0] if feedback_product_ids else None  # Optional[str]
                            )
                            
                            # Guardar perfil
                            if hasattr(user_manager, 'save_user_profile'):
                                user_manager.save_user_profile(user_profile)
                                print("     👤 Feedback guardado en perfil de usuario")
                                
                        except Exception as e:
                            print(f"     ⚠️  Error guardando feedback en perfil: {e}")
                            # Fallback: Guardar en archivo simple
                            try:
                                import json
                                from datetime import datetime
                                
                                feedback_data = {
                                    "timestamp": datetime.now().isoformat(),
                                    "user_id": user_id,
                                    "query": query,
                                    "response": answer,
                                    "rating": feedback_rating,
                                    "products_shown": feedback_product_ids[:3] if feedback_product_ids else [],
                                    "selected_product": feedback_product_ids[0] if feedback_product_ids else None
                                }
                                
                                feedback_dir = Path("data/feedback/user_feedback")
                                feedback_dir.mkdir(parents=True, exist_ok=True)
                                
                                file_path = feedback_dir / f"feedback_{user_id}_{datetime.now().strftime('%Y%m%d')}.json"
                                
                                # Cargar feedback existente
                                existing_feedback = []
                                if file_path.exists():
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        try:
                                            existing_feedback = json.load(f)
                                        except:
                                            existing_feedback = []
                                
                                # Agregar nuevo feedback
                                existing_feedback.append(feedback_data)
                                
                                # Guardar (mantener solo últimos 100)
                                with open(file_path, 'w', encoding='utf-8') as f:
                                    json.dump(existing_feedback[-100:], f, indent=2, ensure_ascii=False)
                                
                                print("     💾 Feedback guardado en archivo (fallback)")
                                
                            except Exception as fallback_error:
                                print(f"     ❌ Error en fallback también: {fallback_error}")
                    
                    # 4. GUARDAR EN HISTORIAL DE CONVERSACIONES (CRÍTICO)
                    # Usar los product_ids que ya extrajimos antes en la variable principal
                    _save_conversation_to_historial(
                        query=query,
                        answer=answer,
                        feedback_rating=feedback_rating,
                        products_shown=product_ids,  # Usar la variable product_ids que ya tenemos
                        user_id=user_id
                    )
                    
                    # 5. Procesar feedback con FeedbackProcessor global
                    if feedback_processor and feedback_rating:
                        try:
                            feedback_processor.save_feedback(
                                query=query,
                                answer=answer,
                                rating=feedback_rating,
                                retrieved_docs=product_ids,  # Usar product_ids principal
                                extra_meta={
                                    "user_id": user_id,
                                    "mode": mode,
                                    "ml_enabled": settings.ML_ENABLED,
                                    "agent_type": "WorkingAdvancedRAGAgent" if working_rag_agent else "BasicRetriever"
                                }
                            )
                            print("     📝 Feedback procesado en tiempo real")
                        except Exception as e:
                            logger.debug(f"     ⚠️  Error procesando feedback: {e}")
                            
                except (KeyboardInterrupt, EOFError):
                    print("\n⚠️  Feedback interrumpido")
                    # Guardar conversación sin feedback
                    _save_conversation_to_historial(
                        query=query,
                        answer=answer,
                        feedback_rating=None,
                        products_shown=product_ids,
                        user_id=user_id
                    )
                except Exception as e:
                    logger.error(f"Error en sección de feedback: {e}")
                    # Guardar conversación con error
                    _save_conversation_to_historial(
                        query=query,
                        answer=answer,
                        feedback_rating=None,
                        products_shown=product_ids,
                        user_id=user_id
                    )
                
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
    print("   5. historial - Verificar historial de conversaciones")
    print("   6. exit - Salir")
    
    while True:
        try:
            choice = input("\n🔍 Elige una opción (1-6): ").strip()
            
            if choice == '1':
                try:
                    from src.core.rag.advanced.WorkingRAGAgent import test_rag_pipeline
                    result = test_rag_pipeline("smartphone barato")
                    print(f"✅ Test RAG completado: {result.get('products_found', 0)} productos")
                except ImportError:
                    print("❌ WorkingAdvancedRAGAgent no disponible")
                except Exception as e:
                    print(f"❌ Error: {e}")
                
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
                print("🧪 Procesamiento ML:")
                run_ml_stats()
                
            elif choice == '4':
                try:
                    from scripts.verify_system import main as verify_main
                    verify_main()
                except ImportError:
                    print("❌ Script verify_system.py no encontrado")
                    
            elif choice == '5':
                historial_dir = Path("data/processed/historial")
                if historial_dir.exists():
                    total_files = len(list(historial_dir.glob("*.json")))
                    print(f"📚 Historial: {total_files} archivos de conversación")
                    
                    # Mostrar archivos recientes
                    for file in sorted(historial_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                        try:
                            with open(file, 'r', encoding='utf-8') as f:
                                conversations = json.load(f)
                                print(f"📄 {file.name}: {len(conversations)} conversaciones")
                        except:
                            print(f"📄 {file.name}: Error leyendo")
                else:
                    print("📚 Historial: Directorio no existe aún")
                    
            elif choice == '6' or choice.lower() == 'exit':
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
    
    # Inicializar sistema
    init_system()
    
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