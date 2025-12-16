#!/usr/bin/env python3
# main.py - Sistema de Recomendación E-Commerce

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
    print("╔════════════════════════════════════════════════════════╗")
    print("║     🎯 Sistema de Recomendación E-Commerce            ║")
    print("║     🤖 Con procesamiento ML 100% Local                ║")
    print("║     🔥 Multi-categoría: Electrónicos, Ropa, Hogar...  ║")
    print("║     📦 ProductReference Configurado                   ║")
    print("╚════════════════════════════════════════════════════════╝")

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
        description="Sistema de Recomendación E-Commerce - ML Local",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s rag --mode enhanced        # ML completo con NLP
  %(prog)s rag --mode basic           # Solo búsqueda básica
  %(prog)s rag --mode balanced        # ML básico sin NLP
  
  %(prog)s index                      # Construir índice
  %(prog)s ml                         # Ver estadísticas ML
  %(prog)s ml repair                  # Reparar embeddings ML
  %(prog)s test product-ref           # Test ProductReference
  %(prog)s verify                     # Verificar sistema completo
        """
    )
    
    parser.add_argument(
        'command',
        choices=['rag', 'index', 'ml', 'train', 'test', 'verify'],
        help='Comando a ejecutar'
    )
    
    parser.add_argument('--mode', 
                       choices=['basic', 'enhanced', 'balanced'],
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
    
    # 🔥 Opciones específicas para ProductReference
    parser.add_argument('--product-ref-debug', action='store_true', 
                       help='Modo debug para ProductReference')
    
    return parser.parse_args()

# =====================================================
#  FUNCIONES CRÍTICAS MANTENIDAS
# =====================================================
def run_index(data_dir: Optional[str] = None, verbose: bool = False):
    """Versión simplificada pero funcional de run_index"""
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
        
        # Estadísticas básicas
        if settings.ML_ENABLED:
            ml_count = sum(1 for p in products if getattr(p, 'ml_processed', False))
            print(f"   • Con ML procesado: {ml_count}")
        
        # Construir índice
        from src.core.rag.basic.retriever import Retriever
        
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
    """Estadísticas ML simplificadas pero completas."""
    print("\n🤖 ESTADÍSTICAS ML")
    print("="*50)
    
    print(f"📊 CONFIGURACIÓN ML:")
    print(f"   • Estado: {'✅ HABILITADO' if settings.ML_ENABLED else '❌ DESHABILITADO'}")
    
    if settings.ML_ENABLED:
        print(f"   • Características: {', '.join(settings.ML_FEATURES)}")
        print(f"   • Modelo embeddings: {settings.ML_EMBEDDING_MODEL}")
    
    # 🔥 MANTENER: Verificar ProductReference
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
        
    except Exception as e:
        print(f"⚠️ Error cargando productos: {e}")

def run_train(args):
    """Comando para entrenar modelos ML - versión simplificada"""
    print("\n🤖 ENTRENAMIENTO DE MODELOS ML")
    print("="*50)
    
    if args.subcommand == "rlhf":
        print("⚠️  RLHF training temporalmente deshabilitado en esta versión")
        print("💡 Use la versión completa para entrenamiento RLHF")
        
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
    """Comandos de testing - versión simplificada."""
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
                    self.title = "Producto de prueba"
                    self.price = 99.99
                    self.main_category = "Electronics"
                    self.ml_processed = True
                
                def to_metadata(self):
                    return {
                        "title": self.title,
                        "price": self.price,
                        "main_category": self.main_category,
                        "ml_processed": self.ml_processed
                    }
            
            test_product = MockProduct()
            ref = ProductReference.from_product(test_product, source="test")
            
            print(f"✅ ProductReference creado: {ref}")
            print(f"   • ID: {ref.id}")
            print(f"   • Title: {ref.title}")
            print(f"   • Source: {ref.source}")
            print(f"   • ML procesado: {ref.is_ml_processed}")
            
            # Test básico de serialización
            ref_dict = ref.to_dict()
            print(f"✅ Convertido a dict: {len(ref_dict)} campos")
            
        except Exception as e:
            print(f"❌ Error en test ProductReference: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print("ℹ️ Subcomandos de test disponibles:")
        print("   • test product-ref     - Test de ProductReference")

# =====================================================
#  RUN_RAG - VERSIÓN HÍBRIDA MEJORADA
# =====================================================
def run_rag(data_dir: Optional[str] = None, 
           mode: str = "enhanced",
           verbose: bool = False,
           ml_enabled: Optional[bool] = None,
           product_ref_debug: bool = False):
    
    print(f"\n🧠 MODO RAG: {mode.upper()}")
    print("="*50)
    
    # 🔥 CORRECCIÓN: Aplicar modo del sistema
    from src.core.config import apply_system_mode, settings
    apply_system_mode(mode)
    
    print(f"\n📋 CONFIGURACIÓN APLICADA:")
    print(f"   • Modo: {settings.CURRENT_MODE}")
    print(f"   • ML: {'✅ HABILITADO' if settings.ML_ENABLED else '❌ DESHABILITADO'}")
    print(f"   • NLP: {'✅ HABILITADO' if settings.NLP_ENABLED else '❌ DESHABILITADO'}")
    print(f"   • LLM: {'🧠 ON' if settings.LOCAL_LLM_ENABLED else 'OFF'}")
    print(f"   • Ref. Productos: {'📦 ON' if settings.PRODUCT_REF_ENABLED else 'OFF'}")
    
    # 🔥 Configurar debug de ProductReference si se solicita
    if product_ref_debug:
        print("🔍 Modo debug de ProductReference activado")
        logging.getLogger('src.core.data.product_reference').setLevel(logging.DEBUG)
    
    # 🔥 Manejo correcto del argumento ml_enabled
    if ml_enabled is not None:
        # Si se especificó explícitamente, usar ese valor
        print(f"🔥 ML especificado explícitamente: {'✅ HABILITADO' if ml_enabled else '❌ DESHABILITADO'}")
        settings.ML_ENABLED = ml_enabled
        if not ml_enabled:
            settings.NLP_ENABLED = False
    
    try:
        # Cargar productos
        from src.core.data.loader import DataLoader
        
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
        
        # 🔥 MANTENER: Distribución de categorías
        print("\n📊 DISTRIBUCIÓN DE CATEGORÍAS (primeros 50 productos):")
        categories = {}
        for p in products[:50]:
            cat = getattr(p, 'main_category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   • {cat}: {count} productos")
        
        # Inicializar RAG (simplificado)
        rag_agent = None
        
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
            print(f"❌ RAG no disponible: {e}")
            return
        
        # Loop interactivo
        print("\n💡 Escribe 'exit' para salir, 'stats' para estadísticas")
        print("="*50)
        
        while True:
            try:
                query = input("\n🔍 Tu consulta: ").strip()
                
                if query.lower() == 'exit':
                    print("👋 ¡Hasta luego!")
                    break
                
                # 🔥 MANTENER: Comando stats
                if query.lower() == 'stats':
                    print(f"\n📊 ESTADÍSTICAS:")
                    print(f"   • Productos totales: {len(products)}")
                    print(f"   • ML habilitado: {settings.ML_ENABLED}")
                    print(f"   • LLM habilitado: {settings.LOCAL_LLM_ENABLED}")
                    continue
                
                if not query:
                    continue
                
                print(f"\n🔍 Buscando: '{query}'...")
                
                # Procesar consulta
                products_result = rag_agent.search(query, top_k=5)
                answer = f"Encontré {len(products_result)} productos"
                
                # Mostrar resultados
                print(f"\n🤖 {answer}")
                
                if products_result:
                    print(f"\n📦 Resultados:")
                    for i, product in enumerate(products_result[:5], 1):
                        title = getattr(product, 'title', str(product)[:50])
                        price = getattr(product, 'price', 0.0)
                        category = getattr(product, 'main_category', 'General')
                        
                        print(f"  {i}. {title[:60]}")
                        if price:
                            print(f"     💰 ${price:.2f}")
                        print(f"     🏷️ {category}")
                
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
                product_ref_debug=args.product_ref_debug
            )
            
        elif args.command == "ml":
            if args.subcommand == "repair":
                print("⚠️  Reparación de embeddings temporalmente deshabilitada")
                print("💡 Use la versión completa para esta funcionalidad")
            else:
                run_ml_stats()
            
        elif args.command == "train":
            run_train(args)
            
        elif args.command == "test":
            run_test_command(args)
            
        elif args.command == "verify":
            print("⚠️  Verificación del sistema temporalmente deshabilitada")
            print("💡 Use la versión completa para esta funcionalidad")
            
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