# main.py
import sys
import json
from pathlib import Path

def print_help():
    print("\n" + "="*80)
    print("SISTEMA HÍBRIDO RAG + NER + RLHF - COMANDOS DISPONIBLES")
    print("="*80)
    print("\n COMANDOS PRINCIPALES:")
    print("  python main.py init           - Inicializar sistema (primera vez)")
    print("  python main.py interactivo    - Recolectar feedback REAL")
    print("  python main.py experimento    - Ejecutar experimento completo")
    print("  python main.py stats          - Ver estadísticas del sistema")
    print("\n FLUJO RECOMENDADO PARA PAPER IEEE:")
    print("  1. python main.py init                # Inicializar sistema")
    print("  2. python main.py interactivo         # Recolectar 30+ clicks REALES")
    print("  3. python main.py experimento         # Ejecutar experimento 4 métodos")
    print("\n DIRECTORIOS IMPORTANTES:")
    print("  • data/interactions/          - Feedback REAL recolectado")
    print("  • results/                    - Resultados de experimentos")
    print("  • logs/                       - Logs detallados")
    print("  • data/cache/                 - Cache para velocidad")
    print("="*80)

def main():
    if len(sys.argv) < 2:
        print_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "init":
        print("\n INICIALIZANDO SISTEMA POR PRIMERA VEZ...")
        print("   Esto creará embeddings para todos los productos (90K)")
        print("   Puede tomar 30-60 minutos...")
        
        confirm = input("\n¿Continuar? (s/n): ").strip().lower()
        if confirm != 's':
            print("Cancelado.")
            return
        
        for dir_name in ['data/cache', 'data/interactions', 'logs', 'results']:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
        
        try:
            from src.unified_system_v2 import UnifiedSystemV2
            system = UnifiedSystemV2()
            
            print("\n🚀 Inicializando con NER...")
            success = system.initialize_with_ner(
                limit=200000,  # Todos los productos
                use_cache=True,
                use_zero_shot=True
            )
            
            if success:
                print("\n SISTEMA INICIALIZADO EXITOSAMENTE")
                print("   • Productos: {len(system.canonical_products):,}")
                print("   • Métodos: Baseline, NER-Enhanced, RLHF, Full-Hybrid")
                print("   • Guardado en: data/cache/unified_system_v2.pkl")
                
                system.save_to_cache()
                
                print("\n PRÓXIMO PASO:")
                print("   python main.py interactivo   # Para recolectar feedback REAL")
            else:
                print("\n Error inicializando sistema")
                
        except Exception as e:
            print(f"\n Error: {e}")
            import traceback
            traceback.print_exc()
    
    elif command == "interactivo":
        print("\n INICIANDO SISTEMA INTERACTIVO REAL...")
        print("   Objetivo: Obtener 30+ clicks REALES para entrenar RLHF")
        
        try:
            from sistema_interactivo import main as interactivo_main
            interactivo_main()
        except ImportError as e:
            print(f" Error: {e}")
            print("   Asegúrate de que sistema_interactivo.py existe")
    
    elif command == "experimento":
        print("\n EJECUTANDO EXPERIMENTO COMPLETO...")
        print("   Evaluará 4 métodos de ranking:")
        print("   1. Baseline (FAISS)")
        print("   2. NER-Enhanced")
        print("   3. RLHF")
        print("   4. Full Hybrid")
        
        confirm = input("\n¿Ejecutar experimento? (s/n): ").strip().lower()
        if confirm != 's':
            print("Cancelado.")
            return
        
        try:
            from experimento_completo_4_metodos import main as experimento_main
            experimento_main()
        except ImportError as e:
            print(f" Error: {e}")
            print("   Asegúrate de que experimento_completo_4_metodos.py existe")
    
    elif command == "stats":
        print("\n ESTADÍSTICAS DEL SISTEMA...")
        
        try:
            from src.unified_system_v2 import UnifiedSystemV2
            
            system = UnifiedSystemV2.load_from_cache()
            
            if not system:
                print(" Sistema no encontrado. Ejecuta primero:")
                print("   python main.py init")
                return
            
            stats = system.get_system_stats()
            
            print("\n ESTADÍSTICAS PRINCIPALES:")
            print(f"   • Productos canonizados: {stats.get('canonical_products', 0):,}")
            print(f"   • Vector Store: {' Disponible' if stats.get('has_vector_store', False) else '❌ No disponible'}")
            print(f"   • NER Enhanced: {' Disponible' if stats.get('has_ner_ranker', False) else '❌ No disponible'}")
            
            # RLHF status
            if 'rl_stats' in stats:
                rl_stats = stats['rl_stats']
                rl_status = ' Entrenado' if rl_stats.get('has_learned', False) else '⚠️ No entrenado'
                print(f"   • RLHF: {rl_status} ({rl_stats.get('feedback_count', 0)} feedback)")
            else:
                print("   • RLHF:  No inicializado")
            
            interactions_file = Path("data/interactions/real_interactions.jsonl")
            if interactions_file.exists():
                try:
                    with open(interactions_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    clicks = sum(1 for line in lines if '"interaction_type": "click"' in line)
                    queries = sum(1 for line in lines if '"interaction_type": "query"' in line)
                    print(f"   • Feedback REAL: {len(lines)} interacciones ({clicks} clicks, {queries} queries)")
                except (OSError, UnicodeDecodeError):
                    print("   • Feedback REAL: Archivo existe")
            else:
                print("   • Feedback REAL:  No hay interacciones")
            
            gt_file = Path("data/interactions/ground_truth_REAL.json")
            if gt_file.exists():
                try:
                    with open(gt_file, 'r') as f:
                        gt = json.load(f)
                    total_relevant = sum(len(ids) for ids in gt.values())
                    print(f"   • Ground Truth: {len(gt)} queries, {total_relevant} productos relevantes")
                except (json.JSONDecodeError, OSError, TypeError):
                    print("   • Ground Truth: Archivo existe")
            
            print("\n ESTADO PARA EXPERIMENTO:")
            
            has_feedback = interactions_file.exists()
            has_ground_truth = gt_file.exists()
            
            if has_feedback and has_ground_truth:
                try:
                    with open(interactions_file, 'r') as f:
                        line_count = sum(1 for _ in f)
                    if line_count >= 10:
                        print("    Listo para experimento (suficiente feedback)")
                    else:
                        print("     Poco feedback. Recomendado: 30+ interacciones")
                        print("      python main.py interactivo")
                except OSError:
                    print("     Error leyendo feedback")
            else:
                print("    No hay datos suficientes. Ejecuta:")
                print("      python main.py interactivo")
                
        except Exception as e:
            print(f" Error obteniendo estadísticas: {e}")
            import traceback
            traceback.print_exc()
    
    elif command == "help" or command == "--help" or command == "-h":
        print_help()
    
    else:
        print(f"\n Comando no reconocido: {command}")
        print_help()

if __name__ == "__main__":
    main()