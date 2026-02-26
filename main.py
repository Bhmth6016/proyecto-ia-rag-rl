# main.py
"""
main.py — Sistema Híbrido RAG + NER + RLHF

Comandos:
    init         — Inicializar sistema (primera vez)
    interactivo  — Comparación A/B para recolectar preferencias RLHF
    rlhf         — Subcomandos RLHF (train-reward, ppo, rank, stats)
    experimento  — Evaluar 4 métodos de ranking
    stats        — Ver estadísticas del sistema
"""
import json
import sys
from pathlib import Path


def print_help():
    print("\n" + "="*80)
    print("  SISTEMA HÍBRIDO RAG + NER + RLHF")
    print("="*80)
    print()
    print("  COMANDOS:")
    print("    python main.py init                  — Inicializar sistema")
    print("    python main.py interactivo           — Comparación A/B (recolectar prefs)")
    print("    python main.py rlhf --preferences    — Solo recolectar preferencias")
    print("    python main.py rlhf --train-reward   — Entrenar Reward Model")
    print("    python main.py rlhf --ppo            — Ciclo PPO")
    print("    python main.py rlhf --rank 'query'   — Rankear con política entrenada")
    print("    python main.py rlhf --stats          — Estado del pipeline RLHF")
    print("    python main.py experimento           — Evaluar 4 métodos")
    print("    python main.py stats                 — Estadísticas del sistema")
    print()
    print("  FLUJO RLHF CORRECTO (nivel paper):")
    print("    1. python main.py init")
    print("    2. python main.py interactivo        # 30+ comparaciones A/B")
    print("    3. python main.py rlhf --train-reward")
    print("    4. python main.py rlhf --ppo")
    print("    5. python main.py experimento")
    print()
    print("  QUÉ ES RLHF REAL en este sistema:")
    print("    [OK] Comparación A/B explícita (usuario elige ranking)")
    print("    [OK] Reward Model (Bradley-Terry loss)")
    print("    [OK] Policy optimizada con PPO + KL penalty")
    print("    [ERR] Click implícito -> NO es RLHF (era lo anterior, eliminado)")
    print("="*80)


def main():
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].lower()
    extra_args = sys.argv[2:]

    # -- init ----------------------------------------------------------
    if command == "init":
        print("\n  INICIALIZANDO SISTEMA...")
        confirm = input("¿Continuar? (s/n): ").strip().lower()
        if confirm != "s":
            return

        for d in ["data/cache", "data/interactions", "data/preferences",
                  "data/cache/rlhf", "logs", "results"]:
            Path(d).mkdir(parents=True, exist_ok=True)

        try:
            from src.unified_system_v2 import UnifiedSystemV2
            system = UnifiedSystemV2()
            success = system.initialize_with_ner(
                limit=100000, use_cache=True, use_zero_shot=True
            )
            if success:
                system.save_to_cache()
                print(f"\n  Sistema listo: {len(system.canonical_products):,} productos")
                print("  Próximo paso: python main.py interactivo")
            else:
                print("\n  Error inicializando")
        except Exception as e:
            print(f"\n  Error: {e}")
            import traceback; traceback.print_exc()

    # -- interactivo (A/B ranking) --------------------------------------
    elif command == "interactivo":
        try:
            from sistema_interactivo import main as interactivo_main
            interactivo_main()
        except ImportError as e:
            print(f"  Error: {e}")

    # -- rlhf ----------------------------------------------------------
    elif command == "rlhf":
        try:
            from src.rlhf_integration import handle_rlhf_command
            handle_rlhf_command(extra_args)
        except ImportError as e:
            print(f"  Error importando RLHF: {e}")
            import traceback; traceback.print_exc()

    # -- experimento ---------------------------------------------------
    elif command == "experimento":
        print("\n  EJECUTANDO EXPERIMENTO 4 MÉTODOS...")
        confirm = input("¿Ejecutar? (s/n): ").strip().lower()
        if confirm != "s":
            return
        try:
            from experimento_completo_4_metodos import main as exp_main
            exp_main()
        except ImportError as e:
            print(f"  Error: {e}")

    # -- stats ---------------------------------------------------------
    elif command == "stats":
        print("\n  ESTADÍSTICAS DEL SISTEMA")
        try:
            from src.unified_system_v2 import UnifiedSystemV2
            system = UnifiedSystemV2.load_from_cache()
            if not system:
                print("  Sistema no encontrado. Ejecuta: python main.py init")
                return

            stats = system.get_system_stats()
            print(f"\n  Sistema base:")
            print(f"    Productos:    {stats.get('canonical_products', 0):,}")
            print(f"    Vector Store: {'[OK]' if stats.get('has_vector_store') else '[ERR]'}")
            print(f"    NER:          {'[OK]' if stats.get('has_ner_ranker') else '[ERR]'}")
            print(f"    RLHF Policy:  {'[OK]' if stats.get('rlhf_policy_trained') else '[ERR]'}")

            rlhf_dir = Path("data/cache/rlhf")
            if (rlhf_dir / "pipeline_state.json").exists():
                with open(rlhf_dir / "pipeline_state.json") as f:
                    rs = json.load(f)
                print(f"\n  RLHF Pipeline:")
                print(f"    Reward Model: {'[OK] entrenado' if rs.get('reward_model_trained') else '[ERR] sin entrenar'}")
                print(f"    Policy PPO:   {'[OK] entrenado' if rs.get('policy_trained') else '[ERR] sin entrenar'}")
                print(f"    Ciclos PPO:   {rs.get('ppo_cycles_completed', 0)}")

            pref_file = Path("data/preferences/preferences.jsonl")
            if pref_file.exists():
                n = sum(1 for _ in open(pref_file))
                print(f"\n  Preferencias A/B: {n}")
                print(f"    {'[OK] Suficiente para entrenar' if n >= 10 else '[ERR] Necesitas 10+ comparaciones'}")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback; traceback.print_exc()

    elif command in ("help", "--help", "-h"):
        print_help()
    else:
        print(f"\n  Comando no reconocido: {command}")
        print_help()


if __name__ == "__main__":
    main()