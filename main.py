#!/usr/bin/env python3
"""
main.py — Sistema Híbrido RAG + NER + RLHF

Comandos:
    init         — Inicializar sistema (primera vez)
    interactivo  — Recolectar feedback implícito de clicks (original)
    rlhf         — Sesión RLHF real (A-vs-B + Reward Model + PPO)
    experimento  — Evaluar 4 métodos de ranking
    stats        — Ver estadísticas del sistema
"""

import json
import sys
from pathlib import Path


def print_help():
    print("\n" + "=" * 80)
    print("SISTEMA HÍBRIDO RAG + NER + RLHF")
    print("=" * 80)
    print("\n COMANDOS PRINCIPALES:")
    print("  python main.py init                  — Inicializar sistema (primera vez)")
    print("  python main.py interactivo           — Recolectar clicks implícitos")
    print("  python main.py rlhf                  — Sesión RLHF completa (A-vs-B)")
    print("  python main.py rlhf --preferences    — Solo recolectar preferencias A-vs-B")
    print("  python main.py rlhf --train-reward   — Solo entrenar Reward Model")
    print("  python main.py rlhf --ppo            — Solo ciclo PPO")
    print("  python main.py rlhf --rank 'query'   — Rankear con política RLHF")
    print("  python main.py rlhf --stats          — Estado del pipeline RLHF")
    print("  python main.py experimento           — Experimento 4 métodos")
    print("  python main.py stats                 — Estadísticas del sistema")
    print()
    print(" FLUJO RLHF REAL (para paper IEEE):")
    print("  1. python main.py init")
    print("  2. python main.py rlhf --preferences   # Recolectar 30+ comparaciones A-vs-B")
    print("  3. python main.py rlhf --train-reward  # Entrenar Reward Model")
    print("  4. python main.py rlhf --ppo           # Optimizar con PPO")
    print("  5. python main.py experimento          # Evaluar mejora")
    print()
    print(" COMPONENTES RLHF IMPLEMENTADOS:")
    print("  ✓ Policy Model       — Transformer entrenable (cross-attention)")
    print("  ✓ Dataset A-vs-B     — Preferencias humanas explícitas")
    print("  ✓ Reward Model       — Red neuronal con Bradley-Terry loss")
    print("  ✓ PPO Trainer        — Proximal Policy Optimization + KL penalty")
    print("  ✓ Ciclo iterativo    — Collect → Reward → PPO → repetir")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].lower()
    extra_args = sys.argv[2:]

    # ── init ──────────────────────────────────────────────────────────────
    if command == "init":
        print("\n INICIALIZANDO SISTEMA...")
        confirm = input("¿Continuar? (s/n): ").strip().lower()
        if confirm != "s":
            print("Cancelado.")
            return

        for d in ["data/cache", "data/interactions", "data/preferences",
                  "data/cache/rlhf", "logs", "results"]:
            Path(d).mkdir(parents=True, exist_ok=True)

        try:
            from src.unified_system_v2 import UnifiedSystemV2
            system = UnifiedSystemV2()
            success = system.initialize_with_ner(
                limit=200000, use_cache=True, use_zero_shot=True
            )
            if success:
                system.save_to_cache()
                print(f"\n Sistema inicializado: {len(system.canonical_products):,} productos")
                print(" Próximo paso: python main.py rlhf --preferences")
            else:
                print("\n Error inicializando sistema")
        except Exception as e:
            print(f"\n Error: {e}")
            import traceback
            traceback.print_exc()

    # ── interactivo (clicks implícitos, sistema original) ─────────────────
    elif command == "interactivo":
        try:
            from sistema_interactivo import main as interactivo_main
            interactivo_main()
        except ImportError as e:
            print(f" Error: {e}")

    # ── rlhf ──────────────────────────────────────────────────────────────
    elif command == "rlhf":
        try:
            from src.rlhf_integration import handle_rlhf_command
            handle_rlhf_command(extra_args)
        except ImportError as e:
            print(f" Error importando RLHF: {e}")
            import traceback
            traceback.print_exc()

    # ── experimento ───────────────────────────────────────────────────────
    elif command == "experimento":
        print("\n EJECUTANDO EXPERIMENTO 4 MÉTODOS...")
        confirm = input("¿Ejecutar? (s/n): ").strip().lower()
        if confirm != "s":
            print("Cancelado.")
            return
        try:
            from experimento_completo_4_metodos import main as exp_main
            exp_main()
        except ImportError as e:
            print(f" Error: {e}")

    # ── stats ─────────────────────────────────────────────────────────────
    elif command == "stats":
        print("\n ESTADÍSTICAS DEL SISTEMA...")
        try:
            from src.unified_system_v2 import UnifiedSystemV2
            system = UnifiedSystemV2.load_from_cache()

            if not system:
                print(" Sistema no encontrado. Ejecuta: python main.py init")
                return

            stats = system.get_system_stats()
            print(f"\n Sistema base:")
            print(f"   Productos:      {stats.get('canonical_products', 0):,}")
            print(f"   Vector Store:   {'✓' if stats.get('has_vector_store') else '✗'}")
            print(f"   NER:            {'✓' if stats.get('has_ner_ranker') else '✗'}")

            # Stats del pipeline RLHF si existe
            rlhf_dir = Path("data/cache/rlhf")
            if (rlhf_dir / "pipeline_state.json").exists():
                with open(rlhf_dir / "pipeline_state.json") as f:
                    rlhf_state = json.load(f)
                print(f"\n RLHF Pipeline:")
                print(f"   Reward Model:   {'✓ Entrenado' if rlhf_state.get('reward_model_trained') else '✗ No entrenado'}")
                print(f"   Policy (PPO):   {'✓ Entrenado' if rlhf_state.get('policy_trained') else '✗ No entrenado'}")
                print(f"   Ciclos PPO:     {rlhf_state.get('ppo_cycles_completed', 0)}")

            pref_file = Path("data/preferences/preferences.jsonl")
            if pref_file.exists():
                n_prefs = sum(1 for _ in open(pref_file))
                print(f"\n Preferencias A-vs-B: {n_prefs}")
                print(f"   {'✓ Suficiente' if n_prefs >= 10 else '✗ Necesita 10+ preferencias'}")

            int_file = Path("data/interactions/real_interactions.jsonl")
            if int_file.exists():
                n_int = sum(1 for _ in open(int_file))
                print(f" Clicks implícitos:   {n_int}")

        except Exception as e:
            print(f" Error: {e}")
            import traceback
            traceback.print_exc()

    # ── help ──────────────────────────────────────────────────────────────
    elif command in ("help", "--help", "-h"):
        print_help()

    else:
        print(f"\n Comando no reconocido: {command}")
        print_help()


if __name__ == "__main__":
    main()