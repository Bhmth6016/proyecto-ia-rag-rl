"""
Integración del RLHF Pipeline con el sistema existente.

Este módulo extiende UnifiedSystemV2 con el pipeline RLHF real,
reemplazando el ranker lineal anterior por los 5 componentes correctos.

Para usar desde main.py:
    from src.rlhf_integration import add_rlhf_to_system
    add_rlhf_to_system(system)
    system.run_rlhf_session(queries)

O directamente desde la línea de comandos:
    python main.py rlhf                 # sesión interactiva completa
    python main.py rlhf --preferences   # solo recolectar preferencias
    python main.py rlhf --train-reward  # solo entrenar reward model
    python main.py rlhf --ppo           # solo ciclo PPO
    python main.py rlhf --rank "query"  # rankear con política entrenada
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Función de integración principal
# ─────────────────────────────────────────────────────────────────────────────

def add_rlhf_to_system(system, device: str = "cpu"):
    """
    Agrega el pipeline RLHF real a un sistema existente (UnifiedSystemV2).

    Inyecta:
        system.rlhf_pipeline      — el pipeline completo
        system.rank_with_rlhf()   — método de ranking con política entrenada
        system.run_rlhf_session() — método para correr sesión completa
    """
    try:
        from src.rlhf.rlhf_pipeline import RLHFPipeline
    except ImportError:
        from rlhf.rlhf_pipeline import RLHFPipeline

    pipeline = RLHFPipeline(
        base_system=system,
        embedding_dim=system.config.get("embedding", {}).get("dimension", 384),
        device=device,
    )
    pipeline.initialize(load_checkpoint=True)

    system.rlhf_pipeline = pipeline

    # Inyectar método de ranking
    def rank_with_rlhf(self, query, products, query_emb_np=None):
        return self.rlhf_pipeline.rank_products(query, products, query_emb_np)

    # Inyectar método de sesión
    def run_rlhf_session(self, queries, n_cycles=3, preferences_per_cycle=10):
        return self.rlhf_pipeline.run_full_rlhf_session(
            queries, n_cycles=n_cycles, preferences_per_cycle=preferences_per_cycle
        )

    import types
    system.rank_with_rlhf = types.MethodType(rank_with_rlhf, system)
    system.run_rlhf_session = types.MethodType(run_rlhf_session, system)

    logger.info("RLHF Pipeline integrado con el sistema")
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Integración con experimento_completo_4_metodos.py
# ─────────────────────────────────────────────────────────────────────────────

def rlhf_method_for_experiment(system, query_text: str, k: int) -> list:
    """
    Función drop-in para reemplazar _method_rlhf() en UnifiedSystemV2.
    Usa el PolicyModel + PPO en lugar del ranker lineal.
    """
    if not hasattr(system, "rlhf_pipeline"):
        return system._process_query_baseline(query_text, k)

    pipeline = system.rlhf_pipeline

    if not pipeline.policy_trained:
        logger.debug("RLHF policy no entrenada, usando baseline")
        return system._process_query_baseline(query_text, k)

    products, query_emb_np, _ = pipeline.retrieve_candidates(query_text, k=k * 2)
    if not products:
        return []

    ranked = pipeline.rank_products(query_text, products, query_emb_np)
    return ranked[:k]


# ─────────────────────────────────────────────────────────────────────────────
# CLI — comandos RLHF para main.py
# ─────────────────────────────────────────────────────────────────────────────

def handle_rlhf_command(args: List[str]):
    """
    Maneja el comando `python main.py rlhf [subcomando]`.

    Subcomandos:
        (sin subcomando)    — sesión interactiva completa
        --preferences       — solo recolectar preferencias A-vs-B
        --train-reward      — solo entrenar el reward model
        --ppo               — solo ejecutar ciclo PPO
        --rank "query"      — rankear una query con la política entrenada
        --stats             — mostrar estado del pipeline
    """
    print("\n" + "═" * 70)
    print("  RLHF REAL — PolicyModel + RewardModel + PPO")
    print("═" * 70)

    # Cargar sistema
    system = _load_system()
    if not system:
        return

    # Añadir pipeline RLHF
    try:
        pipeline = add_rlhf_to_system(system)
    except Exception as e:
        print(f"\n Error inicializando RLHF: {e}")
        import traceback
        traceback.print_exc()
        return

    # Cargar queries del ground truth
    queries = _load_queries()

    # Parsear subcomando
    sub = args[0] if args else ""

    if sub == "--stats":
        _show_rlhf_stats(pipeline)

    elif sub == "--preferences":
        n = int(args[1]) if len(args) > 1 else 10
        print(f"\nRecolectando preferencias para {n} queries...")
        sample = queries[:n] if len(queries) >= n else queries
        pipeline.run_preference_collection_session(sample)

    elif sub == "--train-reward":
        print("\nEntrenando Reward Model...")
        result = pipeline.train_reward_model(epochs=30, lr=1e-4)
        if "error" in result:
            print(f" Error: {result['error']}")
        else:
            print(f" Accuracy final: {result.get('best_accuracy', 0):.4f}")

    elif sub == "--ppo":
        n = int(args[1]) if len(args) > 1 else 20
        print(f"\nEjecutando ciclo PPO con {n} queries...")
        result = pipeline.run_ppo_training_cycle(queries[:n])
        if "error" in result:
            print(f" Error: {result['error']}")
        else:
            print(f" Reward medio: {result.get('mean_reward', 0):.4f}")
            print(f" Ciclos PPO completados: {pipeline.ppo_cycles_completed}")

    elif sub == "--rank":
        query_text = args[1] if len(args) > 1 else "laptop computer"
        print(f"\nRankeando con política RLHF: '{query_text}'")
        products, emb, _ = pipeline.retrieve_candidates(query_text, k=10)
        if not products:
            print(" Sin resultados")
            return
        ranked = pipeline.rank_products(query_text, products, emb)
        print(f"\nTop 10 resultados:")
        for i, p in enumerate(ranked[:10], 1):
            title = getattr(p, "title", "")[:70]
            pid = getattr(p, "id", "")
            rating = getattr(p, "rating", None)
            r_str = f"⭐{rating:.1f}" if rating else "     "
            print(f"  {i:2d}. [{pid}] {r_str} {title}")

    else:
        # Sesión completa por defecto
        n_cycles = int(args[0]) if args and args[0].isdigit() else 3

        print(f"\nIniciando sesión RLHF completa ({n_cycles} ciclos)...")
        print(f"Queries disponibles: {len(queries)}")

        confirm = input("\n¿Continuar? (s/n): ").strip().lower()
        if confirm != "s":
            print("Cancelado.")
            return

        pipeline.run_full_rlhf_session(
            queries,
            n_cycles=n_cycles,
            preferences_per_cycle=10,
        )

    # Mostrar stats finales
    _show_rlhf_stats(pipeline)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers privados
# ─────────────────────────────────────────────────────────────────────────────

def _load_system():
    """Carga UnifiedSystemV2 desde caché."""
    try:
        cache_path = Path("data/cache/unified_system_v2.pkl")
        if not cache_path.exists():
            print(" Sistema no encontrado. Ejecuta primero:")
            print("   python main.py init")
            return None

        from src.unified_system_v2 import UnifiedSystemV2
        system = UnifiedSystemV2.load_from_cache()

        if not system:
            print(" Error cargando sistema")
            return None

        print(f" Sistema cargado: {len(system.canonical_products):,} productos")
        return system

    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def _load_queries() -> List[str]:
    """Carga queries del ground truth."""
    gt_file = Path("data/interactions/ground_truth_REAL.json")

    if not gt_file.exists():
        # Fallback: queries de ejemplo
        return [
            "laptop computer", "bluetooth headphones", "running shoes",
            "coffee maker", "phone case", "gaming mouse", "yoga mat",
            "water bottle", "desk lamp", "book shelf"
        ]

    try:
        with open(gt_file) as f:
            gt = json.load(f)
        return list(gt.keys())
    except Exception:
        return []


def _show_rlhf_stats(pipeline):
    """Muestra estadísticas del pipeline RLHF."""
    stats = pipeline.get_stats()

    print("\n ESTADO RLHF PIPELINE:")
    print(f"   Reward Model entrenado:  {'✓' if stats.get('reward_model_trained') else '✗'}")
    print(f"   Policy entrenada (PPO):  {'✓' if stats.get('policy_trained') else '✗'}")
    print(f"   Ciclos PPO completados:  {stats.get('ppo_cycles_completed', 0)}")

    if "preferences" in stats:
        p = stats["preferences"]
        print(f"   Preferencias A-vs-B:    {p.get('total', 0)}")
        print(f"     → Prefirió A:         {p.get('prefer_a', 0)}")
        print(f"     → Prefirió B:         {p.get('prefer_b', 0)}")
        print(f"     → Empates:            {p.get('equal', 0)}")
        ready = "✓ Listo para entrenar" if p.get("ready_for_training") else "✗ Necesita 10+ prefs"
        print(f"   Estado:                 {ready}")

    if "ppo" in stats:
        pp = stats["ppo"]
        print(f"   Último reward PPO:      {pp.get('last_reward', 0):.4f}")
        print(f"   Última KL divergence:   {pp.get('last_kl', 0):.5f}")

    if "policy_parameters" in stats:
        print(f"   Parámetros del policy:  {stats['policy_parameters']:,}")