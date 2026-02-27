# -*- coding: utf-8 -*-
"""
rlhf_integration.py
Conecta RLHFPipeline con UnifiedSystemV2.

Comandos disponibles:
    python main.py rlhf --stats
    python main.py rlhf --ppo
    python main.py rlhf --rank "query texto"
"""
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Función principal de integración
# ---------------------------------------------------------------------------

def add_rlhf_to_system(system):
    """
    Conecta RLHFPipeline con UnifiedSystemV2.

    Extrae del sistema:
        embedding_model  ← system.canonicalizer.embedding_model
        product_index    ← {product.id: embedding_vector}  (cacheado en disco)
        vector_store     ← system.vector_store
    """
    try:
        from src.rlhf.rlhf_pipeline import RLHFPipeline
    except ImportError:
        from rlhf.rlhf_pipeline import RLHFPipeline

    embedding_model = _get_embedding_model(system)
    if embedding_model is None:
        raise RuntimeError(
            "No se encontró embedding_model en el sistema. "
            "Verifica que system.canonicalizer.embedding_model existe."
        )

    product_index = _load_or_build_product_index(system, embedding_model)
    print(f"  Índice listo: {len(product_index):,} productos con embedding")

    vector_store = _get_vector_store(system)
    if vector_store is None:
        raise RuntimeError(
            "No se encontró vector_store en el sistema. "
            "Verifica que system.vector_store existe."
        )

    pipeline = RLHFPipeline(
        embedding_model=embedding_model,
        product_index=product_index,
        vector_store=vector_store,
        emb_dim=384,
        top_k_ranking=10,
    )
    pipeline.initialize(load_checkpoint=True)

    system.rlhf_pipeline = pipeline

    status  = pipeline.get_status()
    trained = "entrenada" if status.get('policy_trained') else "sin entrenar"
    reward  = "[OK]" if status.get('reward_trained') else "[--]"
    print(f"  RLHFPipeline listo | reward={reward} | policy={trained}")
    print(f"  Preferencias: {status.get('n_preferences', 0)}")
    return pipeline


# ---------------------------------------------------------------------------
# Función para experimento_completo_4_metodos.py
# ---------------------------------------------------------------------------

def rlhf_method_for_experiment(system, query_text: str, k: int) -> list:
    """Método RLHF para usar en el experimento de 4 métodos."""
    if not hasattr(system, "rlhf_pipeline"):
        logger.debug("RLHF no disponible, usando baseline")
        return system._process_query_baseline(query_text, k)

    pipeline = system.rlhf_pipeline
    if not pipeline.policy_trained:
        logger.debug("Policy no entrenada, usando baseline")
        return system._process_query_baseline(query_text, k)

    try:
        products, query_emb, _ = pipeline.retrieve_candidates(query_text, k=k * 2)
        if not products:
            return []
        ranked = pipeline.rank_products(query_text, products, query_emb)
        return ranked[:k]
    except Exception as e:
        logger.warning(f"Error en RLHF method: {e}")
        return system._process_query_baseline(query_text, k)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def handle_rlhf_command(args: List[str]):
    """
    Maneja: python main.py rlhf [subcomando]

    Subcomandos:
        --stats          estado del pipeline
        --ppo            ciclo PPO
        --rank "query"   rankear con política entrenada
    """
    print("\n" + "=" * 70)
    print("  RLHF — PointwiseReward + PPO")
    print("=" * 70)

    system = _load_system()
    if not system:
        return

    try:
        pipeline = add_rlhf_to_system(system)
    except Exception as e:
        print(f"\n  Error inicializando RLHF: {e}")
        import traceback; traceback.print_exc()
        return

    sub = args[0] if args else "--stats"

    if sub == "--stats":
        _show_stats(pipeline)

    elif sub == "--ppo":
        n_queries = int(args[1]) if len(args) > 1 and args[1].isdigit() else 50
        epochs    = int(args[2]) if len(args) > 2 and args[2].isdigit() else 5
        print(f"\n  Ciclo PPO ({n_queries} queries, {epochs} épocas)...")
        if not pipeline.reward_trained:
            print("  [ERR] Reward model no entrenado.")
            print("    Ejecuta primero: python train_pointwise_reward.py")
            return
        result = pipeline.run_ppo_cycle(
            n_queries=n_queries,
            epochs=epochs,
        )
        if "error" in result:
            print(f"\n  [ERR] {result['error']}")
        else:
            print(f"\n  [OK] PPO completado")
            print(f"    Reward final:       {result.get('final_reward', 0):.4f}")
            print(f"    KL promedio:        {result.get('avg_kl', 0):.4f}")
            updated = '[OK]' if result.get('policy_updated') else '[ERR] revisar KL/reward'
            print(f"    Policy actualizada: {updated}")
            versions = pipeline.ppo_trainer.list_versions()
            if versions:
                print(f"    Versiones:          {[v.name for v in versions]}")

    elif sub == "--rank":
        query_text = " ".join(args[1:]) if len(args) > 1 else "laptop computer"
        print(f"\n  Rankeando: '{query_text}'")
        if not pipeline.policy_trained:
            print("  [WARN] Policy no entrenada — mostrando baseline FAISS")
        products, emb, _ = pipeline.retrieve_candidates(query_text, k=20)
        if not products:
            print("  Sin resultados")
            return
        ranked = pipeline.rank_products(query_text, products, emb)
        tag = "(RLHF policy)" if pipeline.policy_trained else "(baseline FAISS)"
        print(f"\n  Top 10 {tag}:")
        for i, p in enumerate(ranked[:10], 1):
            title  = getattr(p, "title", "")[:65]
            rating = getattr(p, "rating", None)
            r_str  = f"*{float(rating):.1f}" if rating else "    "
            print(f"  {i:2d}. {r_str}  {title}")

    else:
        print(f"\n  Subcomando desconocido: '{sub}'")
        print("  Disponibles: --stats | --ppo | --rank 'query'")

    print()


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _get_embedding_model(system):
    """Extrae el modelo de embeddings del sistema."""
    # Opción 1: canonicalizer
    if hasattr(system, 'canonicalizer') and system.canonicalizer is not None:
        emb = getattr(system.canonicalizer, 'embedding_model', None)
        if emb is not None:
            return emb
    # Opción 2: directo en sistema
    if hasattr(system, 'embedding_model') and system.embedding_model is not None:
        return system.embedding_model
    # Opción 3: vector_store
    if hasattr(system, 'vector_store') and system.vector_store is not None:
        emb = getattr(system.vector_store, 'embedding_model', None)
        if emb is not None:
            return emb
    # Fallback: cargar MiniLM directamente
    logger.warning("No se encontró embedding_model. Cargando all-MiniLM-L6-v2...")
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        logger.error(f"No se pudo cargar sentence-transformer: {e}")
        return None


PRODUCT_INDEX_CACHE = Path("data/cache/product_embeddings.npz")


def _load_or_build_product_index(system, embedding_model) -> Dict[str, np.ndarray]:
    """
    Carga el índice product_id → embedding desde cache (rápido)
    o lo construye y lo guarda (lento, primera vez).
    """
    if PRODUCT_INDEX_CACHE.exists():
        try:
            print("  Cargando índice de embeddings desde cache...")
            data  = np.load(PRODUCT_INDEX_CACHE, allow_pickle=True)
            index = {pid: emb for pid, emb in zip(data['ids'], data['embeddings'])}
            print(f"  [OK] Cache: {len(index):,} productos")
            return index
        except Exception as e:
            logger.warning(f"  Cache corrupta, reconstruyendo: {e}")

    # Intentar extraer desde FAISS (sin recomputar embeddings)
    print("  Intentando extraer embeddings del índice FAISS...")
    index = _extract_from_faiss(system)
    if index:
        print(f"  [OK] Extraídos del FAISS: {len(index):,} productos")
        _save_index_cache(index)
        return index

    # Última opción: computar desde títulos (lento)
    print("  Computando embeddings desde títulos...")
    print("  (Solo ocurre la primera vez — se guardará en cache)")
    index = _build_product_index(system, embedding_model)
    if index:
        _save_index_cache(index)
    return index


def _build_product_index(system, embedding_model) -> Dict[str, np.ndarray]:
    """Computa embeddings desde títulos de productos."""
    products = getattr(system, 'canonical_products', [])
    if not products:
        return {}

    pids, titles = [], []
    for p in products:
        pid   = getattr(p, 'id', None) or getattr(p, 'product_id', None)
        title = getattr(p, 'title', '') or ''
        if pid and title:
            pids.append(pid)
            titles.append(title)

    if not titles:
        return {}

    batch_size = 512
    all_embs   = []
    total      = len(titles)
    for i in range(0, total, batch_size):
        batch = titles[i:i + batch_size]
        embs  = embedding_model.encode(
            batch, normalize_embeddings=True, show_progress_bar=False
        )
        all_embs.append(embs)
        if i % (batch_size * 10) == 0:
            print(f"    {i+len(batch):,}/{total:,} ({(i+len(batch))/total*100:.0f}%)")

    all_embs_np = np.vstack(all_embs)
    return {pid: emb for pid, emb in zip(pids, all_embs_np)}


def _extract_from_faiss(system) -> Dict[str, np.ndarray]:
    """Extrae vectores directamente del índice FAISS si es posible."""
    try:
        import faiss
        vs = getattr(system, 'vector_store', None)
        if vs is None:
            return {}

        id_to_idx = (
            getattr(vs, 'id_to_index', None)
            or getattr(vs, '_id_map', None)
            or getattr(vs, 'docstore', {})
        )
        faiss_index = getattr(vs, 'index', None) or getattr(vs, '_index', None)

        if not id_to_idx or faiss_index is None:
            return {}

        n = faiss_index.ntotal
        if n == 0:
            return {}

        all_vectors = np.zeros((n, faiss_index.d), dtype=np.float32)
        faiss_index.reconstruct_n(0, n, all_vectors)

        index = {}
        for pid, idx in id_to_idx.items():
            if isinstance(idx, int) and 0 <= idx < n:
                index[pid] = all_vectors[idx]
        return index

    except Exception as e:
        logger.debug(f"  No se pudo extraer de FAISS: {e}")
        return {}


def _save_index_cache(index: Dict[str, np.ndarray]):
    """Guarda el índice en disco como .npz comprimido."""
    try:
        PRODUCT_INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ids  = np.array(list(index.keys()))
        embs = np.stack(list(index.values()))
        np.savez_compressed(PRODUCT_INDEX_CACHE, ids=ids, embeddings=embs)
        size_mb = PRODUCT_INDEX_CACHE.stat().st_size / 1e6
        print(f"  [OK] Cache guardada: {PRODUCT_INDEX_CACHE} ({size_mb:.1f} MB)")
    except Exception as e:
        logger.warning(f"  No se pudo guardar cache: {e}")


def _get_vector_store(system):
    """Extrae el vector store del sistema."""
    return (
        getattr(system, 'vector_store', None)
        or getattr(system, 'faiss_store', None)
        or getattr(system, '_vector_store', None)
    )


def _load_system():
    """Carga UnifiedSystemV2 desde cache."""
    try:
        from src.unified_system_v2 import UnifiedSystemV2
        system = UnifiedSystemV2.load_from_cache()
        if not system:
            print("  Sistema no encontrado. Ejecuta: python main.py init")
            return None
        print(f"  Sistema cargado: {len(system.canonical_products):,} productos")
        return system
    except Exception as e:
        print(f"  Error: {e}")
        import traceback; traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Mostrar estado
# ---------------------------------------------------------------------------

def _show_stats(pipeline):
    status = pipeline.get_status()
    print("\n  ESTADO DEL PIPELINE RLHF:")
    print(f"    Reward model entrenado:  {'[OK]' if status['reward_trained'] else '[--]'}")
    print(f"    Policy entrenada (PPO):  {'[OK]' if status['policy_trained'] else '[--]'}")
    print(f"    Device:                  {status['device']}")

    cs = status.get('collector_stats', {})
    print(f"\n  PREFERENCIAS A/B:")
    print(f"    Total:           {cs.get('total', 0)}")
    print(f"    Prefirió A:      {cs.get('prefer_A', 0)}")
    print(f"    Prefirió B:      {cs.get('prefer_B', 0)}")
    print(f"    Empates:         {cs.get('equal', 0)}")
    print(f"    Queries únicas:  {cs.get('unique_queries', 0)}")

    ckpts = status.get('checkpoints', {})
    print(f"\n  CHECKPOINTS:")
    print(f"    reward_model.pt: {'[OK]' if ckpts.get('reward') else '[--]'}")
    print(f"    policy_model.pt: {'[OK]' if ckpts.get('policy') else '[--]'}")
    versions = ckpts.get('versions', [])
    if versions:
        print(f"    Versiones:       {versions}")

    ppo = status.get('ppo_status', {})
    print(f"\n  PPO STATUS:")
    print(f"    Beta actual:     {ppo.get('current_beta', 'N/A')}")
    print(f"    KL promedio:     {ppo.get('recent_avg_kl', 0):.4f}")
    print(f"    Target KL:       {ppo.get('target_kl', 'N/A')}")
    print(f"    KL status:       {ppo.get('status', 'N/A')}")

    n = cs.get('total', 0)
    print(f"\n  PRÓXIMO PASO:")
    if n < 10:
        print(f"    → python main.py interactivo   (tienes {n}, necesitas 10+)")
    elif not status['reward_trained']:
        print(f"    → python train_pointwise_reward.py")
    elif not status['policy_trained']:
        print(f"    → python main.py rlhf --ppo")
    else:
        print(f"    → python evaluate_methods.py")