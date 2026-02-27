# src/rlhf_integration.py
# -*- coding: utf-8 -*-
"""
Integracion del RLHF Pipeline con UnifiedSystemV2.

Comandos disponibles:
    python main.py rlhf --stats
    python main.py rlhf --train-reward
    python main.py rlhf --ppo
    python main.py rlhf --rank "query texto"
    python main.py rlhf --validate
"""

import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Función principal de integración - DEBE ESTAR DEFINIDA AL INICIO
# ---------------------------------------------------------------------

def add_rlhf_to_system(system):
    """
    Conecta el nuevo RLHFPipeline con UnifiedSystemV2.

    Extrae del sistema:
        embedding_model  <- system.canonicalizer.embedding_model
        product_index    <- {product.id: embedding_vector}  (cacheado en disco)
        vector_store     <- system.vector_store

    El product_index se guarda en data/cache/product_embeddings.npz
    para evitar recomputar 89k embeddings en cada ejecución (~15 min -> <1 seg).
    """
    try:
        # Intentar importar desde la estructura de paquetes
        try:
            from src.rlhf.rlhf_pipeline import RLHFPipeline
        except ImportError:
            from rlhf.rlhf_pipeline import RLHFPipeline
            
        # Importar PointwiseRewardModel para asegurar que está disponible
        try:
            from src.rlhf.pointwise_reward_model import PointwiseRewardModel
        except ImportError:
            from src.rlhf.pointwise_reward_model import PointwiseRewardModel
    except ImportError as e:
        logger.error(f"Error importando módulos RLHF: {e}")
        raise

    # 1. Obtener el modelo de embeddings
    embedding_model = _get_embedding_model(system)
    if embedding_model is None:
        raise RuntimeError(
            "No se encontró embedding_model en el sistema. "
            "Verifica que system.canonicalizer.embedding_model existe."
        )

    # 2. Construir (o cargar desde cache) el indice product_id -> embedding
    product_index = _load_or_build_product_index(system, embedding_model)
    print(f"  Indice listo: {len(product_index):,} productos con embedding")

    # 3. Obtener el vector store
    vector_store = _get_vector_store(system)
    if vector_store is None:
        raise RuntimeError(
            "No se encontró vector_store en el sistema. "
            "Verifica que system.vector_store existe."
        )

    # 4. Construir el pipeline
    pipeline = RLHFPipeline(
        embedding_model=embedding_model,
        product_index=product_index,
        vector_store=vector_store,
        emb_dim=384,
        top_k_ranking=10,
        reward_mode="pointwise",  # Fase 1 — cambiar a "listwise" para Fase 2
    )
    pipeline.initialize(load_checkpoint=True)

    # Intentar cargar modelo pointwise si existe
    _try_load_pointwise(pipeline)

    # 5. Inyectar en el sistema
    system.rlhf_pipeline = pipeline

    status = pipeline.get_status()
    trained = "entrenada" if status.get('policy_trained') else "sin entrenar"
    reward_ok = "[OK]" if status.get('reward_trained') else "[ERR]"
    print(f"  RLHFPipeline listo | reward={reward_ok} | policy={trained}")
    print(f"  Preferencias: {status.get('n_preferences', 0)}")

    return pipeline


# ---------------------------------------------------------------------
# Función para experimento_completo_4_metodos.py
# ---------------------------------------------------------------------

def rlhf_method_for_experiment(system, query_text: str, k: int) -> list:
    """
    Método RLHF para usar en el experimento de 4 métodos.
    Usa la PolicyModel entrenada con PPO.
    """
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


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def handle_rlhf_command(args: List[str]):
    """
    Maneja `python main.py rlhf [subcomando]`.

    Subcomandos:
        --stats          estado del pipeline
        --train-reward   entrenar reward model
        --ppo            ciclo PPO
        --rank "query"   rankear con policy entrenada
        --validate       validar reward antes de PPO
    """
    print("\n" + "=" * 70)
    print("  RLHF — Reward de Rankings + PPO")
    print("=" * 70)

    system = _load_system()
    if not system:
        return

    try:
        pipeline = add_rlhf_to_system(system)
    except Exception as e:
        print(f"\n  Error inicializando RLHF: {e}")
        import traceback
        traceback.print_exc()
        return

    sub = args[0] if args else "--stats"

    if sub == "--stats":
        _show_stats(pipeline)

    elif sub == "--validate":
        print("\n  Validando reward model...")
        ok = pipeline.validate_reward_before_ppo(min_accuracy=0.60)
        print(f"\n  {'[OK] Listo para PPO' if ok else '[ERR] Necesitas mas datos o mas epocas'}")

    elif sub == "--train-reward":
        epochs = int(args[1]) if len(args) > 1 and args[1].isdigit() else 40
        print(f"\n  Entrenando reward model ({epochs} epocas)...")
        print("  [NOTA] Para Fase 1, usa directamente:")
        print("  python train_pointwise_reward.py")
        print("  Este comando (RankingRewardTrainer) está deprecated para Fase 1\n")
        result = pipeline.train_reward_model(epochs=epochs, min_pairs=10)
        if "error" in result:
            print(f"\n  [ERR] Error: {result['error']}")
        else:
            print(f"\n  [OK] Entrenado")
            print(f"    val_accuracy:  {result.get('best_val_accuracy', 0):.3f}")
            print(f"    val_loglik:    {result.get('final_val_loglik', 0):.3f}")
            print(f"    train/val:     {result.get('n_train', 0)}/{result.get('n_val', 0)}")
            if result.get('best_val_accuracy', 0) < 0.60:
                print("\n  [WARN] Accuracy < 60% — recolecta mas comparaciones A/B antes de PPO")
            else:
                print("\n  -> Puedes ejecutar: python main.py rlhf --ppo")

    elif sub == "--ppo":
        n_queries = int(args[1]) if len(args) > 1 and args[1].isdigit() else 50
        epochs = int(args[2]) if len(args) > 2 and args[2].isdigit() else 5
        print(f"\n  Ciclo PPO ({n_queries} queries, {epochs} epocas)...")
        result = pipeline.run_ppo_cycle(
            n_queries=n_queries,
            epochs=epochs,
            validate_first=True,
        )
        if "error" in result:
            print(f"\n  [ERR] Error: {result['error']}")
        else:
            print(f"\n  [OK] PPO completado")
            print(f"    Reward final:      {result.get('final_reward', 0):.4f}")
            print(f"    KL promedio:       {result.get('avg_kl', 0):.4f}")
            print(f"    Policy actualizada: {'[OK]' if result.get('policy_updated') else '[ERR] (revisar KL/reward)'}")
            versions = pipeline.ppo_trainer.list_versions()
            if versions:
                print(f"    Versiones guardadas: {[v.name for v in versions]}")

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
        print(f"\n  Top 10 {'(RLHF policy)' if pipeline.policy_trained else '(baseline FAISS)'}:")
        for i, p in enumerate(ranked[:10], 1):
            title = getattr(p, "title", "")[:65]
            pid = getattr(p, "id", "")
            rating = getattr(p, "rating", None)
            r_str = f"*{float(rating):.1f}" if rating else "    "
            print(f"  {i:2d}. {r_str}  {title}")

    else:
        print(f"\n  Subcomando desconocido: '{sub}'")
        print("  Disponibles: --stats | --validate | --train-reward | --ppo | --rank 'query'")

    print()


# ---------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------

def _build_product_index(system, embedding_model) -> Dict[str, np.ndarray]:
    """
    Computa embeddings desde titulos de productos. Lento (~15 min para 90k).
    Solo se llama si el cache no existe y FAISS no pudo reconstruir.
    """
    products = getattr(system, 'canonical_products', [])
    if not products:
        logger.warning("  No hay productos en el sistema")
        return {}

    pids, titles = [], []
    for p in products:
        pid = getattr(p, 'id', None) or getattr(p, 'product_id', None)
        title = getattr(p, 'title', '') or ''
        if pid and title:
            pids.append(pid)
            titles.append(title)

    if not titles:
        return {}

    batch_size = 512  # mas grande = mas rápido
    all_embs = []
    total = len(titles)
    for i in range(0, total, batch_size):
        batch = titles[i:i + batch_size]
        embs = embedding_model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embs.append(embs)
        pct = (i + len(batch)) / total * 100
        if i % (batch_size * 10) == 0:
            print(f"    {i+len(batch):,}/{total:,} ({pct:.0f}%)")

    all_embs_np = np.vstack(all_embs)
    return {pid: emb for pid, emb in zip(pids, all_embs_np)}


def _get_embedding_model(system):
    """
    Extrae el modelo de embeddings del sistema.
    Busca en los lugares mas comunes donde UnifiedSystemV2 lo guarda.
    """
    # Opción 1: canonicalizer directo
    if hasattr(system, 'canonicalizer') and system.canonicalizer is not None:
        emb = getattr(system.canonicalizer, 'embedding_model', None)
        if emb is not None:
            return emb

    # Opción 2: embedding_model directo en el sistema
    if hasattr(system, 'embedding_model') and system.embedding_model is not None:
        return system.embedding_model

    # Opción 3: vector_store tiene el modelo
    if hasattr(system, 'vector_store') and system.vector_store is not None:
        emb = getattr(system.vector_store, 'embedding_model', None)
        if emb is not None:
            return emb

    # Opción 4: cargar directamente
    logger.warning("No se encontró embedding_model en el sistema. Cargando sentence-transformer...")
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        logger.error(f"No se pudo cargar sentence-transformer: {e}")
        return None


PRODUCT_INDEX_CACHE = Path("data/cache/product_embeddings.npz")


def _load_or_build_product_index(system, embedding_model) -> Dict[str, np.ndarray]:
    """
    Carga el indice product_id -> embedding desde cache si existe,
    o lo construye y lo guarda para la próxima vez.

    Cache: data/cache/product_embeddings.npz
    Primera vez: ~15 minutos para 89k productos
    Siguientes:  <1 segundo
    """
    # -- Intentar cargar desde cache ----------------------------------
    if PRODUCT_INDEX_CACHE.exists():
        try:
            print("  Cargando indice de embeddings desde cache...")
            data = np.load(PRODUCT_INDEX_CACHE, allow_pickle=True)
            ids = data['ids']
            embs = data['embeddings']
            product_index = {pid: emb for pid, emb in zip(ids, embs)}
            print(f"  [OK] Cache cargada: {len(product_index):,} productos ({PRODUCT_INDEX_CACHE})")
            return product_index
        except Exception as e:
            logger.warning(f"  Cache corrupta, reconstruyendo: {e}")

    # -- Intentar extraer desde FAISS (rápido, sin recomputar) --------
    print("  Intentando extraer embeddings del indice FAISS...")
    product_index = _extract_from_faiss(system)
    if product_index:
        print(f"  [OK] Extraídos del FAISS: {len(product_index):,} productos")
        _save_index_cache(product_index)
        return product_index

    # -- Computar desde titulos (lento, último recurso) ---------------
    print("  Computando embeddings desde titulos de productos...")
    print("  (Solo ocurre la primera vez — se guardará en cache)")
    product_index = _build_product_index(system, embedding_model)
    if product_index:
        _save_index_cache(product_index)
    return product_index


def _extract_from_faiss(system) -> Dict[str, np.ndarray]:
    """
    Intenta extraer vectores directamente del indice FAISS.
    Si tu VectorStore lo soporta, es instantáneo.
    """
    product_index = {}
    try:
        vs = getattr(system, 'vector_store', None)
        if vs is None:
            return {}

        id_to_idx = (
            getattr(vs, 'id_to_index', None)
            or getattr(vs, '_id_map', None)
            or getattr(vs, 'docstore', {})
        )
        faiss_index = (
            getattr(vs, 'index', None)
            or getattr(vs, '_index', None)
        )

        if not id_to_idx or faiss_index is None:
            return {}

        import faiss
        n = faiss_index.ntotal
        if n == 0:
            return {}

        # reconstruct_n funciona en IndexFlatIP y IndexFlatL2
        # Falla en indices comprimidos (IVF, PQ)
        all_vectors = np.zeros((n, faiss_index.d), dtype=np.float32)
        faiss_index.reconstruct_n(0, n, all_vectors)

        for pid, idx in id_to_idx.items():
            if isinstance(idx, int) and 0 <= idx < n:
                product_index[pid] = all_vectors[idx]

        return product_index

    except Exception as e:
        logger.debug(f"  No se pudo extraer de FAISS ({type(e).__name__}): {e}")
        return {}


def _save_index_cache(product_index: Dict[str, np.ndarray]):
    """Guarda el indice en disco como .npz comprimido."""
    try:
        PRODUCT_INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ids = np.array(list(product_index.keys()))
        embs = np.stack(list(product_index.values()))
        np.savez_compressed(PRODUCT_INDEX_CACHE, ids=ids, embeddings=embs)
        size_mb = PRODUCT_INDEX_CACHE.stat().st_size / 1e6
        print(f"  [OK] Cache guardada: {PRODUCT_INDEX_CACHE} ({size_mb:.1f} MB)")
    except Exception as e:
        logger.warning(f"  No se pudo guardar cache: {e}")


def _get_vector_store(system):
    """Extrae el vector store del sistema."""
    vs = getattr(system, 'vector_store', None)
    if vs is not None:
        return vs
    # Algunos sistemas lo guardan con otro nombre
    vs = getattr(system, 'faiss_store', None) or getattr(system, '_vector_store', None)
    return vs


def _load_system():
    """Carga UnifiedSystemV2 desde cache."""
    try:
        cache_path = Path("data/cache/unified_system_v2.pkl")
        if not cache_path.exists():
            print("  Sistema no encontrado. Ejecuta primero: python main.py init")
            return None

        from src.unified_system_v2 import UnifiedSystemV2
        system = UnifiedSystemV2.load_from_cache()

        if not system:
            print("  Error cargando sistema")
            return None

        print(f"  Sistema cargado: {len(system.canonical_products):,} productos")
        return system

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def _try_load_pointwise(pipeline):
    """
    Carga checkpoint del reward model pointwise si existe.
    Fase 1: solo PointwiseRewardModel.
    """
    ckpt_path = Path("data/rlhf_checkpoints/reward_model.pt")
    if not ckpt_path.exists():
        return
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # Intentar diferentes formatos de checkpoint
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            pipeline.reward_model.load_state_dict(ckpt['model_state'])
            pipeline.reward_trained = True
            logger.info("  Reward model (pointwise) cargado desde checkpoint (formato con model_state)")
        elif isinstance(ckpt, dict) and all(k.isdigit() for k in ckpt.keys()):
            # Es un state_dict directo
            pipeline.reward_model.load_state_dict(ckpt)
            pipeline.reward_trained = True
            logger.info("  Reward model (pointwise) cargado desde checkpoint (state_dict directo)")
        else:
            logger.debug(f"  Formato de checkpoint no reconocido: {type(ckpt)}")
            return
            
        pipeline.reward_model.eval()
        
    except Exception as e:
        logger.debug(f"  _try_load_pointwise: {e}")


def _show_stats(pipeline):
    """Muestra el estado completo del pipeline."""
    status = pipeline.get_status()

    print("\n  ESTADO DEL PIPELINE RLHF:")
    print(f"    Reward model entrenado:  {'[OK]' if status['reward_trained'] else '[ERR]'}")
    print(f"    Policy entrenada (PPO):  {'[OK]' if status['policy_trained'] else '[ERR]'}")
    print(f"    Device:                  {status['device']}")

    cs = status.get('collector_stats', {})
    print(f"\n  PREFERENCIAS A/B:")
    print(f"    Total:           {cs.get('total', 0)}")
    print(f"    Prefirió A:      {cs.get('prefer_A', 0)}")
    print(f"    Prefirió B:      {cs.get('prefer_B', 0)}")
    print(f"    Empates:         {cs.get('equal', 0)}")
    print(f"    Queries únicas:  {cs.get('unique_queries', 0)}")
    print(f"    Coverage indice: {cs.get('coverage_pct', 0)}%")

    ckpts = status.get('checkpoints', {})
    print(f"\n  CHECKPOINTS:")
    print(f"    reward_model.pt: {'[OK]' if ckpts.get('reward') else '[ERR]'}")
    print(f"    policy_model.pt: {'[OK]' if ckpts.get('policy') else '[ERR]'}")
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
        print(f"    -> Recolecta preferencias A/B: python sistema_interactivo.py")
        print(f"      (tienes {n}, necesitas 10+)")
    elif not status['reward_trained']:
        print(f"    -> Entrena el reward: python main.py rlhf --train-reward")
    elif not status['policy_trained']:
        print(f"    -> Valida y entrena PPO: python main.py rlhf --validate")
        print(f"                             python main.py rlhf --ppo")
    else:
        print(f"    -> Evalúa el experimento: python main.py experimento")