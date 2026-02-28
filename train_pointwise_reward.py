"""
train_pointwise_reward.py
=========================
Entrena el PointwiseRewardModel desde las preferencias A/B recolectadas
con `python main.py interactivo`.

NO usa ESCI ni datos externos. Los pares vienen de:
    data/preferences/preferences.jsonl

Cada línea tiene:
    {query, chosen_products, rejected_products, preference, timestamp, ...}

Uso:
    python train_pointwise_reward.py
    python train_pointwise_reward.py --epochs 40 --lr 1e-4
    python train_pointwise_reward.py --min-pairs 10   # bajar mínimo para pruebas
"""

import argparse
import json
import logging
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PREFERENCES_PATH = Path("data/preferences/preferences.jsonl")
CHECKPOINT_DIR   = Path("data/rlhf_checkpoints")


# ---------------------------------------------------------------------------
# Carga de preferencias A/B
# ---------------------------------------------------------------------------

def load_ab_preferences(path: Path) -> list:
    """
    Lee preferences.jsonl y extrae pares (query, chosen_id, rejected_id).

    Formato del sistema:
        {
            "query": "action video games",
            "ranking_a_ids": ["B00005RFCT", ...],   # 10 productos en orden
            "ranking_b_ids": ["B00005RFCT", ...],   # mismos productos, distinto orden
            "preference": "B"
        }

    Estrategia de extracción de pares:
        Los dos rankings contienen los mismos (o casi los mismos) productos
        pero en distinto orden. La preferencia indica cuál orden es mejor.

        Un par (chosen, rejected) válido es:
            - chosen aparece en posición i del ranking ganador
            - rejected aparece en posición j del ranking perdedor
            - i < j  (chosen está más arriba en el ganador)
            - El mismo producto está más abajo en el perdedor o no aparece

        Esto captura la señal real: "este producto debería estar más arriba".
    """
    if not path.exists():
        logger.error(f"No encontrado: {path}")
        logger.error("Ejecuta primero: python main.py interactivo")
        sys.exit(1)

    pairs   = []
    skipped = 0

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            query = rec.get('query', '').strip()
            pref  = str(rec.get('preference', '')).strip().upper()
            ids_a = rec.get('ranking_a_ids', [])
            ids_b = rec.get('ranking_b_ids', [])

            if pref not in ('A', 'B') or not query or not ids_a or not ids_b:
                skipped += 1
                continue

            # winner = ranking preferido, loser = ranking rechazado
            if pref == 'A':
                winner, loser = ids_a, ids_b
            else:
                winner, loser = ids_b, ids_a

            # Mapa posición en cada ranking
            winner_pos = {pid: i for i, pid in enumerate(winner)}
            loser_pos  = {pid: i for i, pid in enumerate(loser)}

            # Pares: producto mejor posicionado en winner vs producto
            # mejor posicionado en loser, donde hay diferencia de rango.
            # Solo tomamos los top-5 de cada ranking para no generar ruido.
            top_winner = winner[:5]
            top_loser  = loser[:5]

            for c_id in top_winner:
                for r_id in top_loser:
                    if c_id == r_id:
                        continue
                    pos_c_winner = winner_pos.get(c_id, 99)
                    pos_r_winner = winner_pos.get(r_id, 99)
                    pos_c_loser  = loser_pos.get(c_id, 99)
                    pos_r_loser  = loser_pos.get(r_id, 99)

                    # c_id está más arriba que r_id en el ranking ganador
                    # Y r_id está más arriba que c_id en el ranking perdedor
                    # → señal clara de preferencia de orden
                    if pos_c_winner < pos_r_winner and pos_r_loser < pos_c_loser:
                        pairs.append({
                            'query':    query,
                            'chosen':   c_id,
                            'rejected': r_id,
                        })

    logger.info(f"Preferencias leídas: {len(pairs)} pares útiles "
                f"({skipped} omitidos)")
    return pairs


# ---------------------------------------------------------------------------
# Construcción de tensores
# ---------------------------------------------------------------------------

def build_tensors(pairs: list, prod_cache: dict, query_cache: dict, device: str):
    """Construye tensores de entrenamiento a partir de pares A/B."""
    q_list, c_list, r_list = [], [], []
    skipped = 0

    for pair in pairs:
        q_emb = query_cache.get(pair['query'])
        c_emb = prod_cache.get(str(pair['chosen']))
        r_emb = prod_cache.get(str(pair['rejected']))

        if q_emb is None or c_emb is None or r_emb is None:
            skipped += 1
            continue

        q_list.append(q_emb)
        c_list.append(c_emb)
        r_list.append(r_emb)

    if skipped:
        logger.warning(f"  Pares sin embedding: {skipped} omitidos")

    if not q_list:
        logger.error(
            "Sin tensores válidos. "
            "Verifica que los product IDs en preferences.jsonl "
            "coincidan con los del índice de embeddings."
        )
        sys.exit(1)

    logger.info(f"  Tensores válidos: {len(q_list)}")

    q = torch.tensor(np.array(q_list), dtype=torch.float32).to(device)
    c = torch.tensor(np.array(c_list), dtype=torch.float32).to(device)
    r = torch.tensor(np.array(r_list), dtype=torch.float32).to(device)
    return q, c, r


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train(model, q_tr, c_tr, r_tr, q_vl, c_vl, r_vl,
          epochs, lr, batch_size, margin, device):

    try:
        from src.rlhf.pointwise_reward_model import PointwiseMarginLoss
    except ImportError:
        from rlhf.pointwise_reward_model import PointwiseMarginLoss

    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion  = PointwiseMarginLoss(base_margin=margin)
    n          = q_tr.size(0)
    best_acc   = 0.0
    best_state = None
    patience   = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        perm = torch.randperm(n, device=device)

        for i in range(0, n, batch_size):
            idx   = perm[i:i + batch_size]
            s_c   = model(q_tr[idx], c_tr[idx])
            s_r   = model(q_tr[idx], r_tr[idx])
            # diff=1 para todos (preferencia binaria A/B)
            diff  = torch.ones(idx.size(0), device=device)
            loss  = criterion(s_c, s_r, diff, diff)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * idx.size(0)

        scheduler.step()

        # Validación
        model.eval()
        with torch.no_grad():
            s_c_v = model(q_vl, c_vl)
            s_r_v = model(q_vl, r_vl)
            val_acc  = (s_c_v > s_r_v).float().mean().item()
            diff_v   = torch.ones(q_vl.size(0), device=device)
            val_loss = criterion(s_c_v, s_r_v, diff_v, diff_v).item()

        marker = ' ←' if val_acc > best_acc else ''
        logger.info(
            f"  Epoch {epoch:2d}/{epochs} | "
            f"loss={total_loss/n:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.3f}{marker}"
        )

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1
            if patience >= 7 and epoch > epochs // 2:
                logger.info(f"  Early stopping en época {epoch} (patience=7)")
                break

    if best_state:
        model.load_state_dict(best_state)

    return best_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Entrenar PointwiseRewardModel desde preferencias A/B"
    )
    parser.add_argument('--epochs',    type=int,   default=30)
    parser.add_argument('--lr',        type=float, default=2e-4)
    parser.add_argument('--batch',     type=int,   default=32,
                        help='Batch size (default 32 — pequeño por pocos datos)')
    parser.add_argument('--margin',    type=float, default=0.5,
                        help='Margen de la loss (default 0.5 — más suave para pocos datos)')
    parser.add_argument('--hidden',    type=int,   default=256)
    parser.add_argument('--val-split', type=float, default=0.15)
    parser.add_argument('--min-pairs', type=int,   default=20,
                        help='Mínimo de pares para entrenar (default 20)')
    parser.add_argument('--preferences', type=str, default=str(PREFERENCES_PATH))
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ENTRENAMIENTO REWARD MODEL")
    print("  Fuente: preferencias A/B recolectadas")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Dispositivo: {device}")

    # Cargar sistema
    try:
        from src.unified_system_v2 import UnifiedSystemV2
        system    = UnifiedSystemV2.load_from_cache()
        emb_model = system.canonicalizer.embedding_model
        logger.info(f"Sistema: {len(system.canonical_products):,} productos")
    except Exception as e:
        logger.error(f"Error cargando sistema: {e}")
        sys.exit(1)

    # Cargar preferencias A/B
    pairs = load_ab_preferences(Path(args.preferences))

    if len(pairs) < args.min_pairs:
        logger.error(
            f"Solo {len(pairs)} pares — mínimo {args.min_pairs}.\n"
            f"Recolecta más preferencias: python main.py interactivo"
        )
        sys.exit(1)

    # Embeddings de productos desde cache
    cache_path = Path("data/cache/product_embeddings.npz")
    if not cache_path.exists():
        logger.error(f"Cache no encontrada: {cache_path}")
        logger.error("Ejecuta: python main.py init  (o python main.py interactivo)")
        sys.exit(1)

    logger.info("Cargando cache de embeddings...")
    data       = np.load(cache_path, allow_pickle=True)
    prod_cache = {str(pid): emb.astype(np.float32)
                  for pid, emb in zip(data['ids'], data['embeddings'])}
    logger.info(f"  {len(prod_cache):,} productos con embedding")

    # Embeddings de queries
    queries = list({p['query'] for p in pairs})
    logger.info(f"Generando embeddings para {len(queries)} queries...")
    query_cache = {}
    for q in queries:
        query_cache[q] = emb_model.encode(q, normalize_embeddings=True).astype(np.float32)

    # Construir tensores
    q_t, c_t, r_t = build_tensors(pairs, prod_cache, query_cache, device)

    # Split train/val
    n     = q_t.size(0)
    val_n = max(1, int(n * args.val_split))
    perm  = torch.randperm(n)
    vi    = perm[:val_n]
    ti    = perm[val_n:]

    logger.info(f"Train: {len(ti)} pares | Val: {len(vi)} pares")

    if len(ti) < 5:
        logger.error(
            f"Muy pocos pares de entrenamiento ({len(ti)}). "
            "Recolecta más preferencias A/B."
        )
        sys.exit(1)

    # Crear modelo
    try:
        from src.rlhf.pointwise_reward_model import PointwiseRewardModel
    except ImportError:
        from rlhf.pointwise_reward_model import PointwiseRewardModel

    emb_dim = q_t.size(1)
    model   = PointwiseRewardModel(emb_dim=emb_dim, hidden_dim=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Modelo: {n_params:,} parámetros | emb_dim={emb_dim}")

    # Entrenar
    print(f"\n  Configuración:")
    print(f"    Pares totales:   {n}")
    print(f"    Épocas:          {args.epochs}")
    print(f"    Learning rate:   {args.lr}")
    print(f"    Batch size:      {args.batch}")
    print(f"    Margin:          {args.margin}")
    print()

    t0       = time.time()
    best_acc = train(
        model,
        q_t[ti], c_t[ti], r_t[ti],
        q_t[vi], c_t[vi], r_t[vi],
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        margin=args.margin,
        device=device,
    )
    elapsed = time.time() - t0

    # Guardar
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            'model_state': model.state_dict(),
            'config': {
                'emb_dim':    emb_dim,
                'hidden_dim': args.hidden,
                'model_type': 'pointwise',
            },
            'training': {
                'n_pairs':   n,
                'val_acc':   best_acc,
                'epochs':    args.epochs,
                'source':    'ab_preferences',
            }
        },
        CHECKPOINT_DIR / "reward_model.pt",
    )
    logger.info(f"Guardado: {CHECKPOINT_DIR}/reward_model.pt")

    # Diagnóstico final
    model.eval()
    with torch.no_grad():
        s_c = model(q_t[vi], c_t[vi])
        s_r = model(q_t[vi], r_t[vi])
        sep = (s_c - s_r).mean().item()

    print("\n" + "=" * 60)
    print("  RESULTADO")
    print("=" * 60)
    print(f"  Pares totales:     {n}")
    print(f"  val_accuracy:      {best_acc:.3f}")
    print(f"  chosen - rejected: {sep:+.4f}  ({'OK' if sep > 0 else 'ERR — revisar'})")
    print(f"  Tiempo:            {elapsed:.0f}s")
    print()

    if best_acc >= 0.65:
        print("  [OK] Reward model listo")
        print("\n  SIGUIENTE PASO:")
        print("    python main.py rlhf --ppo")
        print("    python evaluate_methods.py")
    else:
        print(f"  [WARN] val_accuracy={best_acc:.3f} < 0.65")
        print("  Con pocos pares A/B es normal. Opciones:")
        print("    1. Recolectar más preferencias:  python main.py interactivo")
        print("    2. Reducir margin:               --margin 0.3")
        print("    3. Igual puedes continuar con evaluate_methods.py")
    print("=" * 60)


if __name__ == "__main__":
    main()