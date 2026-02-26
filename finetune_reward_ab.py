# finetune_reward_ab.py
"""
=====================
Fine-tuning del reward model pre-entrenado con ESCI
usando las preferencias humanas A/B recolectadas.

Secuencia correcta:
    1. python verify_and_split_esci.py      (una sola vez)
    2. python pretrain_reward_esci.py       -> reward_model_esci_pretrained.pt
    3. python finetune_reward_ab.py         <- ESTE SCRIPT
    4. python main.py rlhf --ppo
    5. python evaluate_esci.py

Qué hace:
    - Carga reward_model_esci_pretrained.pt (no el .pt final)
    - Fine-tunea con tus 82 preferencias humanas claras
    - LR bajo (3e-5) para no destruir el pretrain
    - Guarda reward_model.pt (el que usa PPO)
"""

import json
import logging
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

FINETUNE_EPOCHS  = 15
FINETUNE_LR      = 3e-5
CHECKPOINT_DIR   = Path("data/rlhf_checkpoints")
PRETRAINED_PATH  = CHECKPOINT_DIR / "reward_model_esci_pretrained.pt"
FINAL_PATH       = CHECKPOINT_DIR / "reward_model.pt"
PREFS_FILE       = Path("data/preferences/preferences.jsonl")
EMB_CACHE        = Path("data/cache/product_embeddings.npz")
EMB_DIM          = 384
TOP_K            = 10


def load_product_index():
    if not EMB_CACHE.exists():
        logger.error(f"Embeddings no encontrados: {EMB_CACHE}")
        sys.exit(1)
    data = np.load(EMB_CACHE, allow_pickle=True)
    idx = {str(pid): emb for pid, emb in zip(data['ids'], data['embeddings'])}
    logger.info(f"Embeddings cargados: {len(idx):,} productos")
    return idx


def get_emb(asin, product_index):
    return product_index.get(str(asin), np.zeros(EMB_DIM, dtype=np.float32))


def load_preferences():
    if not PREFS_FILE.exists():
        logger.error(f"No encontrado: {PREFS_FILE}")
        sys.exit(1)
    all_prefs = []
    with open(PREFS_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_prefs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    clear = [p for p in all_prefs if p.get('preference') in ('A', 'B')]
    logger.info(f"Preferencias: {len(all_prefs)} totales | {len(clear)} claras | {len(all_prefs)-len(clear)} empates")
    return clear


def build_tensors(prefs, product_index, embedding_model):
    q_list, ra_list, rb_list, pref_list = [], [], [], []
    skipped = 0

    for p in prefs:
        query  = p.get('query', '')
        pref   = p.get('preference', '')
        ids_a  = p.get('ranking_a_ids') or []
        ids_b  = p.get('ranking_b_ids') or []

        if not ids_a or not ids_b:
            skipped += 1
            continue

        q_emb = embedding_model.encode(query, normalize_embeddings=True)
        ra = np.zeros((TOP_K, EMB_DIM), dtype=np.float32)
        rb = np.zeros((TOP_K, EMB_DIM), dtype=np.float32)
        for k, asin in enumerate(ids_a[:TOP_K]):
            ra[k] = get_emb(asin, product_index)
        for k, asin in enumerate(ids_b[:TOP_K]):
            rb[k] = get_emb(asin, product_index)

        q_list.append(q_emb)
        ra_list.append(ra)
        rb_list.append(rb)
        pref_list.append(pref)

    if skipped:
        logger.warning(f"  {skipped} preferencias omitidas (sin ranking_a_ids / ranking_b_ids)")

    if not q_list:
        logger.error("Sin tensores — verifica que preferences.jsonl tiene 'ranking_a_ids' y 'ranking_b_ids'")
        sys.exit(1)

    q_t  = torch.tensor(np.array(q_list),  dtype=torch.float32)
    ra_t = torch.tensor(np.array(ra_list), dtype=torch.float32)
    rb_t = torch.tensor(np.array(rb_list), dtype=torch.float32)
    logger.info(f"Tensores construidos: {len(pref_list)} pares válidos")
    return q_t, ra_t, rb_t, pref_list


def accuracy(model, q, ra, rb, prefs):
    model.eval()
    correct = 0
    with torch.no_grad():
        r_a = model(q, ra)
        r_b = model(q, rb)
        for i, p in enumerate(prefs):
            if (p == 'A' and r_a[i] > r_b[i]) or (p == 'B' and r_b[i] > r_a[i]):
                correct += 1
    return correct / len(prefs)


def bt_loglik(model, q, ra, rb, prefs):
    model.eval()
    lls = []
    with torch.no_grad():
        r_a = model(q, ra)
        r_b = model(q, rb)
        for i, p in enumerate(prefs):
            diff = r_a[i] - r_b[i] if p == 'A' else r_b[i] - r_a[i]
            lls.append(F.logsigmoid(diff).item())
    return float(np.mean(lls))


def main():
    print("\n" + "="*60)
    print("  FINE-TUNING REWARD — PREFERENCIAS HUMANAS A/B")
    print("="*60)

    # Reward model
    from src.rlhf.reward_model import RankingRewardModel
    model = RankingRewardModel()

    if PRETRAINED_PATH.exists():
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location='cpu'))
        logger.info(f"[OK] Cargado: {PRETRAINED_PATH}")
    else:
        logger.warning(f"[WARN] No encontrado: {PRETRAINED_PATH}")
        logger.warning("  Ejecuta primero: python pretrain_reward_esci.py")
        logger.warning("  Continuando con weights iniciales...")

    # Embeddings
    product_index = load_product_index()

    # Embedding model
    emb_model = None
    try:
        from src.unified_system_v2 import UnifiedSystemV2
        system = UnifiedSystemV2.load_from_cache()
        for attr in ['embedding_model', 'emb_model']:
            m = getattr(system, attr, None)
            if m:
                emb_model = m
                break
            try:
                m = getattr(system.canonicalizer, attr, None)
                if m:
                    emb_model = m
                    break
            except Exception:
                pass
    except Exception:
        pass

    if emb_model is None:
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("  Usando SentenceTransformer: all-MiniLM-L6-v2")

    # Preferencias
    clear_prefs = load_preferences()

    rng = np.random.default_rng(42)
    indices = np.arange(len(clear_prefs))
    rng.shuffle(indices)
    n_val     = max(1, len(clear_prefs) // 5)
    val_prefs   = [clear_prefs[i] for i in indices[:n_val]]
    train_prefs = [clear_prefs[i] for i in indices[n_val:]]
    logger.info(f"Split: {len(train_prefs)} train / {len(val_prefs)} val (seed=42)")

    # Tensores
    q_tr, ra_tr, rb_tr, pr_tr = build_tensors(train_prefs, product_index, emb_model)
    q_va, ra_va, rb_va, pr_va = build_tensors(val_prefs,   product_index, emb_model)

    # Métricas pre-finetune
    pre_val_acc = accuracy(model, q_va, ra_va, rb_va, pr_va)
    pre_val_ll  = bt_loglik(model, q_va, ra_va, rb_va, pr_va)
    print(f"\n  PRE-finetune  ->  val_acc={pre_val_acc:.3f}  val_BT-LL={pre_val_ll:.4f}")

    # Training
    optimizer  = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=0.01)
    best_val_acc = pre_val_acc
    best_epoch   = 0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), FINAL_PATH)

    print(f"\n  {'Ép':>3}  {'Loss':>8}  {'Tr Acc':>7}  {'Va Acc':>7}  {'Va LL':>8}")
    print(f"  {'-'*45}")

    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        r_a = model(q_tr, ra_tr)
        r_b = model(q_tr, rb_tr)

        preferred, rejected = [], []
        for i, p in enumerate(pr_tr):
            if p == 'A':
                preferred.append(r_a[i]); rejected.append(r_b[i])
            else:
                preferred.append(r_b[i]); rejected.append(r_a[i])

        loss = -F.logsigmoid(torch.stack(preferred) - torch.stack(rejected)).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tr_acc = accuracy(model, q_tr, ra_tr, rb_tr, pr_tr)
        va_acc = accuracy(model, q_va, ra_va, rb_va, pr_va)
        va_ll  = bt_loglik(model, q_va, ra_va, rb_va, pr_va)

        marker = ""
        if va_acc >= best_val_acc:
            best_val_acc = va_acc
            best_epoch   = epoch + 1
            torch.save(model.state_dict(), FINAL_PATH)
            marker = " [OK]"

        print(f"  {epoch+1:>3}  {loss.item():>8.4f}  {tr_acc:>7.3f}  {va_acc:>7.3f}  {va_ll:>8.4f}{marker}")

    # Cargar mejor
    model.load_state_dict(torch.load(FINAL_PATH, map_location='cpu'))
    final_val_acc = accuracy(model, q_va, ra_va, rb_va, pr_va)
    final_val_ll  = bt_loglik(model, q_va, ra_va, rb_va, pr_va)

    delta = final_val_acc - pre_val_acc
    arrow = '↑' if delta > 0 else ('↓' if delta < 0 else '=')

    print(f"\n  {'='*60}")
    print(f"  COMPLETADO")
    print(f"  Mejor época:     {best_epoch}/{FINETUNE_EPOCHS}")
    print(f"  Val accuracy:    {pre_val_acc:.3f} -> {final_val_acc:.3f}  {arrow}{abs(delta):.3f}")
    print(f"  Val BT log-lik:  {pre_val_ll:.4f} -> {final_val_ll:.4f}")
    print(f"  Checkpoint:      {FINAL_PATH}")
    print(f"\n  Siguiente paso:")
    print(f"    python main.py rlhf --ppo")
    print(f"    python evaluate_esci.py")


if __name__ == "__main__":
    main()