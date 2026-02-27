"""
train_pointwise_reward.py
Formato confirmado de pretrain_pairs.jsonl:
    query, chosen_asin, rejected_asin, score_diff, weight
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
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
CHECKPOINT_DIR = Path("data/rlhf_checkpoints")


def load_pairs(path):
    """Carga pares desde archivo JSONL"""
    pairs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    logger.info(f"Pares cargados: {len(pairs):,} desde {path}")
    return pairs


def load_prod_cache(path):
    """Carga caché de embeddings de productos"""
    logger.info("Cargando embeddings de productos...")
    data = np.load(path, allow_pickle=True)
    cache = {str(pid): emb.astype(np.float32)
             for pid, emb in zip(data['ids'], data['embeddings'])}
    logger.info(f"  {len(cache):,} embeddings")
    return cache


def gen_query_embs(pairs, emb_model):
    """Genera embeddings para todas las queries únicas"""
    queries = list({p['query'] for p in pairs})
    logger.info(f"Generando embeddings para {len(queries):,} queries...")
    cache, batch = {}, 512
    
    for i in range(0, len(queries), batch):
        b = queries[i:i+batch]
        embs = emb_model.encode(b, normalize_embeddings=True, show_progress_bar=False)
        for q, e in zip(b, embs):
            cache[q] = e.astype(np.float32)
        if i % (batch * 20) == 0 and i > 0:
            logger.info(f"  [{i:,}/{len(queries):,}]")
    
    logger.info(f"  {len(cache):,} queries embedidas")
    return cache


def build_tensors(pairs, prod_cache, query_cache, device):
    """Construye tensores a partir de pares y caches"""
    logger.info(f"Construyendo tensores para {len(pairs):,} pares...")
    q_l, a_l, b_l, d_l, w_l = [], [], [], [], []
    skipped = 0
    
    for p in pairs:
        q_emb = query_cache.get(p['query'])
        a_emb = prod_cache.get(str(p['chosen_asin']))
        b_emb = prod_cache.get(str(p['rejected_asin']))
        
        if q_emb is None or a_emb is None or b_emb is None:
            skipped += 1
            continue
            
        q_l.append(q_emb)
        a_l.append(a_emb)
        b_l.append(b_emb)
        d_l.append(float(p['score_diff']))
        w_l.append(float(p.get('weight', p['score_diff'])))

    if skipped:
        logger.warning(f"  Omitidos: {skipped:,}")
    
    logger.info(f"  Válidos: {len(q_l):,}")
    
    if not q_l:
        logger.error("Sin tensores válidos.")
        sys.exit(1)

    w = np.array(w_l, dtype=np.float32)
    w = w / w.max()  # normalizar pesos
    
    return (
        torch.tensor(np.array(q_l), dtype=torch.float32).to(device),
        torch.tensor(np.array(a_l), dtype=torch.float32).to(device),
        torch.tensor(np.array(b_l), dtype=torch.float32).to(device),
        torch.tensor(np.array(d_l), dtype=torch.float32).to(device),
        torch.tensor(w, dtype=torch.float32).to(device),
    )


def train(model, q_tr, a_tr, b_tr, d_tr, w_tr,
          q_vl, a_vl, b_vl, d_vl, w_vl,
          epochs, lr, batch_size, margin, device):
    """Entrena el modelo pointwise"""
    try:
        from src.rlhf.pointwise_reward_model import PointwiseMarginLoss
    except ImportError:
        from src.rlhf.pointwise_reward_model import PointwiseMarginLoss
        
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = PointwiseMarginLoss(base_margin=margin)
    n = q_tr.size(0)
    best_acc, best_state, patience_cnt = 0.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        perm = torch.randperm(n, device=device)
        
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            s_a = model(q_tr[idx], a_tr[idx])
            s_b = model(q_tr[idx], b_tr[idx])
            loss = criterion(s_a, s_b, d_tr[idx], w_tr[idx])
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            total += loss.item() * idx.size(0)
        
        sched.step()

        # Validación
        model.eval()
        with torch.no_grad():
            s_a_v = model(q_vl, a_vl)
            s_b_v = model(q_vl, b_vl)
            val_acc = (s_a_v > s_b_v).float().mean().item()
            val_loss = criterion(s_a_v, s_b_v, d_vl, w_vl).item()

        logger.info(f"  Epoch {epoch:2d}/{epochs} | loss={total/n:.4f} | "
                    f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 5 and epoch > epochs // 2:
                logger.info(f"  Early stopping en época {epoch}")
                break

    # Cargar mejor estado
    if best_state:
        model.load_state_dict(best_state)

    # Diagnóstico final
    model.eval()
    with torch.no_grad():
        s_a_v = model(q_vl, a_vl)
        s_b_v = model(q_vl, b_vl)

    print("\n" + "-" * 52)
    print("  DIAGNÓSTICO POST-ENTRENAMIENTO")
    print("-" * 52)
    print(f"  val_accuracy global:     {best_acc:.3f}")
    
    for dv in [1, 2, 3]:
        mask = (d_vl == dv)
        if mask.sum() > 0:
            acc = (s_a_v[mask] > s_b_v[mask]).float().mean().item()
            lbl = {3: 'Exact>Irrel', 2: 'Exact>Compl', 1: 'diff=1'}.get(dv)
            print(f"  acc diff={dv} ({lbl}): {acc:.3f}  (n={mask.sum().item():,})")
    
    sep = (s_a_v - s_b_v).mean().item()
    print(f"  Media chosen-rejected:   {sep:.4f}")
    print(f"  {'[OK]' if sep > 0 else '[ERR]'} chosen {'>' if sep > 0 else '<'} rejected")
    print("-" * 52)
    
    return best_acc


def main():
    p = argparse.ArgumentParser(description="Entrenar reward model pointwise con pares ESCI")
    p.add_argument('--epochs', type=int, default=25,
                   help='Número de épocas (default: 25)')
    p.add_argument('--lr', type=float, default=2e-4,
                   help='Learning rate (default: 2e-4)')
    p.add_argument('--batch', type=int, default=256,
                   help='Batch size (default: 256)')
    p.add_argument('--margin', type=float, default=1.0,
                   help='Margen base para la loss (default: 1.0)')
    p.add_argument('--hidden', type=int, default=256,
                   help='Dimensión capas ocultas (default: 256)')
    p.add_argument('--pairs', type=str, default="data/esci/pretrain_pairs_pointwise.jsonl",
                   help='Ruta al archivo JSONL con pares (default: data/esci/pretrain_pairs_pointwise.jsonl)')
    p.add_argument('--emb-cache', type=str, default="data/cache/product_embeddings.npz",
                   help='Ruta al caché de embeddings (default: data/cache/product_embeddings.npz)')
    p.add_argument('--output-dir', type=str, default="data/rlhf_checkpoints",
                   help='Directorio de salida para checkpoints (default: data/rlhf_checkpoints)')
    p.add_argument('--val-split', type=float, default=0.1,
                   help='Fracción para validación (default: 0.1)')
    
    args = p.parse_args()

    print("\n" + "=" * 60)
    print("  FASE 1 - REWARD MODEL POINTWISE")
    print("  Arquitectura: concat(q, p, q-p, q*p) -> MLP")
    print("=" * 60)
    print(f"\nConfiguración:")
    print(f"  Pares:         {args.pairs}")
    print(f"  Embeddings:    {args.emb_cache}")
    print(f"  Épocas:        {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size:    {args.batch}")
    print(f"  Hidden dim:    {args.hidden}")
    print(f"  Margin:        {args.margin}")
    print(f"  Val split:     {args.val_split}")
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Dispositivo: {device}")

    # Verificar que existen los archivos necesarios
    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        logger.error(f"❌ Archivo de pares no encontrado: {pairs_path}")
        sys.exit(1)

    emb_cache_path = Path(args.emb_cache)
    if not emb_cache_path.exists():
        logger.error(f"❌ Caché de embeddings no encontrado: {emb_cache_path}")
        logger.error("Ejecuta primero: python src/rlhf_integration.py o init_esci_system.py")
        sys.exit(1)

    # Cargar datos
    pairs = load_pairs(pairs_path)
    prod_cache = load_prod_cache(emb_cache_path)

    # Cargar sistema para obtener modelo de embeddings
    try:
        from src.unified_system_v2 import UnifiedSystemV2
        system = UnifiedSystemV2.load_from_cache()
        emb_model = system.canonicalizer.embedding_model
        logger.info("✅ Sistema cargado correctamente")
    except Exception as e:
        logger.error(f"❌ Error cargando sistema: {e}")
        logger.error("¿Ejecutaste init_esci_system.py primero?")
        sys.exit(1)

    # Generar embeddings de queries
    query_cache = gen_query_embs(pairs, emb_model)

    # Construir tensores
    q_t, a_t, b_t, d_t, w_t = build_tensors(pairs, prod_cache, query_cache, device)

    # Split train/val
    n = q_t.size(0)
    val_n = max(1, int(n * args.val_split))
    perm = torch.randperm(n)
    vi, ti = perm[:val_n], perm[val_n:]

    logger.info(f"\nSplit - Train: {len(ti):,} | Val: {len(vi):,}")
    for dv in [1, 2, 3]:
        nd = (d_t[ti] == dv).sum().item()
        logger.info(f"  diff={dv}: {nd:,} ({nd/len(ti)*100:.1f}%)")

    # Crear modelo
    try:
        from src.rlhf.pointwise_reward_model import PointwiseRewardModel
    except ImportError:
        from src.rlhf.pointwise_reward_model import PointwiseRewardModel
        
    model = PointwiseRewardModel(emb_dim=q_t.size(1), hidden_dim=args.hidden).to(device)
    logger.info(f"Modelo creado: {sum(p.numel() for p in model.parameters()):,} parámetros")

    # Entrenar
    t0 = time.time()
    best_acc = train(
        model,
        q_t[ti], a_t[ti], b_t[ti], d_t[ti], w_t[ti],
        q_t[vi], a_t[vi], b_t[vi], d_t[vi], w_t[vi],
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        margin=args.margin,
        device=device,
    )

    # Guardar checkpoints
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar versión pointwise específica
    model.save(str(output_dir / "reward_model_pointwise.pt"))
    
    # Guardar versión principal (para carga automática)
    torch.save({
        'model_state': model.state_dict(),
        'config': {
            'emb_dim': model.emb_dim,
            'hidden_dim': model.hidden_dim,
            'model_type': 'pointwise'
        }
    }, output_dir / "reward_model.pt")
    
    logger.info(f"✅ Modelo guardado en {output_dir}/reward_model.pt")

    elapsed = time.time() - t0
    
    print("\n" + "=" * 60)
    print("  COMPLETADO")
    print("=" * 60)
    print(f"  Pares entrenamiento: {len(ti):,}")
    print(f"  Pares validación:    {len(vi):,}")
    print(f"  val_accuracy:        {best_acc:.3f}")
    print(f"  Tiempo:              {elapsed:.0f}s")
    
    if best_acc >= 0.75:
        print("\n  ✅ SUPERA PUNTO DE CONTROL (val_acc >= 0.75)")
        print("  Siguiente: python evaluate_esci_v2.py --quick --alpha 0.3")
    elif best_acc >= 0.70:
        print("\n  ⚠️  ACEPTABLE (val_acc >= 0.70)")
        print("  Puedes continuar, pero intenta mejorar con más épocas o datos")
    else:
        print(f"\n  ❌ BAJA ACCURACY ({best_acc:.3f} < 0.70)")
        print("  Acciones recomendadas:")
        print("    1. Aumentar --epochs (ej. 35-40)")
        print("    2. Reducir --lr (ej. 1e-4)")
        print("    3. Verificar calidad de los pares")
        print("    4. Aumentar tamaño del corpus ESCI")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()