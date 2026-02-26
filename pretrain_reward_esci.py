"""
pretrain_reward_esci_pointwise.py
==================================
FASE 1 — Pre-entrenamiento del Reward Model con Amazon ESCI (train split).
VERSIÓN CORREGIDA: Pointwise puro (no listwise)

OBJETIVO DE FASE 1:
    Que el reward model aprenda relevancia absoluta producto-query.
    Score = f(query_embedding, product_embedding)

ARQUITECTURA:
    - Modelo pointwise: [batch, 384] + [batch, 384] → [batch, 1]
    - Loss: Bradley-Terry ponderado con log-sigmoid
    - Sin rankings artificiales, sin TOP_K, sin 3D tensors

PUNTO DE CONTROL:
    1. val_accuracy en pares ESCI > 0.70
    2. nDCG@10 reward-only > nDCG@10 FAISS baseline (0.317)
       -> medido en evaluate_esci.py

Uso:
    python pretrain_reward_esci_pointwise.py
"""
import json
import logging
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -- Configuración Fase 1 ----------------------------------------------
LABEL_TO_SCORE = {"Exact": 3, "Substitute": 2, "Complement": 1, "Irrelevant": 0}
PRETRAIN_EPOCHS     = 20
PRETRAIN_LR         = 2e-4
BATCH_SIZE          = 64
MAX_PAIRS_PER_QUERY = 20
MIN_SCORE_DIFF      = 1
VAL_SPLIT           = 0.1

# Peso por score_diff: pares más informativos pesan más
DIFF_WEIGHT = {1: 1.0, 2: 2.0, 3: 3.0}
# ---------------------------------------------------------------------

class PairDataset(Dataset):
    """Dataset para pares pointwise (sin rankings artificiales)"""
    def __init__(self, q_t: torch.Tensor, p_c_t: torch.Tensor, 
                 p_r_t: torch.Tensor, weights: torch.Tensor):
        self.q_t = q_t
        self.p_c_t = p_c_t
        self.p_r_t = p_r_t
        self.weights = weights
        
    def __len__(self):
        return len(self.weights)
    
    def __getitem__(self, idx):
        return (self.q_t[idx], self.p_c_t[idx], 
                self.p_r_t[idx], self.weights[idx])

def load_system_and_pipeline():
    """Carga sistema y pipeline RLHF"""
    if not Path("data/cache/unified_system_v2.pkl").exists():
        logger.error("Sistema no encontrado. Ejecuta: python main.py init")
        sys.exit(1)
    
    try:
        from src.unified_system_v2 import UnifiedSystemV2
        system = UnifiedSystemV2.load_from_cache()
        logger.info(f"Sistema: {len(system.canonical_products):,} productos")
        
        from src.rlhf_integration import add_rlhf_to_system
        pipeline = add_rlhf_to_system(system)
        
        # 🔥 IMPORTANTE: Verificar que el modelo es pointwise
        logger.info(f"Modelo type: {type(pipeline.reward_model).__name__}")
        
        return system, pipeline
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

def get_corpus_asins(system) -> set:
    """Extrae ASINs del corpus"""
    asins = set()
    for p in getattr(system, 'canonical_products', []):
        pid = getattr(p, 'id', None) or getattr(p, 'product_id', None)
        if pid:
            asins.add(str(pid))
    logger.info(f"ASINs en corpus: {len(asins):,}")
    return asins

def load_esci_train() -> "pd.DataFrame":
    """Carga ESCI train desde split limpio"""
    import pandas as pd
    train_path = Path("data/esci/esci_train.parquet")
    
    if train_path.exists():
        logger.info(f"Cargando ESCI train desde: {train_path}")
        df = pd.read_parquet(train_path)
        logger.info(f"  ESCI train: {len(df):,} filas, {df['query'].nunique():,} queries")
        return df
    
    # Fallback: descargar
    logger.warning("esci_train.parquet no encontrado. Ejecuta verify_and_split_esci.py")
    try:
        from datasets import load_dataset
        ds = load_dataset("tasksource/esci")
        df = ds['train'].to_pandas()
        logger.info(f"Descargado ESCI train: {len(df):,} filas")
        return df
    except Exception as e:
        logger.error(f"No se pudo descargar ESCI: {e}")
        sys.exit(1)

def filter_and_score_esci(df, corpus_asins: set) -> "pd.DataFrame":
    """Filtra ESCI a locale 'us' y ASINs en corpus"""
    import pandas as pd
    
    # Normalizar columna ASIN
    if 'product_id' in df.columns and 'asin' not in df.columns:
        df = df.rename(columns={'product_id': 'asin'})
    
    # Filtrar locale
    for lc in ['product_locale', 'locale']:
        if lc in df.columns:
            n = len(df)
            df = df[df[lc] == 'us'].copy()
            logger.info(f"Locale 'us': {n:,} -> {len(df):,}")
            break
    
    # Score numérico
    df = df.copy()
    df['score'] = df['esci_label'].map(LABEL_TO_SCORE)
    df = df[df['score'].notna()].copy()
    df['score'] = df['score'].astype(int)
    
    # Intersección con corpus
    n = len(df)
    df = df[df['asin'].isin(corpus_asins)].copy()
    logger.info(f"Intersección corpus: {n:,} -> {len(df):,} ({df['asin'].nunique():,} ASINs únicos)")
    
    if len(df) == 0:
        logger.error("Intersección vacía. Verifica que los ASINs coincidan.")
        sys.exit(1)
    
    return df

def generate_pairs(df) -> Tuple[List[dict], dict]:
    """
    Genera pares (chosen > rejected) desde etiquetas ESCI.
    VERSIÓN POINTWISE: Solo guarda los ASINs, no construye rankings.
    """
    random.seed(42)
    pairs = []
    skipped_queries = 0
    pair_stats = defaultdict(int)
    
    for query, gdf in df.groupby('query'):
        # Agrupar por score
        by_score = defaultdict(list)
        for _, row in gdf.iterrows():
            by_score[row['score']].append(row['asin'])
        
        query_pairs = []
        scores = sorted(by_score.keys(), reverse=True)
        
        # Generar pares con diff significativo
        for i, s_high in enumerate(scores):
            for s_low in scores[i+1:]:
                diff = s_high - s_low
                if diff < MIN_SCORE_DIFF:
                    continue
                
                for chosen in by_score[s_high]:
                    for rejected in by_score[s_low]:
                        query_pairs.append({
                            'query': str(query),
                            'chosen_asin': chosen,
                            'rejected_asin': rejected,
                            'chosen_score': int(s_high),
                            'rejected_score': int(s_low),
                            'score_diff': int(diff),
                            'weight': DIFF_WEIGHT.get(diff, 1.0),
                        })
                        pair_stats[f'diff_{diff}'] += 1
        
        if not query_pairs:
            skipped_queries += 1
            continue
        
        # Priorizar pares con mayor diff
        query_pairs.sort(key=lambda x: x['score_diff'], reverse=True)
        pairs.extend(query_pairs[:MAX_PAIRS_PER_QUERY])
    
    # Estadísticas
    total = len(pairs)
    logger.info(f"\nPares generados: {total:,} ({skipped_queries} queries sin pares)")
    for diff in [3, 2, 1]:
        count = pair_stats.get(f'diff_{diff}', 0)
        pct = count / total * 100 if total > 0 else 0
        logger.info(f"  diff={diff}: {count:>6,} ({pct:>5.1f}%)")
    
    stats = {
        'total_pairs': total,
        'skipped_queries': skipped_queries,
        'queries_with_pairs': df['query'].nunique() - skipped_queries,
        'diff_distribution': {f'diff_{d}': pair_stats.get(f'diff_{d}', 0) for d in [1, 2, 3]},
    }
    
    return pairs, stats

def build_pointwise_tensors(
    pairs: List[dict],
    product_index: Dict[str, np.ndarray],
    emb_model,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    🔥 CONSTRUCCIÓN POINTWISE PURA 🔥
    
    NO construye rankings artificiales.
    NO usa TOP_K.
    NO usa relleno.
    NO usa tensores 3D.
    
    Returns:
        q_t:      [N, emb_dim]  - embeddings de queries
        p_c_t:    [N, emb_dim]  - embeddings de productos elegidos
        p_r_t:    [N, emb_dim]  - embeddings de productos rechazados
        weights:  [N]           - pesos por par
    """
    logger.info(f"Construyendo tensores pointwise para {len(pairs):,} pares...")
    
    # Cache para embeddings de queries
    query_cache: Dict[str, np.ndarray] = {}
    
    q_list, p_c_list, p_r_list, weight_list = [], [], [], []
    skipped = 0
    
    for pair in pairs:
        chosen = pair['chosen_asin']
        rejected = pair['rejected_asin']
        
        # Verificar que ambos productos existen
        if chosen not in product_index or rejected not in product_index:
            skipped += 1
            continue
        
        # Query embedding (con cache)
        q = pair['query']
        if q not in query_cache:
            query_cache[q] = emb_model.encode(q, normalize_embeddings=True)
        
        # ✅ Pointwise puro: solo los embeddings de productos
        q_list.append(query_cache[q])
        p_c_list.append(product_index[chosen])
        p_r_list.append(product_index[rejected])
        weight_list.append(pair['weight'])
    
    if skipped:
        logger.warning(f"  Pares omitidos (embedding faltante): {skipped:,}")
    
    n = len(q_list)
    logger.info(f"  Tensores pointwise listos: {n:,} pares válidos")
    
    # Convertir a tensores
    q_t   = torch.tensor(np.array(q_list),   dtype=torch.float32, device=device)
    p_c_t = torch.tensor(np.array(p_c_list), dtype=torch.float32, device=device)
    p_r_t = torch.tensor(np.array(p_r_list), dtype=torch.float32, device=device)
    w_t   = torch.tensor(np.array(weight_list), dtype=torch.float32, device=device)
    
    return q_t, p_c_t, p_r_t, w_t

def reset_reward_weights(model):
    """Reinicia pesos para entrenamiento limpio"""
    from torch.nn import init
    
    for m in model.modules():
        if hasattr(m, 'weight') and m.weight is not None:
            if m.weight.dim() >= 2:
                init.xavier_uniform_(m.weight, gain=0.1)
            else:
                init.uniform_(m.weight, -0.01, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            init.zeros_(m.bias)
    
    logger.info("Pesos del reward model reiniciados (entrenamiento limpio)")

def pointwise_bt_loss(
    model,
    q: torch.Tensor,
    p_c: torch.Tensor,
    p_r: torch.Tensor,
    weights: torch.Tensor
) -> torch.Tensor:
    """
    🔥 BRADLEY-TERRY POINTWISE 🔥
    
    Loss = -Σ w_i * log σ( score(chosen) - score(rejected) )
    
    Args:
        q:      [batch, emb_dim] - embeddings de queries
        p_c:    [batch, emb_dim] - embeddings de productos elegidos
        p_r:    [batch, emb_dim] - embeddings de productos rechazados
        weights:[batch]           - pesos por par
    """
    model.train()
    
    # Scores pointwise
    score_c = model(q, p_c).squeeze(-1)  # [batch]
    score_r = model(q, p_r).squeeze(-1)  # [batch]
    
    # Diferencia de scores
    diff = score_c - score_r
    
    # Normalizar pesos del batch
    w_norm = weights / (weights.sum() + 1e-8)
    
    # Loss ponderada
    loss = -(w_norm * F.logsigmoid(diff)).sum()
    
    return loss

def pretrain_pointwise(
    pipeline,
    q_t: torch.Tensor,
    p_c_t: torch.Tensor,
    p_r_t: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[dict, float]:
    """
    Entrenamiento pointwise con split train/val
    """
    n = len(weights)
    idx = list(range(n))
    random.shuffle(idx)
    
    # Split train/val
    val_n = max(1, int(n * VAL_SPLIT))
    train_idx = idx[val_n:]
    val_idx   = idx[:val_n]
    
    # Crear datasets
    train_dataset = PairDataset(
        q_t[train_idx], p_c_t[train_idx], p_r_t[train_idx], weights[train_idx]
    )
    val_dataset = PairDataset(
        q_t[val_idx], p_c_t[val_idx], p_r_t[val_idx], weights[val_idx]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"Split — Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")
    
    # Configurar optimizer
    optimizer = torch.optim.AdamW(pipeline.reward_model.parameters(), lr=PRETRAIN_LR)
    
    history = {
        'train_loss': [],
        'val_acc': [],
        'val_loss': [],
    }
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        # --- Training ---
        pipeline.reward_model.train()
        epoch_losses = []
        
        for batch_q, batch_pc, batch_pr, batch_w in train_loader:
            loss = pointwise_bt_loss(
                pipeline.reward_model, batch_q, batch_pc, batch_pr, batch_w
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipeline.reward_model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = float(np.mean(epoch_losses))
        
        # --- Validation ---
        val_acc, val_loss = validate_pointwise(pipeline.reward_model, val_loader)
        
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        logger.info(
            f"Epoch {epoch:2d}/{PRETRAIN_EPOCHS} | "
            f"loss={avg_loss:.4f} | val_acc={val_acc:.3f} | val_loss={val_loss:.4f}"
        )
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in pipeline.reward_model.state_dict().items()}
    
    # Restaurar mejor estado
    if best_state:
        pipeline.reward_model.load_state_dict(best_state)
        logger.info(f"✅ Mejor estado restaurado — val_accuracy={best_val_acc:.3f}")
    
    return history, best_val_acc

def validate_pointwise(model, val_loader) -> Tuple[float, float]:
    """Validación: accuracy y loss"""
    model.eval()
    correct = 0
    total = 0
    losses = []
    
    with torch.no_grad():
        for batch_q, batch_pc, batch_pr, batch_w in val_loader:
            score_c = model(batch_q, batch_pc).squeeze(-1)
            score_r = model(batch_q, batch_pr).squeeze(-1)
            
            # Accuracy: ¿score_chosen > score_rejected?
            correct += (score_c > score_r).sum().item()
            total += len(score_c)
            
            # Loss de validación
            diff = score_c - score_r
            w_norm = batch_w / (batch_w.sum() + 1e-8)
            loss = -(w_norm * F.logsigmoid(diff)).sum()
            losses.append(loss.item())
    
    accuracy = correct / total if total > 0 else 0.0
    val_loss = float(np.mean(losses)) if losses else 0.0
    
    return accuracy, val_loss

def diagnose_pointwise_reward(model, q_t, p_c_t, p_r_t, pairs, weights, device):
    """
    Diagnóstico post-entrenamiento para modelo pointwise
    """
    print("\n" + "-" * 60)
    print("  DIAGNÓSTICO POST-ENTRENAMIENTO (POINTWISE)")
    print("-" * 60)
    
    model.eval()
    n = len(weights)
    
    with torch.no_grad():
        score_c = model(q_t, p_c_t).squeeze(-1).cpu().numpy()
        score_r = model(q_t, p_r_t).squeeze(-1).cpu().numpy()
        diffs = score_c - score_r
    
    # Accuracy global
    correct = (diffs > 0).mean()
    print(f"  Accuracy global: {correct:.3f} (n={n})")
    
    # Accuracy por nivel de diff
    for target_diff in [3, 2, 1]:
        mask = [i for i, p in enumerate(pairs[:n]) if p['score_diff'] == target_diff]
        if mask:
            acc_d = (diffs[mask] > 0).mean()
            n_d = len(mask)
            label = {3: "Exact>Irrelevant", 2: "diff=2", 1: "diff=1"}[target_diff]
            print(f"  Accuracy {label:<18}: {acc_d:.3f}  (n={n_d:>4})")
    
    # Análisis de scores
    print(f"\n  Estadísticas de scores:")
    print(f"    Chosen mean:  {score_c.mean():.4f} ± {score_c.std():.4f}")
    print(f"    Rejected mean: {score_r.mean():.4f} ± {score_r.std():.4f}")
    print(f"    Gap mean:      {diffs.mean():.4f}")
    
    # Detectar colapso
    if diffs.std() < 0.01:
        print("  ⚠️  POSIBLE COLAPSO: diferencias muy pequeñas")
    else:
        print("  ✅ Sin colapso detectable")
    
    print("-" * 60)

def verify_reward_model_compatibility(pipeline):
    """
    Verifica que el reward model sea compatible con uso pointwise
    """
    model = pipeline.reward_model
    device = pipeline.device
    
    # Crear tensores de prueba
    q_test = torch.randn(2, 384, device=device)
    p_test = torch.randn(2, 384, device=device)
    
    try:
        # Test forward pointwise
        with torch.no_grad():
            scores = model(q_test, p_test)
        
        logger.info(f"✅ Modelo compatible pointwise")
        logger.info(f"   Input shapes: q={q_test.shape}, p={p_test.shape}")
        logger.info(f"   Output shape: {scores.shape}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Modelo NO compatible pointwise: {e}")
        logger.error("   El modelo debe aceptar (query_emb, product_emb)")
        return False

def main():
    print("\n" + "=" * 70)
    print("  FASE 1 — PRE-ENTRENAMIENTO REWARD MODEL (POINTWISE)")
    print("  Señal: Amazon ESCI train (relevancia objetiva)")
    print("=" * 70)
    
    # 1. Cargar sistema
    system, pipeline = load_system_and_pipeline()
    corpus_asins = get_corpus_asins(system)
    
    # 2. Verificar compatibilidad pointwise
    if not verify_reward_model_compatibility(pipeline):
        sys.exit(1)
    
    # 3. Cargar ESCI train
    df = load_esci_train()
    df = filter_and_score_esci(df, corpus_asins)
    
    # 4. Generar pares
    pairs, pair_stats = generate_pairs(df)
    
    if len(pairs) < 100:
        logger.warning(f"⚠️  Pocos pares ({len(pairs)}). Intersección pequeña.")
    
    # 5. Guardar pares para auditoría
    Path("data/esci").mkdir(parents=True, exist_ok=True)
    with open("data/esci/pretrain_pairs_pointwise.jsonl", 'w') as f:
        for p in pairs:
            f.write(json.dumps(p) + '\n')
    logger.info(f"Pares guardados: data/esci/pretrain_pairs_pointwise.jsonl")
    
    # 6. Cargar embeddings
    cache = Path("data/cache/product_embeddings.npz")
    if not cache.exists():
        logger.error(f"Embeddings no encontrados: {cache}")
        sys.exit(1)
    
    data = np.load(cache, allow_pickle=True)
    product_index = {str(pid): emb for pid, emb in zip(data['ids'], data['embeddings'])}
    logger.info(f"Embeddings cargados: {len(product_index):,}")
    
    # 7. 🔥 CONSTRUIR TENSORES POINTWISE (sin rankings) 🔥
    q_t, p_c_t, p_r_t, weights = build_pointwise_tensors(
        pairs, product_index, pipeline.emb_model, pipeline.device
    )
    
    # 8. Reset y entrenar
    reset_reward_weights(pipeline.reward_model)
    history, best_val_acc = pretrain_pointwise(
        pipeline, q_t, p_c_t, p_r_t, weights
    )
    
    # 9. Diagnóstico
    diagnose_pointwise_reward(
        pipeline.reward_model, q_t, p_c_t, p_r_t, 
        pairs[:len(weights)], weights, pipeline.device
    )
    
    # 10. Guardar checkpoints
    ckpt_dir = Path("data/rlhf_checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    
    esci_ckpt = ckpt_dir / "reward_model_esci_pretrained_pointwise.pt"
    main_ckpt = ckpt_dir / "reward_model.pt"
    
    torch.save(pipeline.reward_model.state_dict(), esci_ckpt)
    torch.save(pipeline.reward_model.state_dict(), main_ckpt)
    
    logger.info(f"✅ Checkpoint guardado: {esci_ckpt}")
    logger.info(f"✅ Modelo actualizado: {main_ckpt}")
    
    pipeline.reward_trained = True
    
    # 11. Guardar estadísticas
    stats = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'phase1_esci_pretrain_pointwise',
        'n_pairs_generated': pair_stats['total_pairs'],
        'n_pairs_used': len(weights),
        'n_queries': pair_stats['queries_with_pairs'],
        'diff_distribution': pair_stats['diff_distribution'],
        'best_val_accuracy': best_val_acc,
        'epochs': PRETRAIN_EPOCHS,
        'lr': PRETRAIN_LR,
        'max_pairs_per_query': MAX_PAIRS_PER_QUERY,
        'history': history,
        'methodology': (
            f"Fase 1 Pointwise: {len(weights):,} pares con diff≥{MIN_SCORE_DIFF}. "
            f"Pesos proporcionales a diff. Bradley-Terry pointwise. "
            f"Best val_accuracy={best_val_acc:.3f}."
        ),
    }
    
    with open("data/esci/pretrain_stats_pointwise.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 12. Punto de control
    print("\n" + "=" * 70)
    print("  FASE 1 COMPLETADA — VERSIÓN POINTWISE")
    print("=" * 70)
    print(f"  Pares generados:      {pair_stats['total_pairs']:,}")
    print(f"  Pares usados:          {len(weights):,}")
    print(f"  Queries con pares:     {pair_stats['queries_with_pairs']:,}")
    print(f"  Best val accuracy:     {best_val_acc:.3f}")
    print(f"  Checkpoint:            {esci_ckpt}")
    
    print("\n" + "-" * 70)
    print("  PUNTO DE CONTROL FASE 1")
    print("-" * 70)
    
    if best_val_acc >= 0.70:
        print(f"  ✅ val_accuracy={best_val_acc:.3f} ≥ 0.70")
        print(f"\n  MEDICIÓN OBLIGATORIA:")
        print(f"    python build_esci_ground_truth.py")
        print(f"    python evaluate_esci.py")
        print(f"\n  Si nDCG@10 reward-only > 0.317 (baseline):")
        print(f"    ✅ Reward alineado. Continuar a Fase 2:")
        print(f"      python finetune_reward_ab.py")
    elif best_val_acc >= 0.60:
        print(f"  ⚠️  val_accuracy={best_val_acc:.3f} — marginal")
        print(f"  Considera aumentar PRETRAIN_EPOCHS o revisar intersección.")
    else:
        print(f"  ❌ val_accuracy={best_val_acc:.3f} < 0.60")
        print(f"  NO continuar a Fase 2.")
    
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()