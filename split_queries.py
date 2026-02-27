"""
split_queries.py
================
Divide las queries del ground truth en train (70%) y test (30%).

REGLA FUNDAMENTAL:
    Las test queries NUNCA se usan para entrenar reward ni PPO.
    El split se hace por query, no por producto ni por par A/B.

Salidas:
    data/interactions/train_queries.json   — queries para entrenar reward
    data/interactions/test_queries.json    — queries para evaluación final
    data/interactions/split_info.json      — metadatos del split

Uso:
    python split_queries.py
    python split_queries.py --test-size 0.25 --seed 42
    python split_queries.py --verify      # verifica split existente
"""

import argparse
import json
import logging
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

GT_PATH          = Path("data/interactions/ground_truth_REAL.json")
TRAIN_SPLIT_PATH = Path("data/interactions/train_queries.json")
TEST_SPLIT_PATH  = Path("data/interactions/test_queries.json")
SPLIT_INFO_PATH  = Path("data/interactions/split_info.json")


def load_gt(path: Path) -> dict:
    if not path.exists():
        logger.error(f"Ground truth no encontrado: {path}")
        logger.error("Ejecuta primero: python ground_truth_builder.py")
        sys.exit(1)
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def make_split(gt: dict, test_size: float, seed: int):
    """
    Split estratificado por número de relevantes.
    Asegura que train y test tengan distribución similar de
    queries fáciles (muchos relevantes) y difíciles (pocos relevantes).
    """
    queries = list(gt.keys())
    n       = len(queries)

    if n < 4:
        logger.error(f"Necesitas al menos 4 queries (tienes {n}).")
        sys.exit(1)

    rng = np.random.default_rng(seed)

    # Estratificar por número de relevantes: pocas, medias, muchas
    def stratum(q):
        nr = len(gt[q])
        if nr <= 2:   return 'low'
        elif nr <= 5: return 'mid'
        else:         return 'high'

    groups = {}
    for q in queries:
        s = stratum(q)
        groups.setdefault(s, []).append(q)

    train_queries, test_queries = [], []
    for s, qs in groups.items():
        rng.shuffle(qs)
        n_test = max(1, int(len(qs) * test_size))
        test_queries.extend(qs[:n_test])
        train_queries.extend(qs[n_test:])

    # Garantía: no overlap
    train_set = set(train_queries)
    test_set  = set(test_queries)
    assert len(train_set & test_set) == 0, "ERROR: overlap entre train y test"

    return sorted(train_queries), sorted(test_queries)


def verify_split():
    """Verifica que el split existente no tenga leakage."""
    if not TRAIN_SPLIT_PATH.exists() or not TEST_SPLIT_PATH.exists():
        logger.error("Split no encontrado. Ejecuta: python split_queries.py")
        sys.exit(1)

    with open(TRAIN_SPLIT_PATH) as f: train_q = set(json.load(f))
    with open(TEST_SPLIT_PATH)  as f: test_q  = set(json.load(f))

    overlap = train_q & test_q
    print(f"\n  Train queries: {len(train_q)}")
    print(f"  Test  queries: {len(test_q)}")
    print(f"  Overlap:       {len(overlap)}")

    if overlap:
        print(f"\n  [ERR] LEAKAGE DETECTADO. Queries compartidas:")
        for q in sorted(overlap)[:5]:
            print(f"    - {q}")
        sys.exit(1)
    else:
        print(f"\n  [OK] Sin leakage — split limpio")

    # Leer info del split
    if SPLIT_INFO_PATH.exists():
        with open(SPLIT_INFO_PATH) as f:
            info = json.load(f)
        print(f"\n  Split creado:  {info.get('timestamp', 'desconocido')}")
        print(f"  Seed:          {info.get('seed', 'desconocido')}")
        print(f"  Test fraction: {info.get('test_fraction', 'desconocido')}")


def print_split_stats(train_queries: list, test_queries: list, gt: dict):
    n_train_rel = sum(len(gt[q]) for q in train_queries)
    n_test_rel  = sum(len(gt[q]) for q in test_queries)

    print(f"\n  SPLIT COMPLETADO:")
    print(f"    Train queries:      {len(train_queries):3d}  "
          f"({len(train_queries)/(len(train_queries)+len(test_queries))*100:.0f}%)")
    print(f"    Test  queries:      {len(test_queries):3d}  "
          f"({len(test_queries)/(len(train_queries)+len(test_queries))*100:.0f}%)")
    print(f"    Relevantes train:   {n_train_rel}")
    print(f"    Relevantes test:    {n_test_rel}")

    if n_train_rel > 0:
        avg_train = n_train_rel / len(train_queries)
        avg_test  = n_test_rel  / len(test_queries)
        print(f"    Media rel/query train: {avg_train:.1f}")
        print(f"    Media rel/query test:  {avg_test:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-size', type=float, default=0.30,
                        help='Fracción para test (default: 0.30)')
    parser.add_argument('--seed',      type=int,   default=42,
                        help='Semilla aleatoria (default: 42) — NO cambiar después')
    parser.add_argument('--gt',        type=str,   default=str(GT_PATH),
                        help='Ruta al ground truth')
    parser.add_argument('--verify',    action='store_true',
                        help='Solo verificar split existente')
    parser.add_argument('--force',     action='store_true',
                        help='Sobreescribir split existente')
    args = parser.parse_args()

    if args.verify:
        verify_split()
        return

    print("\n" + "=" * 60)
    print("  SPLIT TRAIN / TEST DE QUERIES")
    print("=" * 60)
    print(f"  Test fraction: {args.test_size:.0%}")
    print(f"  Seed:          {args.seed}  (FIJO — no cambiar)")
    print()

    # Verificar que no existe ya (sin --force)
    if TRAIN_SPLIT_PATH.exists() and not args.force:
        print(f"  [WARN] Split ya existe. Usa --force para sobreescribir.")
        print(f"         O usa --verify para verificarlo.")
        verify_split()
        return

    gt = load_gt(Path(args.gt))

    if len(gt) == 0:
        logger.error("Ground truth vacío.")
        sys.exit(1)

    logger.info(f"Ground truth: {len(gt)} queries")

    train_queries, test_queries = make_split(gt, args.test_size, args.seed)

    # Guardar splits
    Path("data/interactions").mkdir(parents=True, exist_ok=True)

    with open(TRAIN_SPLIT_PATH, 'w', encoding='utf-8') as f:
        json.dump(train_queries, f, indent=2, ensure_ascii=False)

    with open(TEST_SPLIT_PATH, 'w', encoding='utf-8') as f:
        json.dump(test_queries, f, indent=2, ensure_ascii=False)

    split_info = {
        'timestamp':     datetime.now().isoformat(),
        'seed':          args.seed,
        'test_fraction': args.test_size,
        'n_train':       len(train_queries),
        'n_test':        len(test_queries),
        'gt_path':       args.gt,
        'leakage_free':  True,
        'note': (
            "Split estratificado por número de relevantes. "
            "Seed fijo = reproducible. "
            "Test queries NUNCA deben usarse en entrenamiento."
        )
    }
    with open(SPLIT_INFO_PATH, 'w') as f:
        json.dump(split_info, f, indent=2)

    print_split_stats(train_queries, test_queries, gt)

    print(f"\n  Archivos:")
    print(f"    {TRAIN_SPLIT_PATH}")
    print(f"    {TEST_SPLIT_PATH}")
    print(f"    {SPLIT_INFO_PATH}")

    print(f"\n  REGLA: las queries de test NO deben verse en entrenamiento.")
    print(f"\n  SIGUIENTE PASO:")
    print(f"    1. Entrenar reward:  python train_pointwise_reward.py")
    print(f"    2. Evaluar:          python evaluate_methods.py")


if __name__ == "__main__":
    main()