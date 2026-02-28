"""
inspect_ground_truth.py
=======================
Muestra qué queries tienen relevantes y cuáles quedaron fuera.
Útil para decidir si bajar el umbral o descartar queries irrelevantes.

Uso:
    python inspect_ground_truth.py
    python inspect_ground_truth.py --show-products   # muestra títulos de productos relevantes
"""

import argparse
import json
import sys
from pathlib import Path

GT_PATH      = Path("data/interactions/ground_truth_REAL.json")
QUERIES_PATH = Path("data/interactions/queries.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-products', action='store_true',
                        help='Mostrar títulos de los productos relevantes')
    args = parser.parse_args()

    # Cargar GT
    if not GT_PATH.exists():
        print("No encontrado: ground_truth_REAL.json")
        print("Ejecuta: python ground_truth_builder.py")
        sys.exit(1)

    with open(GT_PATH, encoding='utf-8') as f:
        gt = json.load(f)

    # Cargar todas las queries
    all_queries = []
    if QUERIES_PATH.exists():
        all_queries = [l.strip() for l in QUERIES_PATH.read_text(encoding='utf-8').splitlines() if l.strip()]

    queries_sin_rel = [q for q in all_queries if q not in gt]
    queries_con_rel = sorted(gt.keys(), key=lambda q: len(gt[q]), reverse=True)

    print(f"\n{'='*60}")
    print(f"  INSPECCIÓN DEL GROUND TRUTH")
    print(f"{'='*60}")
    print(f"  Total queries en queries.txt:  {len(all_queries)}")
    print(f"  Con relevantes:                {len(gt)}")
    print(f"  Sin relevantes (descartadas):  {len(queries_sin_rel)}")

    # Queries con relevantes
    print(f"\n  QUERIES CON RELEVANTES (ordenadas por cantidad):")
    print(f"  {'Query':<35} {'#rel':>5}")
    print(f"  {'-'*42}")
    for q in queries_con_rel:
        n = len(gt[q])
        bar = '█' * min(n, 30)
        print(f"  {q:<35} {n:>3}  {bar}")

    # Queries sin relevantes
    if queries_sin_rel:
        print(f"\n  QUERIES SIN RELEVANTES (umbral muy alto o no en catálogo):")
        for q in queries_sin_rel:
            print(f"    ✗  {q}")
        print(f"\n  → Para recuperarlas: python ground_truth_builder.py --threshold 0.40")
        print(f"  → O ignóralas si esas categorías no están en tu catálogo")

    # Productos relevantes si se pide
    if args.show_products:
        try:
            from src.unified_system_v2 import UnifiedSystemV2
            system = UnifiedSystemV2.load_from_cache()
            if system:
                id_map = {
                    str(getattr(p, 'id', '') or getattr(p, 'product_id', '')): p
                    for p in system.canonical_products
                }
                print(f"\n  PRODUCTOS RELEVANTES POR QUERY:")
                for q in queries_con_rel[:10]:  # top 10
                    print(f"\n  '{q}' ({len(gt[q])} relevantes):")
                    for pid in gt[q][:5]:
                        p = id_map.get(pid)
                        title = getattr(p, 'title', pid)[:60] if p else pid
                        print(f"    - {title}")
                    if len(gt[q]) > 5:
                        print(f"    ... y {len(gt[q])-5} más")
        except Exception as e:
            print(f"\n  (no se pudieron cargar títulos: {e})")

    # Recomendación para split
    n_con = len(gt)
    n_test = max(1, int(n_con * 0.30))
    n_train = n_con - n_test
    print(f"\n  SPLIT PROYECTADO (70/30):")
    print(f"    Train: ~{n_train} queries")
    print(f"    Test:  ~{n_test} queries")
    if n_test < 5:
        print(f"    [WARN] Pocas queries de test — considera bajar el umbral")
        print(f"           python ground_truth_builder.py --threshold 0.40")
    else:
        print(f"    [OK] Suficiente para evaluación")
        print(f"\n  SIGUIENTE PASO:")
        print(f"    python split_queries.py")


if __name__ == "__main__":
    main()