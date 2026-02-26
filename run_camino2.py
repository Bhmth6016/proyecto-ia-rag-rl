# -*- coding: utf-8 -*-
import os; os.environ.setdefault("PYTHONUTF8", "1")
"""
run_camino2.py
==============
Orquestador completo del Camino 2.

ORDEN DE EJECUCIÓN:
    1. build_esci_corpus.py       — corpus nuevo desde ESCI train
    2. init_esci_system.py        — reemplazar corpus en el sistema
    3. pretrain_reward_esci.py    — reward con señal real (miles de pares)
    4. build_esci_ground_truth_v2.py — ground truth con intersección ~100%
    5. evaluate_esci_v2.py        — MEDICIÓN: punto de control Fase 1
    6. [Si Fase 1 OK] -> fusion_ranker.py --grid-search

Uso:
    python run_camino2.py                    # flujo completo
    python run_camino2.py --size 50000       # corpus más pequeño (más rápido)
    python run_camino2.py --skip-corpus      # si ya tienes el corpus
    python run_camino2.py --quick-eval       # evaluación en 100 queries
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run(label: str, cmd: list, abort=True) -> bool:
    print(f"\n{'-'*65}")
    print(f"  PASO: {label}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'-'*65}")
    t0 = time.time()
    r  = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"\n  [ERR] FALLÓ ({elapsed:.0f}s)")
        if abort: sys.exit(r.returncode)
        return False
    print(f"\n  [OK] OK ({elapsed:.0f}s)")
    return True


def banner(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size',         type=int, default=100_000,
                        help='ASINs en el corpus nuevo (default: 100,000)')
    parser.add_argument('--skip-corpus',  action='store_true',
                        help='Omitir construcción e inicialización del corpus')
    parser.add_argument('--skip-train',   action='store_true',
                        help='Omitir pre-entrenamiento del reward')
    parser.add_argument('--quick-eval',   action='store_true',
                        help='Evaluación en primeras 100 queries')
    parser.add_argument('--no-fusion',    action='store_true',
                        help='Omitir grid search de fusión')
    args = parser.parse_args()

    py = sys.executable

    banner("CAMINO 2 — CORPUS ESCI + REWARD ALINEADO")
    print(f"""
  Corpus nuevo:    {args.size:,} ASINs de ESCI train
  Intersección:    ~100% (todo el corpus está en ESCI)
  Pares esperados: miles (vs 74 actuales)
  Backup:          automático del sistema original
""")

    t_total = time.time()

    # PASO 1: Construir corpus ESCI
    if not args.skip_corpus:
        banner("PASO 1/5 — CORPUS DESDE ESCI TRAIN")
        run("Construir corpus ESCI", [py, "build_esci_corpus.py", "--size", str(args.size)])

        banner("PASO 2/5 — INICIALIZAR SISTEMA CON CORPUS ESCI")
        run("Inicializar sistema", [py, "init_esci_system.py"])
    else:
        print("\n  [SKIP] Corpus — usando corpus existente")

    # PASO 2: Pre-entrenar reward (ahora con miles de pares)
    if not args.skip_train:
        banner("PASO 3/5 — PRE-ENTRENAR REWARD (señal real)")
        run("Pretrain reward pointwise con ESCI", [py, "train_pointwise_reward.py"])
    else:
        print("\n  [SKIP] Pre-entrenamiento del reward")

    # PASO 3: Ground truth v2
    banner("PASO 4/5 — GROUND TRUTH v2")
    run("Construir ground truth v2", [py, "build_esci_ground_truth_v2.py"])

    # PASO 4: Evaluación — punto de control
    banner("PASO 5/5 — EVALUACIÓN (punto de control Fase 1)")
    eval_cmd = [py, "evaluate_esci_v2.py"]
    if args.quick_eval:
        eval_cmd.append("--quick")
    run("Evaluación ESCI v2", eval_cmd, abort=False)

    # PASO 5: Grid search fusión (Opción A)
    if not args.no_fusion:
        banner("BONUS — SCORE FUSION (Opción A)")
        ok = run("Grid search α/β", [py, "fusion_ranker.py", "--grid-search"], abort=False)
        if ok:
            grid_path = Path("results/fusion_grid_search.json")
            if grid_path.exists():
                with open(grid_path) as f:
                    grid = json.load(f)
                a, b = grid['best_alpha'], grid['best_beta']
                run(f"Evaluar fusion α={a}/β={b}",
                    [py, "fusion_ranker.py", "--alpha", str(a), "--beta", str(b), "--evaluate"],
                    abort=False)

    # Resumen
    elapsed = (time.time() - t_total) / 60
    banner("RESUMEN CAMINO 2")

    # Leer manifest del corpus
    manifest_path = Path("data/esci_corpus/corpus_manifest.json")
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        proj = manifest.get('training_projection', {})
        print(f"\n  Corpus nuevo:")
        print(f"    Productos:           {manifest.get('n_products', '?'):,}")
        print(f"    Intersección ESCI:   ~100%")
        print(f"    Pares de training:   {proj.get('estimated_pairs', '?'):,}")
        print(f"    Queries con pares:   {proj.get('queries_with_pairs', '?'):,}")

    # Leer resultado de evaluación
    results_dir = Path("results")
    v2_results = sorted(results_dir.glob("esci_v2_evaluation_*.json")) if results_dir.exists() else []
    if v2_results:
        with open(v2_results[-1]) as f:
            ev = json.load(f)
        summary = ev.get('summary', {})
        bl  = summary.get('baseline', {}).get('ndcg@10_mean', 0)
        ro  = summary.get('reward_only', {}).get('ndcg@10_mean', 0)
        print(f"\n  Evaluación Fase 1:")
        print(f"    Baseline nDCG@10:    {bl:.4f}")
        print(f"    Reward-Only nDCG@10: {ro:.4f}")
        print(f"    Δ:                   {ro - bl:+.4f}")
        if ro > bl:
            print(f"\n  [OK] FASE 1 SUPERADA")
            print(f"    -> Continuar: python finetune_reward_ab.py")
        else:
            print(f"\n  [ERR] Reward aún no supera baseline.")
            print(f"    -> Revisar pretrain_reward_esci.py")

    # Leer grid search si existe
    grid_path = Path("results/fusion_grid_search.json")
    if grid_path.exists():
        with open(grid_path) as f:
            grid = json.load(f)
        print(f"\n  Score Fusion:")
        print(f"    Mejor α/β:           {grid['best_alpha']:.1f}/{grid['best_beta']:.1f}")
        print(f"    Fusion nDCG@10:      {grid['best_ndcg@10']:.4f}")
        print(f"    Mejora vs baseline:  {grid['improvement']:+.4f}")

    print(f"\n  Tiempo total: {elapsed:.1f} min")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()