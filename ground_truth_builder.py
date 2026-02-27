"""
ground_truth_builder.py
=======================
Construye el ground truth binario desde data/raw.

Estrategia:
    Para cada query anotada manualmente (o semi-automática), guarda
    los product_ids que son relevantes.

Formato de salida:
    data/interactions/ground_truth_REAL.json
    {
        "query_texto": ["id1", "id2", ...],
        ...
    }

Modos de uso:
    1. Interactivo — el usuario confirma qué productos son relevantes
    2. Automático  — usa similitud coseno >= umbral como proxy de relevancia
       (válido para baseline, pero las preferencias A/B son la señal real)

Uso:
    python ground_truth_builder.py --mode auto --threshold 0.6
    python ground_truth_builder.py --mode interactive
    python ground_truth_builder.py --mode auto --queries queries.txt
"""

import argparse
import json
import logging
import sys
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

GT_PATH      = Path("data/interactions/ground_truth_REAL.json")
QUERIES_PATH = Path("data/interactions/queries.txt")


# ---------------------------------------------------------------------------
# Carga del sistema
# ---------------------------------------------------------------------------

def load_system():
    from src.unified_system_v2 import UnifiedSystemV2
    system = UnifiedSystemV2.load_from_cache()
    if system is None:
        logger.error("Sistema no encontrado. Ejecuta: python main.py init")
        sys.exit(1)
    logger.info(f"Sistema: {len(system.canonical_products):,} productos")
    return system


def load_queries(path: Path) -> list:
    """
    Carga queries desde un archivo de texto (una por línea)
    o las extrae de las preferencias A/B ya recolectadas.
    """
    if path.exists():
        queries = [l.strip() for l in path.read_text(encoding='utf-8').splitlines()
                   if l.strip()]
        logger.info(f"Queries desde {path}: {len(queries)}")
        return queries

    # Fallback: extraer desde preferencias recolectadas
    prefs_file = Path("data/preferences/preferences.jsonl")
    if prefs_file.exists():
        queries = set()
        with open(prefs_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        q = r.get('query', '').strip()
                        if q:
                            queries.add(q)
                    except json.JSONDecodeError:
                        continue
        queries = sorted(queries)
        logger.info(f"Queries extraídas de preferencias A/B: {len(queries)}")
        return queries

    logger.error(
        f"No se encontraron queries en {path} ni en preferencias A/B.\n"
        "Crea data/interactions/queries.txt con una query por línea."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Modo automático — similitud coseno como proxy de relevancia
# ---------------------------------------------------------------------------

def build_auto(system, queries: list, threshold: float, k_pool: int) -> dict:
    """
    Para cada query, recupera los top-k_pool productos y marca como
    relevantes los que tienen similitud coseno >= threshold.

    Este método es un proxy, no verdad absoluta.
    Para paper: reportar que el GT fue construido con umbral=threshold.
    """
    emb_model = system.canonicalizer.embedding_model
    gt = {}

    for i, query in enumerate(queries, 1):
        logger.info(f"  [{i}/{len(queries)}] '{query[:50]}'")
        q_emb = emb_model.encode(query, normalize_embeddings=True)
        candidates = system.vector_store.search(q_emb, k=k_pool)
        if not candidates:
            continue

        relevant = []
        for p in candidates:
            pid  = getattr(p, 'id', None) or getattr(p, 'product_id', None)
            emb  = getattr(p, 'content_embedding', None)
            if pid is None:
                continue
            if emb is not None:
                emb = np.array(emb)
                sim = float(np.dot(q_emb, emb) /
                            (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-8))
                if sim >= threshold:
                    relevant.append(pid)
            else:
                # Sin embedding: incluir los top-3 como relevantes por defecto
                if len(relevant) < 3:
                    relevant.append(pid)

        if relevant:
            gt[query] = relevant

    logger.info(f"Ground truth automático: {len(gt)} queries con relevantes")
    return gt


# ---------------------------------------------------------------------------
# Modo interactivo — el usuario confirma relevancia
# ---------------------------------------------------------------------------

def build_interactive(system, queries: list, k_show: int = 10) -> dict:
    """
    Muestra los top-k resultados para cada query y pide al usuario
    que confirme cuáles son relevantes.
    """
    emb_model = system.canonicalizer.embedding_model
    gt = {}

    print("\n" + "=" * 65)
    print("  CONSTRUCCIÓN INTERACTIVA DE GROUND TRUTH")
    print("  Para cada query, escribe los números de productos relevantes")
    print("  separados por coma (ej: 1,3,5) o 'todos' o 'ninguno'")
    print("=" * 65)

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] Query: '{query}'")
        print("-" * 50)

        q_emb      = emb_model.encode(query, normalize_embeddings=True)
        candidates = system.vector_store.search(q_emb, k=k_show)
        if not candidates:
            print("  Sin resultados FAISS para esta query.")
            continue

        pids = []
        for j, p in enumerate(candidates, 1):
            pid    = getattr(p, 'id', '') or getattr(p, 'product_id', '')
            title  = getattr(p, 'title', 'Sin título')[:60]
            rating = getattr(p, 'rating', None)
            r_str  = f"*{float(rating):.1f}" if rating else "     "
            print(f"  {j:2d}. {r_str}  {title}")
            pids.append(pid)

        while True:
            try:
                resp = input(
                    "\n  Relevantes (ej: 1,3 / todos / ninguno / skip): "
                ).strip().lower()

                if resp == 'skip' or resp == 's':
                    break
                elif resp in ('todos', 'all', 't'):
                    gt[query] = pids
                    print(f"  ✓ {len(pids)} productos marcados como relevantes")
                    break
                elif resp in ('ninguno', 'none', 'n', ''):
                    print("  ✓ Ningún producto marcado (query sin relevantes)")
                    break
                else:
                    nums = [int(x.strip()) for x in resp.split(',')
                            if x.strip().isdigit()]
                    valid = [pids[n-1] for n in nums if 1 <= n <= len(pids)]
                    if valid:
                        gt[query] = valid
                        print(f"  ✓ {len(valid)} productos marcados como relevantes")
                        break
                    else:
                        print("  Número(s) inválidos. Intenta de nuevo.")
            except (ValueError, EOFError, KeyboardInterrupt):
                print("\n  Interrumpido.")
                return gt

    logger.info(f"Ground truth interactivo: {len(gt)} queries con relevantes")
    return gt


# ---------------------------------------------------------------------------
# Guardar y cargar
# ---------------------------------------------------------------------------

def save_gt(gt: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)
    logger.info(f"Ground truth guardado: {path} ({len(gt)} queries)")


def load_gt(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def print_stats(gt: dict):
    if not gt:
        print("  Ground truth vacío.")
        return
    n_rel = [len(v) for v in gt.values()]
    print(f"\n  ESTADÍSTICAS DEL GROUND TRUTH:")
    print(f"    Queries:              {len(gt)}")
    print(f"    Relevantes totales:   {sum(n_rel)}")
    print(f"    Media por query:      {sum(n_rel)/len(n_rel):.1f}")
    print(f"    Mín / Máx por query:  {min(n_rel)} / {max(n_rel)}")
    print(f"    Queries con ≥3 rel:   {sum(1 for n in n_rel if n >= 3)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Construir ground truth desde data/raw")
    parser.add_argument('--mode',      choices=['auto', 'interactive'], default='auto',
                        help="Modo de construcción (default: auto)")
    parser.add_argument('--threshold', type=float, default=0.55,
                        help="Umbral de similitud coseno para modo auto (default: 0.55)")
    parser.add_argument('--k-pool',    type=int,   default=30,
                        help="Candidatos FAISS a considerar por query (default: 30)")
    parser.add_argument('--k-show',    type=int,   default=10,
                        help="Productos a mostrar en modo interactivo (default: 10)")
    parser.add_argument('--queries',   type=str,   default=str(QUERIES_PATH),
                        help="Archivo con queries (una por línea)")
    parser.add_argument('--output',    type=str,   default=str(GT_PATH),
                        help="Ruta de salida del ground truth")
    parser.add_argument('--merge',     action='store_true',
                        help="Combinar con GT existente en lugar de sobreescribir")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  CONSTRUCCIÓN DE GROUND TRUTH")
    print("=" * 65)
    print(f"  Modo:      {args.mode}")
    if args.mode == 'auto':
        print(f"  Umbral:    {args.threshold}")
        print(f"  Pool:      {args.k_pool} candidatos por query")
    print(f"  Salida:    {args.output}")
    print()

    system  = load_system()
    queries = load_queries(Path(args.queries))

    if not queries:
        logger.error("Sin queries.")
        sys.exit(1)

    print(f"  Procesando {len(queries)} queries...\n")

    if args.mode == 'auto':
        gt_new = build_auto(system, queries, args.threshold, args.k_pool)
    else:
        gt_new = build_interactive(system, queries, args.k_show)

    # Merge con GT existente si se pide
    output_path = Path(args.output)
    if args.merge:
        gt_existing = load_gt(output_path)
        n_before    = len(gt_existing)
        gt_existing.update(gt_new)
        gt_new = gt_existing
        print(f"\n  Merge: {n_before} → {len(gt_new)} queries")

    save_gt(gt_new, output_path)
    print_stats(gt_new)

    print(f"\n  SIGUIENTE PASO:")
    print(f"    python split_queries.py")


if __name__ == "__main__":
    main()