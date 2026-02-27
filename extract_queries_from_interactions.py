"""
extract_queries_from_interactions.py
=====================================
Extrae queries únicas desde data/interactions/real_interactions.jsonl
y las guarda en data/interactions/queries.txt

Uso:
    python extract_queries_from_interactions.py
    python extract_queries_from_interactions.py --min-count 2   # solo queries repetidas 2+ veces
    python extract_queries_from_interactions.py --preview       # solo mostrar stats
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

INTERACTIONS_PATH = Path("data/interactions/real_interactions.jsonl")
QUERIES_PATH      = Path("data/interactions/queries.txt")


def extract_queries(path: Path, min_count: int = 1) -> list:
    counts = Counter()
    errors = 0

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                itype  = record.get('interaction_type', '')
                if itype != 'query':
                    continue
                query = (
                    record.get('context', {}).get('query', '')
                    or record.get('query', '')
                ).strip()
                if query:
                    counts[query] += 1
            except json.JSONDecodeError:
                errors += 1

    if errors:
        logger.warning(f"  {errors} líneas con JSON inválido (ignoradas)")

    queries = [q for q, c in counts.most_common() if c >= min_count]
    return queries, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-count', type=int, default=1,
                        help='Frecuencia mínima para incluir una query (default: 1)')
    parser.add_argument('--preview',   action='store_true',
                        help='Solo mostrar estadísticas, no guardar')
    parser.add_argument('--input',  type=str, default=str(INTERACTIONS_PATH))
    parser.add_argument('--output', type=str, default=str(QUERIES_PATH))
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        logger.error(f"No encontrado: {path}")
        return

    queries, counts = extract_queries(path, args.min_count)

    print(f"\n  QUERIES EXTRAÍDAS DE INTERACCIONES REALES")
    print(f"  {'='*50}")
    print(f"  Total líneas query:    {sum(counts.values())}")
    print(f"  Queries únicas:        {len(counts)}")
    print(f"  Con min_count≥{args.min_count}:       {len(queries)}")
    print(f"\n  Top 20 más frecuentes:")
    for q, c in counts.most_common(20):
        bar = '█' * min(c, 20)
        print(f"    {c:3d}x  {bar}  {q}")

    if len(queries) > 20:
        print(f"\n  ... y {len(queries)-20} queries más")

    if args.preview:
        print("\n  (--preview: no se guardó)")
        return

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(queries) + '\n', encoding='utf-8')

    print(f"\n  Guardado: {output} ({len(queries)} queries)")
    print(f"\n  SIGUIENTE PASO:")
    print(f"    python ground_truth_builder.py --mode auto --threshold 0.55")


if __name__ == "__main__":
    main()