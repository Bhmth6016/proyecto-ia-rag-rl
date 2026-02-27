"""
add_queries.py
==============
Registra queries manualmente en real_interactions.jsonl
con el mismo formato que usa el sistema.

Uso:
    python add_queries.py                          # modo interactivo
    python add_queries.py "laptop gaming" "smartwatch" "running shoes"
"""

import json
import sys
from datetime import datetime
from pathlib import Path
import uuid

INTERACTIONS_PATH = Path("data/interactions/real_interactions.jsonl")

QUERIES_SUGERIDAS = [
    # Tecnología
    "laptop gaming", "wireless headphones", "smartwatch", "tablet android",
    "phone case", "usb hub", "webcam",
    # Hogar
    "desk lamp", "coffee maker", "vacuum cleaner", "air purifier", "bed sheets",
    # Ropa
    "running shoes", "winter jacket", "casual sneakers", "sport socks",
    # Libros
    "fantasy books", "science fiction", "comic books", "mystery novels",
    # Juguetes
    "lego sets", "board games", "baby toys", "puzzle",
    # Deportes
    "yoga mat", "dumbbells", "bicycle", "swimming goggles",
    # Videojuegos (complementar los que ya tienes)
    "rpg games", "strategy games", "racing games", "ps5 games",
    "indie games", "multiplayer games",
]


def write_query(query: str):
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    record = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "interaction_type": "query",
        "context": {
            "query": query,
            "results_count": 20,
            "timestamp": datetime.now().isoformat()
        }
    }
    INTERACTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INTERACTIONS_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def modo_interactivo():
    print("\n" + "="*55)
    print("  AÑADIR QUERIES A real_interactions.jsonl")
    print("="*55)
    print("  Escribe queries una por línea.")
    print("  Enter vacío = terminar\n")

    print("  Sugerencias (copia las que quieras):")
    for i, q in enumerate(QUERIES_SUGERIDAS, 1):
        print(f"  {i:2d}. {q}")
    print()

    added = []
    while True:
        try:
            query = input("  Query: ").strip()
            if not query:
                break
            write_query(query)
            added.append(query)
            print(f"       ✓ guardada")
        except (KeyboardInterrupt, EOFError):
            break

    if added:
        print(f"\n  {len(added)} queries añadidas:")
        for q in added:
            print(f"    - {q}")
        print(f"\n  Archivo: {INTERACTIONS_PATH}")
        print(f"\n  Siguiente paso:")
        print(f"    python extract_queries_from_interactions.py")
    else:
        print("\n  Sin cambios.")


def modo_args(queries: list):
    for q in queries:
        write_query(q)
        print(f"  ✓ {q}")
    print(f"\n  {len(queries)} queries añadidas a {INTERACTIONS_PATH}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        modo_args(args)
    else:
        modo_interactivo()