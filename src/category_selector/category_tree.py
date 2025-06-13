import os
import pickle
import json
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path.home() / "OneDrive" / "Documentos" / "Github" / "proyecto-ia-rag-rl-data"
PROCESSED_DIR = BASE_DIR / "processed"
OUTPUT_FILE = BASE_DIR / "category_filters.json"

def extract_category_filters():
    if OUTPUT_FILE.exists():
        latest_pkl_time = max(f.stat().st_mtime for f in PROCESSED_DIR.glob("*.pkl"))
        if OUTPUT_FILE.stat().st_mtime >= latest_pkl_time:
            logger.info("Categorías y filtros ya existentes. No se requiere regeneración.")
            return

    category_map: Dict[str, List[Dict]] = {}

    for file in PROCESSED_DIR.glob("*.pkl"):
        try:
            with open(file, "rb") as f:
                data = pickle.load(f)

            category_name = file.stem.replace("meta_", "").replace("_processed", "").replace("_", " ").title()

            if category_name not in category_map:
                category_map[category_name] = []

            for item in data:
                category_map[category_name].append({
                    "title": item.get("title", ""),
                    "price": item.get("price"),
                    "average_rating": item.get("average_rating", 0.0),
                    "main_category": item.get("main_category", category_name)
                })
        except Exception as e:
            logger.warning(f"Error leyendo {file.name}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(category_map, f, indent=2, ensure_ascii=False)
    logger.info(f"Archivo de categorías y filtros actualizado: {OUTPUT_FILE}")
