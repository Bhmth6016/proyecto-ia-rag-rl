# reviews_from_products_10000.py
import json
import logging
from pathlib import Path
from typing import Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ================= CONFIGURACIÓN EXPLÍCITA =================
PRODUCT_ID_FIELD = "parent_asin"     # campo en meta_*.jsonl
REVIEW_ID_FIELD  = "parent_asin"     # campo en reviews/*.jsonl
# ===========================================================

def load_product_ids(raw_dir: Path) -> Set[str]:
    """
    Carga TODOS los IDs de productos desde meta_*_10000.jsonl
    """
    product_ids: Set[str] = set()
    product_files = sorted(raw_dir.glob("meta_*_10000.jsonl"))

    if not product_files:
        raise RuntimeError("No se encontraron archivos meta_*_10000.jsonl")

    for file in product_files:
        logger.info(f"📦 Leyendo productos: {file.name}")
        with open(file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    obj = json.loads(line)
                    pid = obj.get(PRODUCT_ID_FIELD)
                    if pid:
                        product_ids.add(str(pid))
                except json.JSONDecodeError:
                    logger.warning(f"Línea inválida {line_num} en {file.name}")

    logger.info(f"✅ Total IDs de productos cargados: {len(product_ids)}")
    return product_ids


def filter_reviews_file(
    reviews_file: Path,
    output_file: Path,
    valid_product_ids: Set[str]
):
    """
    Filtra un archivo de reviews manteniendo solo los que
    correspondan a productos muestreados
    """
    total = 0
    kept = 0

    with open(reviews_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, 1):
            total += 1
            try:
                review = json.loads(line)
                rid = review.get(REVIEW_ID_FIELD)

                if rid and str(rid) in valid_product_ids:
                    fout.write(json.dumps(review, ensure_ascii=False) + "\n")
                    kept += 1

            except json.JSONDecodeError:
                logger.warning(f"Línea inválida {line_num} en {reviews_file.name}")

    logger.info(f"   Reviews totales: {total}")
    logger.info(f"   Reviews conservadas: {kept}")
    return kept


def main():
    BASE_DIR = Path(__file__).parent.parent if "src" in str(Path(__file__)) else Path(__file__).parent

    RAW_DIR = BASE_DIR / "data" / "raw"
    REVIEWS_DIR = BASE_DIR / "data" / "reviews"

    if not RAW_DIR.exists():
        raise RuntimeError(f"No existe {RAW_DIR}")

    if not REVIEWS_DIR.exists():
        raise RuntimeError(f"No existe {REVIEWS_DIR}")

    logger.info("🔍 Cargando IDs de productos muestreados...")
    product_ids = load_product_ids(RAW_DIR)

    review_files = sorted(
        f for f in REVIEWS_DIR.glob("*.jsonl")
        if not f.stem.endswith("_10000")
    )
    if not review_files:
        raise RuntimeError("No se encontraron archivos de reviews")

    for review_file in review_files:
        output_file = REVIEWS_DIR / f"{review_file.stem}_10000.jsonl"

        if output_file.exists():
            logger.info(f"⏭️  {output_file.name} ya existe, omitiendo")
            continue

        logger.info(f"\n📝 Filtrando reviews: {review_file.name}")
        kept = filter_reviews_file(
            reviews_file=review_file,
            output_file=output_file,
            valid_product_ids=product_ids
        )

        logger.info(f"✅ Guardado: {output_file.name} ({kept} reviews)")

    logger.info("\n🎉 Limpieza de reviews completada correctamente")


if __name__ == "__main__":
    main()
