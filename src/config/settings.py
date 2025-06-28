from pathlib import Path

# Directorios base
BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Archivos específicos
CATEGORY_FILTERS_FILE = PROCESSED_DATA_DIR / "category_filters.json"