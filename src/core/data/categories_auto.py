import os
import re

DATA_DIR = "C:/Users/evill/OneDrive/Documentos/Github/proyecto-ia-rag-rl/data/processed"

_CATEGORY_KEYWORDS = {}
_TAG_KEYWORDS = {}
FILENAME_TO_CATEGORY = {}

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".pkl"):
        category_name = re.sub(r'\.pkl$', '', filename).replace('_', ' ').title()
        FILENAME_TO_CATEGORY[filename] = category_name

        # Generar keywords simples a partir del nombre
        words = category_name.lower().split()
        _CATEGORY_KEYWORDS[category_name] = words
        _TAG_KEYWORDS[category_name] = [w for w in words if len(w) > 3]  # Ejemplo: palabras >3 letras
