# run_chroma_builder.py

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path de Python
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from src.core.data.chroma_builder import build_chroma_from_cli

if __name__ == "__main__":
    build_chroma_from_cli()