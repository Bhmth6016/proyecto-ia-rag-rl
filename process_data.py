from src.core.data.loader import DataLoader
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def main():
    loader = DataLoader(
        raw_dir=Path("data/raw"),
        processed_dir=Path("data/processed")
    )
    
    try:
        output_file = loader.process_all_jsonl()
        print(f"Proceso completado. Archivo generado: {output_file}")
    except Exception as e:
        logging.error(f"Error en el procesamiento: {str(e)}")
        raise

if __name__ == "__main__":
    main()