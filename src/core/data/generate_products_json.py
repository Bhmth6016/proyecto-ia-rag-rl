import json
from pathlib import Path
from src.core.data.loader import DataLoader
from src.core.config import settings

def generate_products_json(output_file: str = "products.json") -> None:
    """
    Genera un archivo JSON unificado con todos los productos procesados.
    
    Args:
        output_file: Nombre del archivo de salida (por defecto: 'products.json')
    """
    # Configurar rutas
    output_path = Path(settings.PROC_DIR) / output_file
    
    # Cargar datos usando DataLoader
    loader = DataLoader()
    products = loader.load_data(use_cache=True)
    
    # Convertir productos a diccionarios
    products_data = [product.dict() for product in products]
    
    # Guardar como JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(products_data, f, ensure_ascii=False, indent=2)
    
    print(f"Archivo generado exitosamente: {output_path}")
    print(f"Total de productos procesados: {len(products_data)}")

if __name__ == "__main__":
    generate_products_json()