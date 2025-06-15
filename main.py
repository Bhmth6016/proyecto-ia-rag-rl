from src.data_loader import DataLoader
from src.ui_client.ui_interface import run_interface
from src.category_selector.category_tree import load_category_tree

def main():
    # Paso 1: Cargar los datos procesados (opcional si ya se generaron los archivos .pkl)
    loader = DataLoader()
    data = loader.load_data(use_cache=True)

    # Mostrar información básica sobre los datos cargados
    print(f"\nSe cargaron {len(data)} productos procesados")

    # Paso 2: Cargar árbol de categorías y filtros (esto internamente revisa los archivos .pkl)
    category_tree = load_category_tree()
    if not category_tree:
        print("No se encontraron categorías procesadas. Verifica que existan archivos .pkl en la carpeta 'data/processed'.")
        return

    # Paso 3: Ejecutar interfaz de usuario
    run_interface()

# Llamada directa a main()
import json  # Importación para el print de ejemplo
main()