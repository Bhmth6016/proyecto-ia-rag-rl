from src.data_loader import DataLoader
from src.category_selector.category_tree import extract_category_filters
from src.ui_client.ui_interface import run_user_interface

def main():
    # Paso 1: Procesar y guardar data reducida si es necesario
    loader = DataLoader()
    loader.save_reduced_data("reduced_data.jsonl")

    # Paso 2: Generar archivo de categorías si es necesario
    extract_category_filters()

    # Paso 3: Ejecutar interfaz de usuario
    run_user_interface()

if __name__ == "__main__":
    main()
