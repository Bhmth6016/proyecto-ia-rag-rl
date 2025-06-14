from src.data_loader import DataLoader
#from src.category_selector.category_tree import extract_category_filters
#from src.ui_client.ui_interface import run_user_interface

def main():
    # Paso 1: Cargar los datos procesados
    loader = DataLoader()
    data = loader.load_data(use_cache=True)
    
    # Opcional: Mostrar información básica sobre los datos cargados
    print(f"\nSe cargaron {len(data)} productos procesados")

    
    # Paso 2: Generar archivo de categorías si es necesario (descomentar cuando esté listo)
    # extract_category_filters(data)
    
    # Paso 3: Ejecutar interfaz de usuario (descomentar cuando esté listo)
    # run_user_interface(data)

if __name__ == "__main__":
    import json  # Importación añadida para el print de ejemplo
    main()