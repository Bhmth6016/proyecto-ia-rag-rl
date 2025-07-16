from src.core.utils.indexador import convertir_jsonl_a_json, indexar_productos

def main():
    # Configura tus rutas aquí
    jsonl_path = "data/raw/meta_Baby_Products.jsonl"
    json_path = "data/processed/productos.json"
    
    print("🔍 Procesando archivo JSONL...")
    convertir_jsonl_a_json(jsonl_path, json_path)
    
    print("🛠️ Indexando productos...")
    indexar_productos(json_path=json_path)

if __name__ == "__main__":
    main()