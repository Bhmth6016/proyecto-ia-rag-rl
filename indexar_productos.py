from src.utils.indexador import convertir_jsonl_a_json, indexar_productos
from langchain_community.embeddings import HuggingFaceEmbeddings



jsonl_path = r"C:\Users\x\Downloads\rag\data\productosndjson\meta_Baby_Products.jsonl"
json_path = r"C:\Users\x\Downloads\rag\data\procesados\productos.json"

print("🔍 Verificando archivo JSONL...")
convertir_jsonl_a_json(jsonl_path, json_path)
indexar_productos(json_path=json_path)