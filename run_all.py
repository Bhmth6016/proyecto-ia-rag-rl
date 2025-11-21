# run_all.py

from pathlib import Path
from src.core.data.chroma_builder import OptimizedChromaBuilder
from src.core.data.loader import AutomatedDataLoader

def main():
    print("\n=== 🚀 INICIANDO EJECUCIÓN DE LOS TRES MÓDULOS ===\n")

    # -----------------------------
    # 1. Ejecutar loader.py
    # -----------------------------
    print("📦 Ejecutando AutomatedDataLoader...")
    loader = AutomatedDataLoader(
    auto_categories=False,
    auto_tags=False,
    min_samples_for_training=0
)

    products = loader.load_data(use_cache=False, output_file="products.json")

    print(f"✔ Productos cargados: {len(products)}")

    # -----------------------------
    # 2. Ejecutar chroma_builder.py
    # -----------------------------
    print("\n📄 Construyendo documentos e índice Chroma...")
    builder = OptimizedChromaBuilder(
        processed_json_path=loader.processed_dir / "products.json"
    )

    documents = builder.create_documents_optimized(products)
    print(f"✔ Documentos generados: {len(documents)}")

    chroma_index = builder.build_index_optimized()
    print("\n✔ Índice Chroma generado exitosamente.")

    print("\n=== ✅ PROCESO COMPLETO ===")

if __name__ == "__main__":
    main()
