"""
Corrige typos en los archivos del proyecto
"""
import os
from pathlib import Path

def fix_evaluator_typo():
    """Corrige el typo en evaluator.py"""
    evaluator_path = Path("src/evaluator/evaluator.py")
    
    if not evaluator_path.exists():
        print(f"❌ Archivo no encontrado: {evaluator_path}")
        return False
    
    try:
        with open(evaluator_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Corregir typo
        if "CanicalProduct" in content:
            content = content.replace("CanicalProduct", "CanonicalProduct")
            print("✅ Typo 'CanicalProduct' corregido a 'CanonicalProduct'")
        else:
            print("✅ No se encontró el typo 'CanicalProduct'")
        
        # Guardar
        with open(evaluator_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"❌ Error al corregir typo: {e}")
        return False

def create_missing_files():
    """Crea archivos faltantes"""
    files_to_create = {
        "config/experiment.yaml": """# Configuración experimental básica
experiment:
  name: "ecommerce_rag_rl_demo"
  seed: 42
  version: "1.0"

embedding:
  model: "all-MiniLM-L6-v2"
  dimension: 384

ranking:
  baseline_weights:
    content_similarity: 0.4
    title_similarity: 0.2
    category_exact_match: 0.15
    rating_normalized: 0.1
    price_available: 0.05
    has_popularity: 0.05
    title_length: 0.025
    desc_length: 0.025

evaluation:
  test_queries: [
    "nintendo switch games",
    "laptop for programming",
    "wireless headphones"
  ]
""",
        "test_structure.py": """"""
    }
    
    created = []
    for file_path, content in files_to_create.items():
        path = Path(file_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            created.append(file_path)
            print(f"✅ Creado: {file_path}")
        else:
            print(f"✅ Ya existe: {file_path}")
    
    return created

def check_all_files():
    """Verifica todos los archivos importantes"""
    important_files = [
        "src/main_simple.py",
        "src/evaluator/evaluator.py",
        "config/experiment.yaml",
        "data/raw/sample_products.json",
        "src/data/canonicalizer.py",
        "src/data/vector_store.py",
        "src/query/query_understanding.py",
        "src/features/features.py",
        "src/ranking/ranking_engine.py",
        "src/consistency_checker/__init__.py"
    ]
    
    print("\n🔍 VERIFICANDO ARCHIVOS IMPORTANTES:")
    print("="*50)
    
    existing = []
    missing = []
    
    for file_path in important_files:
        if Path(file_path).exists():
            existing.append(file_path)
            print(f"✅ {file_path}")
        else:
            missing.append(file_path)
            print(f"❌ {file_path}")
    
    print("="*50)
    return existing, missing

def main():
    """Corrige todos los problemas"""
    print("🔧 CORRIGIENDO PROBLEMAS DEL PROYECTO")
    print("="*60)
    
    # 1. Corregir typo en evaluator
    print("\n1. Corrigiendo typo en evaluator.py...")
    fix_evaluator_typo()
    
    # 2. Crear archivos faltantes
    print("\n2. Creando archivos faltantes...")
    create_missing_files()
    
    # 3. Verificar estructura
    print("\n3. Verificando estructura completa...")
    existing, missing = check_all_files()
    
    print("\n" + "="*60)
    print("RESUMEN:")
    print(f"✅ Archivos existentes: {len(existing)}")
    print(f"⚠️  Archivos faltantes: {len(missing)}")
    
    if missing:
        print("\nArchivos que necesitas crear:")
        for file in missing:
            print(f"  - {file}")
    
    print("\n🎯 PARA EJECUTAR:")
    print("  python src/main_simple.py --config config/experiment.yaml")

if __name__ == "__main__":
    main()