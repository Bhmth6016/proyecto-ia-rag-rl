# test_enhanced_structure.py
"""
Test de estructura corregida
"""
import sys
import os
from pathlib import Path

def test_imports_simple():
    """Test simple de importaciones usando sys.path"""
    
    # Añadir src al path de forma explícita
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    modules_to_test = [
        ("data.canonicalizer", "ProductCanonicalizer"),
        ("data.vector_store", "VectorStore"),
        ("query.query_understanding", "QueryUnderstanding"),
        ("consistency_checker", "ConsistencyChecker"),
    ]
    
    print("🔍 TESTEANDO IMPORTACIONES BÁSICAS")
    print("="*50)
    
    passed = []
    failed = []
    
    for module_path, class_name in modules_to_test:
        try:
            # Importar módulo
            module = __import__(module_path, fromlist=[class_name])
            # Verificar que la clase existe
            if hasattr(module, class_name):
                passed.append(f"{module_path}.{class_name}")
                print(f"✅ {module_path}.{class_name}")
            else:
                failed.append(f"{module_path}.{class_name} - Clase no encontrada")
                print(f"❌ {module_path}.{class_name} - Clase no encontrada")
        except Exception as e:
            failed.append(f"{module_path}.{class_name} - Error: {str(e)[:50]}")
            print(f"❌ {module_path}.{class_name} - Error: {str(e)[:50]}")
    
    print("="*50)
    return passed, failed


def check_directory_structure():
    """Verifica estructura de directorios"""
    
    required_dirs = [
        "src",
        "src/config",
        "src/data",
        "src/query",
        "src/features",
        "src/ranking",
        "src/evaluator",
        "results"
    ]
    
    print("\n📁 VERIFICANDO ESTRUCTURA DE DIRECTORIOS")
    print("="*50)
    
    existing = []
    missing = []
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            existing.append(dir_path)
            print(f"✅ {dir_path}")
        else:
            missing.append(dir_path)
            print(f"❌ {dir_path}")
    
    print("="*50)
    return existing, missing


if __name__ == "__main__":
    print("\n" + "="*60)
    print("VERIFICACIÓN DE ESTRUCTURA CORREGIDA")
    print("="*60 + "\n")
    
    passed_imports, failed_imports = test_imports_simple()
    existing_dirs, missing_dirs = check_directory_structure()
    
    print("\n" + "="*60)
    print("RESUMEN:")
    print(f"✅ Importaciones exitosas: {len(passed_imports)}/{len(passed_imports)+len(failed_imports)}")
    print(f"✅ Directorios existentes: {len(existing_dirs)}/{len(existing_dirs)+len(missing_dirs)}")
    
    if len(failed_imports) == 0 and len(missing_dirs) == 0:
        print("\n🎉 ¡ESTRUCTURA CORRECTA!")
        print("\nPuedes ejecutar:")
        print("  python src/main_enhanced.py --config config/experiment.yaml --demo")
    else:
        print("\n⚠️  Problemas encontrados:")
        if failed_imports:
            print("\n  Importaciones fallidas:")
            for fail in failed_imports:
                print(f"    - {fail}")
        if missing_dirs:
            print("\n  Directorios faltantes:")
            for missing in missing_dirs:
                print(f"    - {missing}")
        
        print("\nEjecuta primero:")
        print("  python setup_project.py")