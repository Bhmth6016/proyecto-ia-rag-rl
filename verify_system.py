#!/usr/bin/env python3
"""
Verificación paso a paso del sistema
"""
import importlib
from pathlib import Path

def check_imports():
    """Verifica que todos los imports funcionen"""
    modules = [
        "src.core.data.loader",
        "src.core.data.product", 
        "src.core.rag.basic.retriever",
        "src.core.rag.advanced.RAGAgent",
        "src.core.rag.advanced.feedback_processor",
        "src.core.config"
    ]
    
    print("🔍 Verificando imports...")
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")

def check_directories():
    """Verifica que existan los directorios necesarios"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/feedback",
        "logs"
    ]
    
    print("\n📁 Verificando directorios...")
    for dir_path in directories:
        path = Path(dir_path)
        if path.exists():
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ⚠️  {dir_path} (creando...)")
            path.mkdir(parents=True, exist_ok=True)

def check_data_files():
    """Verifica que haya datos para procesar"""
    data_dir = Path("data/raw")
    print("\n📊 Verificando archivos de datos...")
    
    json_files = list(data_dir.glob("*.json"))
    jsonl_files = list(data_dir.glob("*.jsonl"))
    
    all_files = json_files + jsonl_files
    
    if all_files:
        for file in all_files[:3]:  # Mostrar primeros 3
            print(f"   ✅ {file.name}")
        if len(all_files) > 3:
            print(f"   ... y {len(all_files) - 3} más")
    else:
        print("   ❌ No hay archivos de datos en data/raw/")
        print("   💡 Coloca aquí tus archivos .json o .jsonl con productos")

if __name__ == "__main__":
    print("🎯 VERIFICACIÓN DEL SISTEMA")
    print("=" * 50)
    
    check_imports()
    check_directories() 
    check_data_files()
    
    print("\n" + "=" * 50)
    print("📝 PRÓXIMOS PASOS:")
    print("   1. Asegúrate de tener datos en data/raw/")
    print("   2. Ejecuta: python test_complete_system.py")
    print("   3. O usa: python main.py rag")