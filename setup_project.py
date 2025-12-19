"""
Script de inicialización del proyecto
"""
import os
import sys
from pathlib import Path

def setup_project_structure():
    """Crea la estructura completa del proyecto"""
    
    # Directorios principales
    directories = [
        "src",
        "src/config",
        "src/data",
        "src/query",
        "src/features",
        "src/ranking",
        "src/evaluator",
        "src/validation",
        "src/experiments",
        "src/consistency_checker",
        "results",
        "results/plots",
        "results/logs",
        "results/checkpoints",
        "data/raw",
        "data/processed",
        "config",
        "docs"
    ]
    
    print("📁 Creando estructura de directorios...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ {directory}")
    
    # Archivos __init__.py
    init_files = [
        "src/__init__.py",
        "src/config/__init__.py",
        "src/data/__init__.py",
        "src/query/__init__.py",
        "src/features/__init__.py",
        "src/ranking/__init__.py",
        "src/evaluator/__init__.py",
        "src/validation/__init__.py",
        "src/experiments/__init__.py",
        "src/consistency_checker/__init__.py"
    ]
    
    print("\n📄 Creando archivos __init__.py...")
    for init_file in init_files:
        Path(init_file).touch()
        print(f"  ✅ {init_file}")
    
    # Verificar archivos principales
    required_files = [
        "src/main_enhanced.py",
        "config/experiment.yaml",
        "test_structure.py"
    ]
    
    print("\n🔍 Verificando archivos principales...")
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ⚠️  {file_path} - No encontrado")
    
    print("\n" + "="*50)
    print("✅ ESTRUCTURA DEL PROYECTO CONFIGURADA")
    print("\nPara probar el sistema:")
    print("1. python test_structure.py")
    print("2. python src/main_enhanced.py --config config/experiment.yaml")
    print("\nPara instalar dependencias:")
    print("pip install faiss-cpu sentence-transformers spacy transformers pyyaml")
    print("python -m spacy download en_core_web_sm")


if __name__ == "__main__":
    setup_project_structure()