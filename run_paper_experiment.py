"""
Script para ejecutar el experimento completo del paper
"""
import subprocess
import sys
import os
from pathlib import Path
import time

def check_dependencies():
    """Verifica que todas las dependencias estén instaladas"""
    print("🔍 Verificando dependencias...")
    
    dependencies = [
        ("faiss-cpu", "faiss"),
        ("sentence-transformers", "sentence_transformers"),
        ("spacy", "spacy"),
        ("transformers", "transformers"),
        ("pyyaml", "yaml"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("numpy", "numpy")
    ]
    
    missing = []
    for pip_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"  ✅ {pip_name}")
        except ImportError:
            missing.append(pip_name)
            print(f"  ❌ {pip_name}")
    
    if missing:
        print(f"\n⚠️  Dependencias faltantes: {', '.join(missing)}")
        print("Instalar con: pip install " + " ".join(missing))
        return False
    
    print("✅ Todas las dependencias están instaladas")
    return True

def setup_project():
    """Configura la estructura del proyecto"""
    print("\n📁 Configurando estructura del proyecto...")
    
    # Directorios necesarios
    dirs = [
        "data/raw",
        "data/processed", 
        "data/index",
        "config",
        "logs",
        "results",
        "src",
        "docs/paper"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_path}")
    
    # Verificar datos
    raw_files = list(Path("data/raw").glob("*.jsonl"))
    if raw_files:
        print(f"\n📊 Archivos de datos encontrados: {len(raw_files)}")
        for f in raw_files[:3]:  # Mostrar primeros 3
            print(f"  • {f.name}")
        if len(raw_files) > 3:
            print(f"  • ... y {len(raw_files) - 3} más")
    else:
        print("\n⚠️  No se encontraron archivos .jsonl en data/raw/")
        print("   El sistema usará datos de ejemplo")
    
    return True

def run_experiment():
    """Ejecuta el experimento completo"""
    print("\n" + "="*80)
    print("🚀 EJECUTANDO EXPERIMENTO DEL PAPER")
    print("="*80)
    
    # Verificar que el script principal existe
    main_script = Path("src/main_paper.py")
    if not main_script.exists():
        print(f"❌ No se encuentra el script principal: {main_script}")
        print("   Asegúrate de que src/main_paper.py existe")
        return False
    
    # Verificar configuración
    config_file = Path("config/paper_experiment.yaml")
    if not config_file.exists():
        print(f"❌ No se encuentra el archivo de configuración: {config_file}")
        print("   Creando configuración por defecto...")
        
        # Crear configuración mínima
        config_file.parent.mkdir(exist_ok=True)
        config_content = """experiment:
  name: "paper_experiment"
  seed: 42

embedding:
  model: "all-MiniLM-L6-v2"
  dimension: 384

evaluation:
  test_queries: ["test query"]
"""
        with open(config_file, 'w') as f:
            f.write(config_content)
        print(f"  ✅ Configuración creada: {config_file}")
    
    # Ejecutar experimento
    print(f"\n▶️  Ejecutando: python {main_script} --config {config_file}")
    print("-"*80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(main_script), "--config", str(config_file)],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Mostrar output
        print(result.stdout)
        if result.stderr:
            print("⚠️  Errores:")
            print(result.stderr)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar el experimento:")
        print(f"   Código: {e.returncode}")
        print(f"   Output: {e.output}")
        print(f"   Error: {e.stderr}")
        return False
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Tiempo total: {elapsed_time:.1f} segundos")
    
    return True

def generate_report():
    """Genera reporte del experimento"""
    print("\n📄 Generando reporte del experimento...")
    
    # Buscar resultados más recientes
    results_dirs = list(Path("results").glob("*"))
    if not results_dirs:
        print("  ⚠️  No se encontraron resultados")
        return
    
    # Ordenar por fecha de modificación
    latest_dir = max(results_dirs, key=lambda x: x.stat().st_mtime)
    
    print(f"  📁 Directorio de resultados: {latest_dir}")
    
    # Verificar archivos generados
    expected_files = [
        "executive_report.txt",
        "experiment_config.json",
        "comparative_results.json"
    ]
    
    for file_name in expected_files:
        file_path = latest_dir / file_name
        if file_path.exists():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ⚠️  {file_name} (no encontrado)")
    
    # Mostrar resumen del reporte ejecutivo
    report_path = latest_dir / "executive_report.txt"
    if report_path.exists():
        print(f"\n📋 RESUMEN EJECUTIVO:")
        print("-"*40)
        with open(report_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:20]  # Primeras 20 líneas
            for line in lines:
                print(line.rstrip())
    
    return True

def main():
    """Función principal"""
    print("\n" + "="*80)
    print("📚 SISTEMA HÍBRIDO RAG+RL PARA E-COMMERCE")
    print("   Implementación del Paper Académico")
    print("="*80)
    
    # Paso 1: Verificar dependencias
    if not check_dependencies():
        print("\n❌ Instala las dependencias faltantes antes de continuar")
        return
    
    # Paso 2: Configurar proyecto
    if not setup_project():
        print("\n❌ Error al configurar el proyecto")
        return
    
    # Paso 3: Ejecutar experimento
    if not run_experiment():
        print("\n❌ El experimento falló")
        return
    
    # Paso 4: Generar reporte
    generate_report()
    
    print("\n" + "="*80)
    print("🎉 ¡EXPERIMENTO COMPLETADO EXITOSAMENTE!")
    print("\n📌 Próximos pasos:")
    print("   1. Revisa los resultados en: results/[fecha_hora]/")
    print("   2. Usa las gráficas y tablas para tu paper")
    print("   3. Modifica la configuración en config/paper_experiment.yaml")
    print("   4. Ejecuta experimentos adicionales si es necesario")
    print("="*80)

if __name__ == "__main__":
    main()