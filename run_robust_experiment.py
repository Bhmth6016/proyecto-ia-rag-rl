"""
Script robusto para ejecutar experimentos
"""
import subprocess
import sys
from pathlib import Path
import time
import json

def check_environment():
    """Verifica el entorno"""
    print("🔍 VERIFICANDO ENTORNO")
    print("-"*40)
    
    # Verificar estructura de directorios
    required_dirs = ["data/raw", "config", "src", "results"]
    all_ok = True
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - Creando...")
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            all_ok = False
    
    # Verificar archivos de datos
    raw_files = list(Path("data/raw").glob("*.jsonl"))
    if raw_files:
        print(f"✅ Archivos de datos: {len(raw_files)} encontrados")
        for f in raw_files[:3]:
            print(f"  • {f.name}")
        if len(raw_files) > 3:
            print(f"  • ... y {len(raw_files) - 3} más")
    else:
        print("⚠️  No hay archivos .jsonl en data/raw/")
        print("   Se usarán datos de ejemplo")
        all_ok = False
    
    # Verificar configuración
    config_file = Path("config/paper_experiment.yaml")
    if config_file.exists():
        print(f"✅ Configuración: {config_file}")
    else:
        print(f"❌ Configuración no encontrada: {config_file}")
        print("   Creando configuración básica...")
        create_basic_config(config_file)
        all_ok = False
    
    print("-"*40)
    return all_ok

def create_basic_config(config_path: Path):
    """Crea configuración básica"""
    config_content = """# Configuración básica para experimento RAG+RL
experiment:
  name: "rag_rl_ecommerce_basic"
  seed: 42
  version: "1.0"

dataset:
  raw_path: "data/raw"
  sample_size: 1000
  max_files: 2

embedding:
  model: "all-MiniLM-L6-v2"
  dimension: 384

retrieval:
  top_k: 50

ranking:
  baseline_weights:
    content_similarity: 0.4
    title_similarity: 0.2
    category_exact_match: 0.15
    rating_normalized: 0.1
    price_available: 0.05
    has_brand: 0.05
    title_length: 0.025
    desc_length: 0.025

evaluation:
  test_queries: [
    "smartphone with camera",
    "laptop for work",
    "headphones wireless"
  ]
"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"   ✓ Configuración creada: {config_path}")

def run_robust_experiment():
    """Ejecuta experimento robusto"""
    print("\n🚀 EJECUTANDO EXPERIMENTO ROBUSTO")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Usar la versión robusta
        result = subprocess.run(
            [sys.executable, "src/main_robust.py", "--config", "config/paper_experiment.yaml"],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        # Mostrar output
        print("📋 OUTPUT DEL EXPERIMENTO:")
        print("-"*40)
        lines = result.stdout.split('\n')
        for line in lines[-50:]:  # Últimas 50 líneas
            if line.strip():
                print(line)
        
        if result.stderr:
            print("\n⚠️  ERRORES:")
            print("-"*40)
            for line in result.stderr.split('\n')[:20]:  # Primeras 20 líneas de error
                if line.strip():
                    print(line)
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Tiempo total: {elapsed_time:.1f} segundos")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error al ejecutar: código {e.returncode}")
        
        # Mostrar error de forma más amigable
        if "ModuleNotFoundError" in e.stderr:
            print("🔧 Problema de importación detectado")
            print("   Ejecuta: pip install -r requirements.txt")
        
        print("\n📋 Últimas líneas de error:")
        lines = e.stderr.split('\n')
        for line in lines[-20:]:
            if line.strip():
                print(f"   {line}")
        
        return False
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        return False

def show_results():
    """Muestra resultados del experimento"""
    print("\n📊 RESULTADOS DEL EXPERIMENTO")
    print("="*60)
    
    # Buscar directorio de resultados más reciente
    results_dirs = list(Path("results").glob("*"))
    if not results_dirs:
        print("⚠️  No se encontraron directorios de resultados")
        return
    
    # Ordenar por fecha de creación
    results_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_dir = results_dirs[0]
    
    print(f"📁 Directorio más reciente: {latest_dir.name}")
    print("-"*40)
    
    # Mostrar archivos en el directorio
    files = list(latest_dir.glob("*"))
    if files:
        print("📄 Archivos generados:")
        for file_path in files[:10]:  # Mostrar primeros 10
            size_kb = file_path.stat().st_size / 1024
            print(f"  • {file_path.name} ({size_kb:.1f} KB)")
        
        if len(files) > 10:
            print(f"  • ... y {len(files) - 10} más")
    else:
        print("⚠️  No hay archivos en el directorio")
    
    # Intentar mostrar reporte si existe
    report_path = latest_dir / "experiment_report.txt"
    if report_path.exists():
        print(f"\n📋 Extracto del reporte:")
        print("-"*40)
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[:15]:  # Primeras 15 líneas
                    print(f"  {line.rstrip()}")
        except:
            print("  (No se pudo leer el reporte)")
    
    print(f"\n📍 Ruta completa: {latest_dir.absolute()}")

def main():
    """Función principal"""
    print("\n" + "="*60)
    print("🤖 SISTEMA RAG+RL PARA E-COMMERCE")
    print("   Ejecución Robusta de Experimentos")
    print("="*60)
    
    # Paso 1: Verificar entorno
    if not check_environment():
        print("\n⚠️  Algunos problemas detectados, pero continuando...")
    
    # Paso 2: Ejecutar experimento
    print("\n" + "="*60)
    success = run_robust_experiment()
    
    # Paso 3: Mostrar resultados
    if success:
        show_results()
        
        print("\n" + "="*60)
        print("🎉 ¡EXPERIMENTO COMPLETADO!")
        print("\n📌 Para ejecutar experimentos específicos:")
        print("   python src/main_robust.py --config config/paper_experiment.yaml")
        print("\n📌 Para ver todos los resultados:")
        print("   ls -la results/*/")
    else:
        print("\n" + "="*60)
        print("❌ EL EXPERIMENTO FALLÓ")
        print("\n🔧 Solución de problemas:")
        print("   1. Verifica que todos los módulos estén en src/")
        print("   2. Asegúrate de tener datos en data/raw/")
        print("   3. Revisa los logs en la salida anterior")
        print("   4. Ejecuta en modo simple: python src/main_simple.py")
    
    print("="*60)

if __name__ == "__main__":
    main()