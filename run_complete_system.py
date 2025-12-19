"""
Script para ejecutar el sistema completo con los 4 puntos
"""
import subprocess
import sys
from pathlib import Path
import time
import json
import yaml

def create_complete_config(config_file: Path):
    """Crea configuración completa"""
    print("   Creando configuración completa...")
    
    complete_config = """experiment:
  name: "complete_rag_rl_system"
  seed: 42

dataset:
  raw_path: "data/raw"
  sample_size: 2000
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
  ml_weights:
    content_similarity: 0.35
    title_similarity: 0.2
    category_exact_match: 0.2
    rating_normalized: 0.15
    price_available: 0.05
    has_brand: 0.05

evaluation:
  test_queries: [
    "smartphone with good camera",
    "gaming laptop",
    "wireless headphones",
    "science fiction books",
    "running shoes"
  ]

rlhf:
  alpha: 0.1
  num_episodes: 20
"""
    
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        f.write(complete_config)
    print(f"   ✓ Configuración completa creada: {config_file}")

def check_and_fix_config(config_file: Path):
    """Verifica y repara la configuración si es necesario"""
    if not config_file.exists():
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verificar secciones obligatorias
        required_sections = ['experiment', 'dataset', 'embedding', 'ranking', 'evaluation']
        
        missing_sections = []
        for section in required_sections:
            if section not in config:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"⚠️  Configuración incompleta. Faltan: {missing_sections}")
            return False
        
        # Verificar sub-secciones importantes
        if 'ranking' in config and 'baseline_weights' not in config['ranking']:
            print("⚠️  Faltan baseline_weights en ranking")
            return False
        
        if 'evaluation' in config and 'test_queries' not in config['evaluation']:
            print("⚠️  Faltan test_queries en evaluation")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error leyendo configuración: {e}")
        return False

def main():
    """Ejecuta el sistema completo"""
    print("\n" + "="*70)
    print("🎯 SISTEMA COMPLETO RAG+RL - 4 PUNTOS DEL PAPER")
    print("="*70)
    
    # Verificar configuración
    config_file = Path("config/paper_experiment.yaml")
    
    if not config_file.exists() or not check_and_fix_config(config_file):
        if config_file.exists():
            print("❌ Configuración existente incompleta o inválida")
        else:
            print("❌ No se encuentra el archivo de configuración")
        
        create_complete_config(config_file)
    
    print(f"\n📋 Configuración: {config_file}")
    print("🔄 Ejecutando sistema completo...")
    print("-"*70)
    
    start_time = time.time()
    
    try:
        # Ejecutar sistema completo
        result = subprocess.run(
            [sys.executable, "src/main_complete.py", "--config", str(config_file)],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'  # Ignorar errores de encoding
        )
        
        # Mostrar output resumido
        print("\n📋 SALIDA DEL SISTEMA:")
        print("-"*40)
        
        lines = result.stdout.split('\n')
        important_lines = []
        
        # Filtrar líneas importantes
        keywords = ['PUNTO', '✅', '📊', '📋', '📈', '📄', '🎯', '❌', '⚠️', 'ERROR']
        for line in lines:
            if any(keyword in line for keyword in keywords) or 'implementado' in line.lower():
                important_lines.append(line)
        
        # Mostrar últimas 40 líneas importantes o todas si son menos
        display_lines = important_lines[-40:] if len(important_lines) > 40 else important_lines
        for line in display_lines:
            if line.strip():  # Solo mostrar líneas no vacías
                print(line)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n⏱️  Tiempo total de ejecución: {elapsed_time:.1f} segundos")
        
        # Mostrar resultados generados
        print("\n📁 RESULTADOS GENERADOS:")
        print("-"*40)
        
        # Buscar directorio de resultados más reciente
        results_dirs = list(Path("results").glob("*"))
        if results_dirs:
            results_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_dir = results_dirs[0]
            
            print(f"📂 Directorio: {latest_dir.name}")
            
            # Mostrar archivos importantes
            important_files = [
                "executive_summary.txt",
                "point_metrics.json",
                "experiment_documentation.txt"
            ]
            
            for file_name in important_files:
                file_path = latest_dir / file_name
                if file_path.exists():
                    print(f"  ✓ {file_name}")
                    
                    # Mostrar extracto del resumen ejecutivo
                    if file_name == "executive_summary.txt":
                        print("\n  📄 Extracto del resumen ejecutivo:")
                        print("  " + "-"*36)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                for i in range(min(10, len(lines))):
                                    print(f"  {lines[i].rstrip()}")
                            print("  ...")
                        except:
                            pass
            
            # Mostrar archivos en plots/
            plots_dir = latest_dir / "plots"
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*.png"))
                if plot_files:
                    print(f"\n  🖼️  Gráficas generadas: {len(plot_files)}")
                    for plot in plot_files[:3]:
                        print(f"    • {plot.name}")
                    if len(plot_files) > 3:
                        print(f"    • ... y {len(plot_files)-3} más")
            
            # Mostrar archivos en tables/
            tables_dir = latest_dir / "tables"
            if tables_dir.exists():
                table_files = list(tables_dir.glob("*"))
                if table_files:
                    print(f"\n  📊 Tablas generadas: {len(table_files)}")
                    for table in table_files[:3]:
                        print(f"    • {table.name}")
        
        print("\n" + "="*70)
        print("🎉 ¡SISTEMA COMPLETO EJECUTADO EXITOSAMENTE!")
        print("\n📌 Próximos pasos para tu paper:")
        print("   1. Revisa results/[fecha_hora]/executive_summary.txt")
        print("   2. Usa las tablas en results/[fecha_hora]/tables/")
        print("   3. Incluye las gráficas de results/[fecha_hora]/plots/")
        print("   4. Cita los resultados en tu metodología y análisis")
        print("="*70)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en la ejecución (código {e.returncode}):")
        print("-"*40)
        
        # Mostrar error de forma amigable
        error_lines = e.stderr.split('\n')[-20:]  # Últimas 20 líneas
        for line in error_lines:
            if line.strip():
                print(f"  {line}")
        
        print(f"\n💡 Sugerencias:")
        print("  1. Verifica que todos los módulos estén en src/")
        print("  2. Asegúrate de tener permisos de lectura en data/raw/")
        print("  3. Prueba ejecutar primero: python src/main_simple.py")
        print("  4. Revisa los logs en logs/")
        
        print("\n" + "="*70)
        print("❌ EJECUCIÓN FALLIDA - REVISA LOS ERRORES ARRIBA")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        print(traceback.format_exc())
        print("="*70)


if __name__ == "__main__":
    main()