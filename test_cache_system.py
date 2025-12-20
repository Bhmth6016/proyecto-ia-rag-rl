# test_cache_system.py
"""
Script para probar el sistema con caché
"""
import sys
from pathlib import Path

# Configurar paths
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

def test_cache():
    """Prueba el sistema con caché"""
    print("🧪 PROBANDO SISTEMA CON CACHÉ")
    print("="*50)
    
    # Primero, verificar si hay caché
    cache_dir = Path("data/cache")
    cache_files = list(cache_dir.glob("*")) if cache_dir.exists() else []
    
    if cache_files:
        print("✅ Caché encontrado")
        for f in cache_files[:3]:
            size_mb = f.stat().st_size / (1024*1024)
            print(f"  • {f.name}: {size_mb:.1f} MB")
    else:
        print("⚠️  No hay caché, se creará nuevo")
    
    # Ejecutar sistema con caché
    print("\n🚀 Ejecutando sistema con caché...")
    
    import subprocess
    import time
    
    # Medir tiempo con caché
    start_time = time.time()
    result = subprocess.run([sys.executable, "run_simple_cache.py", "--query", "car parts"], 
                          capture_output=True, text=True)
    elapsed_with_cache = time.time() - start_time
    
    print(f"\n⏱️  Tiempo con caché: {elapsed_with_cache:.1f} segundos")
    
    # Ejecutar sin caché para comparar
    print("\n🚀 Ejecutando sistema SIN caché (solo para comparación)...")
    
    start_time = time.time()
    result_no_cache = subprocess.run([sys.executable, "run_simple_cache.py", "--no-cache", "--query", "car parts"],
                                   capture_output=True, text=True, timeout=300)  # 5 minutos máximo
    elapsed_without_cache = time.time() - start_time
    
    print(f"\n⏱️  Tiempo SIN caché: {elapsed_without_cache:.1f} segundos")
    print(f"📈 Mejora: {(elapsed_without_cache/elapsed_with_cache):.1f}x más rápido con caché")

if __name__ == "__main__":
    test_cache()