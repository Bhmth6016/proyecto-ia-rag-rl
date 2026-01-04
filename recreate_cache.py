# recreate_cache.py
"""
Recrea el caché del sistema desde cero
"""
import pickle
from pathlib import Path
import sys

# Asegurar que RLHFRankerFixed esté definido
sys.path.insert(0, str(Path(__file__).parent))

from src.unified_system import UnifiedRAGRLSystem

def main():
    print("🔄 Recreando caché del sistema...")
    
    # 1. Crear sistema nuevo
    system = UnifiedRAGRLSystem()
    
    # 2. Inicializar con menos productos para prueba rápida
    print("📥 Cargando productos...")
    success = system.initialize_from_raw_all_files(limit=5000)  # Menos productos para prueba
    
    if not success:
        print("❌ Error inicializando sistema")
        return
    
    # 3. Guardar en caché
    cache_path = Path("data/cache/unified_system.pkl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(cache_path, 'wb') as f:
        pickle.dump(system, f)
    
    print(f"✅ Sistema guardado en: {cache_path}")
    print(f"📊 Productos cargados: {len(system.canonical_products):,}")
    
    # 4. Verificar que se puede cargar
    print("\n🔍 Verificando carga del caché...")
    try:
        with open(cache_path, 'rb') as f:
            loaded_system = pickle.load(f)
        print(f"✅ Caché cargado correctamente")
        print(f"   Productos: {len(loaded_system.canonical_products):,}")
    except Exception as e:
        print(f"❌ Error cargando caché: {e}")

if __name__ == "__main__":
    main()