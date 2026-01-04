# Crear archivo: initialize_system.py
"""
Inicializa y guarda el sistema base
"""
import pickle
from pathlib import Path
from src.unified_system import UnifiedRAGRLSystem

print("🚀 Inicializando sistema...")
system = UnifiedRAGRLSystem()
system.initialize_from_raw_all_files(limit=20000)

# Guardar sistema base
cache_dir = Path("data/cache")
cache_dir.mkdir(exist_ok=True)

with open(cache_dir / "unified_system.pkl", 'wb') as f:
    pickle.dump(system, f)

print("✅ Sistema base guardado en data/cache/unified_system.pkl")