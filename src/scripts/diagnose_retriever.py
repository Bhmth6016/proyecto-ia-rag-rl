#!/usr/bin/env python3
"""
Diagnóstico y reparación del retriever
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def diagnose_retriever_issue():
    """Diagnostica el problema del retriever"""
    print("🔍 DIAGNÓSTICO DEL RETRIEVER")
    print("=" * 50)
    
    from src.core.rag.basic.retriever import Retriever
    from src.core.data.loader import DataLoader
    from src.core.config import settings
    
    # 1. Cargar datos
    print("📦 Cargando datos...")
    loader = DataLoader()
    products = loader.load_data()[:50]  # Pocos para prueba rápida
    print(f"   ✅ {len(products)} productos cargados")
    
    # 2. Crear retriever
    print("\n🔧 Inicializando retriever...")
    retriever = Retriever()
    
    # 3. Verificar estado actual
    print("\n📊 ESTADO ACTUAL:")
    print(f"   Index exists: {retriever.index_exists()}")
    print(f"   Store: {retriever.store}")
    print(f"   Vector index path: {settings.VECTOR_INDEX_PATH}")
    
    # 4. Verificar si el directorio de índice existe
    index_path = Path(settings.VECTOR_INDEX_PATH)
    print(f"   Index path exists: {index_path.exists()}")
    if index_path.exists():
        contents = list(index_path.iterdir())
        print(f"   Contents: {[f.name for f in contents]}")
    
    # 5. Intentar construir índice
    print("\n🛠️ Construyendo índice...")
    try:
        retriever.build_index(products)
        print("   ✅ Índice construido exitosamente")
    except Exception as e:
        print(f"   ❌ Error construyendo índice: {e}")
        return False
    
    # 6. Verificar estado después de construcción
    print("\n📊 ESTADO DESPUÉS DE CONSTRUIR:")
    print(f"   Store: {retriever.store}")
    print(f"   Store type: {type(retriever.store)}")
    
    # 7. Probar búsqueda
    print("\n🔍 Probando búsqueda...")
    try:
        results = retriever.retrieve("laptop", k=3)
        print(f"   ✅ Búsqueda exitosa: {len(results)} resultados")
        for i, product in enumerate(results, 1):
            title = getattr(product, 'title', 'N/A')[:50]
            print(f"      {i}. {title}")
        return True
    except Exception as e:
        print(f"   ❌ Error en búsqueda: {e}")
        return False

def fix_retriever_issue():
    """Solución alternativa para el retriever"""
    print("\n🛠️ APLICANDO SOLUCIÓN ALTERNATIVA...")
    print("=" * 50)
    
    from src.core.data.loader import DataLoader
    from src.core.rag.basic.retriever import Retriever
    import shutil
    from pathlib import Path
    from src.core.config import settings
    
    # 1. Limpiar índice existente (puede estar corrupto)
    index_path = Path(settings.VECTOR_INDEX_PATH)
    if index_path.exists():
        print("🗑️  Limpiando índice existente...")
        shutil.rmtree(index_path)
        print("   ✅ Índice anterior eliminado")
    
    # 2. Reconstruir desde cero
    print("📦 Cargando datos...")
    loader = DataLoader()
    products = loader.load_data()[:100]
    print(f"   ✅ {len(products)} productos cargados")
    
    # 3. Reconstruir índice
    print("🔧 Reconstruyendo índice...")
    retriever = Retriever()
    
    try:
        retriever.build_index(products)
        print("   ✅ Índice reconstruido")
        
        # 4. Verificar
        print("🔍 Verificando búsqueda...")
        results = retriever.retrieve("laptop", k=3)
        print(f"   ✅ Búsqueda funciona: {len(results)} resultados")
        
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🎯 DIAGNÓSTICO Y REPARACIÓN DEL RETRIEVER")
    
    # Primero diagnosticar
    if not diagnose_retriever_issue():
        print("\n" + "⚠️" * 20)
        print("SE DETECTÓ PROBLEMA - APLICANDO REPARACIÓN...")
        print("⚠️" * 20)
        
        # Luego reparar
        if fix_retriever_issue():
            print("\n🎉 ¡RETRIEVER REPARADO!")
            print("\n🔁 Ejecuta nuevamente: python test_complete_system.py")
        else:
            print("\n❌ No se pudo reparar automáticamente")
    else:
        print("\n✅ El retriever ya funciona correctamente")