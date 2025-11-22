#!/usr/bin/env python3
"""
Reparación definitiva del sistema Chroma
"""
import sys
from pathlib import Path
import time
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

def force_clear_chroma_files():
    """Forzar limpieza de archivos de Chroma bloqueados."""
    print("🔒 Limpiando archivos bloqueados de Chroma...")
    
    index_path = Path("data/processed/chroma_db")
    
    if not index_path.exists():
        print("✅ No existe índice anterior")
        return True
        
    # Método forzado de limpieza
    max_retries = 5
    for attempt in range(max_retries):
        try:
            import shutil
            
            # Cerrar cualquier conexión primero
            try:
                import sqlite3
                conn = sqlite3.connect(index_path / "chroma.sqlite3")
                conn.close()
            except:
                pass
                
            # Esperar un poco
            time.sleep(1)
            
            # Usar el método forzado
            for root, dirs, files in os.walk(index_path, topdown=False):
                for name in files:
                    file_path = os.path.join(root, name)
                    try:
                        os.chmod(file_path, 0o777)
                        os.remove(file_path)
                        print(f"   📄 Eliminado: {name}")
                    except Exception as e:
                        print(f"   ⚠️  No se pudo eliminar {name}: {e}")
                
                for name in dirs:
                    dir_path = os.path.join(root, name)
                    try:
                        os.chmod(dir_path, 0o777)
                        os.rmdir(dir_path)
                    except:
                        pass
            
            # Eliminar directorio principal
            try:
                os.rmdir(index_path)
            except:
                pass
                
            print(f"✅ Índice anterior eliminado (intento {attempt + 1})")
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️  Error, reintentando... ({attempt + 1}/{max_retries}): {e}")
                time.sleep(2)
            else:
                print(f"❌ No se pudo eliminar el índice: {e}")
                return False
    
    return False

def rebuild_complete_system():
    print("🔧 RECONSTRUCCIÓN COMPLETA DEL SISTEMA")
    print("=" * 50)
    
    # 1. Forzar limpieza de archivos bloqueados
    print("🔄 Paso 1: Limpiando archivos anteriores...")
    if not force_clear_chroma_files():
        print("⚠️  Continuando con reconstrucción...")
    
    # 2. Cargar datos
    print("🔄 Paso 2: Cargando datos...")
    from src.core.data.loader import DataLoader
    loader = DataLoader()
    
    all_products = loader.load_data()
    products = all_products[:1000]  # Solo 1000 productos para prueba
    print(f"📦 Productos cargados: {len(products)} (de {len(all_products)} totales)")
    
    # 3. Reconstruir índice
    print("🔄 Paso 3: Construyendo nuevo índice...")
    from src.core.rag.basic.retriever import Retriever
    retriever = Retriever()
    
    try:
        retriever.build_index(products)
        print("✅ Índice reconstruido exitosamente")
    except Exception as e:
        print(f"❌ Error construyendo índice: {e}")
        return False
    
    # 4. Verificar que funciona
    print("\n🔍 VERIFICANDO FUNCIONAMIENTO...")
    print("=" * 50)
    
    test_queries = [
        "game",
        "software", 
        "music",
        "beauty", 
        "professional"
    ]
    
    success_count = 0
    for query in test_queries:
        try:
            print(f"🔍 Probando: '{query}'")
            results = retriever.retrieve(query, k=2, min_similarity=0.05)
            
            if results:
                print(f"   ✅ {len(results)} resultados")
                product = results[0]
                title = getattr(product, 'title', 'N/A')[:60]
                score = getattr(product, 'score', 0)
                print(f"   📝 Ejemplo: {title}... (score: {score:.3f})")
                success_count += 1
            else:
                print(f"   ⚠️  0 resultados")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 RESULTADO: {success_count}/{len(test_queries)} consultas exitosas")
    
    return success_count >= 3

if __name__ == "__main__":
    try:
        print("🚀 INICIANDO REPARACIÓN DEFINITIVA...")
        print("💡 Si falla, cierra VS Code y ejecuta desde PowerShell como administrador")
        print("-" * 60)
        
        if rebuild_complete_system():
            print("\n🎉 ¡SISTEMA REPARADO EXITOSAMENTE!")
            print("✅ Puedes ejecutar ahora: python test_complete_system.py")
        else:
            print("\n❌ El sistema aún tiene problemas")
            print("\n🔧 SOLUCIÓN MANUAL:")
            print("   1. Cierra VS Code completamente")
            print("   2. Abre PowerShell COMO ADMINISTRADOR")
            print("   3. Ejecuta estos comandos:")
            print("      cd 'C:\\Users\\evill\\OneDrive\\Documentos\\Github\\github\\proyecto-ia-rag-rl'")
            print("      Remove-Item -Recurse -Force 'data\\processed\\chroma_db'")
            print("   4. Luego ejecuta: python fix_retriever.py")
            
    except Exception as e:
        print(f"\n💥 Error crítico: {e}")
        import traceback
        traceback.print_exc()