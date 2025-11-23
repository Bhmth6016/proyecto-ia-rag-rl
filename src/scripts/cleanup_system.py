#!/usr/bin/env python3
"""
Limpieza específica - Solo elimina traducción y categorías complejas
MANTIENE todo el enriquecimiento de datos + CORRIGE imports RLHF
"""
# src/core/scripts/cleanup_system.py
import shutil
import json
from pathlib import Path

def remove_translation_components():
    """Elimina solo componentes de traducción"""
    
    translation_files = [
        "src/core/utils/translator.py",
    ]
    
    category_complex_files = [
        "src/core/category_search/category_tree.py",
    ]
    
    print("🧹 ELIMINANDO COMPONENTES DE TRADUCCIÓN...")
    
    removed_count = 0
    
    # Eliminar archivos de traducción
    for file_path in translation_files:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            print(f"🗑️  Eliminado traductor: {file_path}")
            removed_count += 1
        else:
            print(f"✅ Ya eliminado: {file_path}")
    
    # Eliminar categorías complejas (no el enriquecimiento básico)
    for file_path in category_complex_files:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            print(f"🗑️  Eliminado categorías complejas: {file_path}")
            removed_count += 1
        else:
            print(f"✅ Ya eliminado: {file_path}")
    
    # Intentar eliminar directorio vacío
    category_dir = Path("src/core/category_search/")
    if category_dir.exists() and category_dir.is_dir():
        try:
            if not any(category_dir.iterdir()):
                category_dir.rmdir()
                print(f"🗑️  Eliminado directorio vacío: category_search/")
            else:
                print(f"⚠️  Directorio no vacío, manteniendo: category_search/")
        except Exception as e:
            print(f"⚠️  No se pudo eliminar directorio: {e}")
    
    print(f"✅ Eliminados {removed_count} componentes obsoletos")

def verify_data_enrichment_remains():
    """Verifica que el enriquecimiento de datos se mantiene intacto"""
    
    key_files = [
        "src/core/data/loader.py",
        "src/core/data/product.py", 
        "src/core/data/chroma_builder.py"
    ]
    
    print("\n🔍 VERIFICANDO ENRIQUECIMIENTO DE DATOS...")
    
    enrichment_keywords = [
        "auto_discover_categories",
        "to_text()",
        "to_metadata()", 
        "embedding",
        "features",
        "details",
        "clean_image_urls",
        "content_hash"
    ]
    
    for file_path in key_files:
        path = Path(file_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            found_features = []
            for keyword in enrichment_keywords:
                if keyword in content:
                    found_features.append(keyword)
            
            if found_features:
                print(f"✅ {file_path} - Mantiene: {len(found_features)} features de enriquecimiento")
            else:
                print(f"⚠️  {file_path} - ¿Features de enriquecimiento?")
        else:
            print(f"❌ {file_path} - No encontrado")

def check_obsolete_imports():
    """Busca imports obsoletos en archivos clave"""
    
    files_to_check = [
        "src/core/rag/advanced/RAGAgent.py",
        "src/core/rag/advanced/feedback_processor.py",
        "main.py"
    ]
    
    obsolete_imports = [
        "translator",
        "TextTranslator",
        "Language", 
        "category_tree",
        "CategoryTree"
    ]
    
    print("\n🔍 BUSCANDO IMPORTS OBSOLETOS...")
    
    issues_found = 0
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            found = []
            for obsolete in obsolete_imports:
                if obsolete in content:
                    found.append(obsolete)
            
            if found:
                print(f"⚠️  {file_path} necesita actualización: {', '.join(found)}")
                issues_found += 1
            else:
                print(f"✅ {file_path} - Sin imports obsoletos")
    
    return issues_found

def fix_rlhf_imports():
    """Corrige importaciones incorrectas de RLHF"""
    
    files_to_fix = {
        "main.py": {
            "old": "from src.core.rag.advanced.rlhf import RAGAgent",
            "new": "from src.core.rag.advanced.RAGAgent import RAGAgent"
        },
        "scripts/initialize_system.py": {
            "old": "from src.core.rag.advanced.rlhf import RAGAgent", 
            "new": "from src.core.rag.advanced.RAGAgent import RAGAgent"
        }
    }
    
    print("\n🔧 CORRIGIENDO IMPORTACIONES RLHF...")
    
    fixed_count = 0
    for file_path, replacements in files_to_fix.items():
        path = Path(file_path)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Reemplazar import incorrecto
                if replacements["old"] in content:
                    content = content.replace(replacements["old"], replacements["new"])
                    
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"✅ Corregido: {file_path}")
                    fixed_count += 1
                else:
                    # Verificar si ya está correcto
                    if replacements["new"] in content:
                        print(f"✅ Ya corregido: {file_path}")
                    else:
                        print(f"⚠️  No se encontró import RLHF en: {file_path}")
                        
            except Exception as e:
                print(f"❌ Error corrigiendo {file_path}: {e}")
        else:
            print(f"📁 No existe: {file_path}")
    
    return fixed_count

if __name__ == "__main__":
    print("🎯 LIMPIEZA SELECTIVA + CORRECCIÓN RLHF")
    print("=" * 50)
    
    remove_translation_components()
    verify_data_enrichment_remains()
    issues = check_obsolete_imports()
    fixed_imports = fix_rlhf_imports()  # ← NUEVA FUNCIÓN
    
    print("\n" + "=" * 50)
    print(f"🔧 Importaciones RLHF corregidas: {fixed_imports}")
    print(f"⚠️  Archivos con imports obsoletos: {issues}")
    
    if issues == 0 and fixed_imports > 0:
        print("🎉 SISTEMA CORREGIDO - Listo para usar")
    elif issues == 0 and fixed_imports == 0:
        print("✅ Sistema ya está limpio y corregido")
    else:
        print("📝 Algunas correcciones necesarias manuales")
    
    print("\n📝 PRÓXIMOS PASOS:")
    print("   1. Revisar manualmente los imports marcados con ⚠️")
    print("   2. Ejecutar 'python main.py rag' para probar")
    print("   3. Verificar que el enriquecimiento de datos funciona")