#!/usr/bin/env python3
"""
FIX COMPLETO: Arregla IDs y crea ground truth funcional
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

def load_system():
    """Carga sistema actual"""
    cache_file = Path("data/cache/unified_system_v2.pkl")
    
    if not cache_file.exists():
        print("❌ Sistema no encontrado. Ejecuta: python main.py init")
        return None
    
    with open(cache_file, 'rb') as f:
        system = pickle.load(f)
    
    print(f"✅ Sistema cargado: {len(system.canonical_products):,} productos")
    return system

def build_product_maps(system):
    """Crea mapas para buscar productos"""
    # Mapa completo: ID → Producto
    id_to_product = {}
    
    # Mapas para buscar:
    title_to_ids = defaultdict(list)  # Título → [IDs]
    category_title_to_id = {}  # (Categoría, Título) → ID
    
    for product in system.canonical_products:
        product_id = str(product.id)
        title = getattr(product, 'title', '').lower().strip()
        category = getattr(product, 'category', '').lower().strip()
        
        id_to_product[product_id] = product
        
        if title:
            title_to_ids[title].append(product_id)
            if category:
                category_title_to_id[(category, title)] = product_id
    
    print(f"📊 Mapas creados:")
    print(f"   • {len(id_to_product):,} productos por ID")
    print(f"   • {len(title_to_ids):,} títulos únicos")
    
    return id_to_product, title_to_ids, category_title_to_id

def find_product_id(old_id: str, id_to_product: Dict, 
                   title_to_ids: Dict, system) -> str:
    """Encuentra ID correcto del producto"""
    
    # 1. ID ya es correcto
    if old_id in id_to_product:
        return old_id
    
    # 2. Buscar por índice (meta_Video_Games_10000_554 → índice 554)
    try:
        parts = old_id.split('_')
        if len(parts) >= 2:
            # Extraer categoría e índice
            if parts[0] == 'meta':
                category = parts[1]  # Video, Games, etc
                if len(parts) > 3:
                    # Formato: meta_Video_Games_10000_554
                    idx = int(parts[-1])
                    
                    # Buscar productos de esa categoría
                    for prod_id, product in id_to_product.items():
                        prod_category = getattr(product, 'category', '').lower()
                        if category.lower() in prod_category.replace(' ', ''):
                            # Verificar si coincide con índice aproximado
                            if prod_id.endswith(str(idx)) or str(idx) in prod_id:
                                return prod_id
    except:
        pass
    
    # 3. No encontrado
    return None

def process_ground_truth_real():
    """Procesa ground_truth_REAL.json"""
    gt_file = Path("data/interactions/ground_truth_REAL.json")
    
    if not gt_file.exists():
        print("⚠️ ground_truth_REAL.json no encontrado")
        return {}
    
    with open(gt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📂 Procesando ground_truth_REAL.json: {len(data)} queries")
    return data

def process_user_interactions(id_to_product: Dict, title_to_ids: Dict, system):
    """Procesa user_interactions.jsonl"""
    file_path = Path("data/interactions/user_interactions.jsonl")
    
    if not file_path.exists():
        print("⚠️ user_interactions.jsonl no encontrado")
        return {}
    
    clicks_by_query = defaultdict(list)
    matched = 0
    unmatched = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                
                if data.get('interaction_type') == 'click':
                    context = data.get('context', {})
                    query = context.get('query', '').strip().lower()
                    old_id = context.get('product_id', '')
                    position = context.get('position', 999)
                    
                    if not query or not old_id:
                        continue
                    
                    # Buscar ID correcto
                    new_id = find_product_id(old_id, id_to_product, title_to_ids, system)
                    
                    if new_id:
                        clicks_by_query[query].append({
                            'product_id': new_id,
                            'position': position
                        })
                        matched += 1
                    else:
                        unmatched += 1
                        
            except json.JSONDecodeError:
                continue
    
    # Ordenar por posición
    result = {}
    for query, clicks in clicks_by_query.items():
        clicks_sorted = sorted(clicks, key=lambda x: x['position'])
        result[query] = [c['product_id'] for c in clicks_sorted]
    
    print(f"\n📂 Procesando user_interactions.jsonl:")
    print(f"   ✅ {matched} clicks matched")
    print(f"   ❌ {unmatched} clicks NO matched")
    print(f"   📊 {len(result)} queries únicas")
    
    return result

def merge_and_save(gt_real: Dict, user_int: Dict):
    """Merge y guardar ground truth final"""
    
    merged = defaultdict(set)
    
    print("\n📊 Consolidando...")
    
    # 1. Ground truth real (máxima prioridad)
    for query, products in gt_real.items():
        query_norm = query.strip().lower()
        merged[query_norm].update(products)
        print(f"   ✅ '{query}': {len(products)} productos (ground_truth_REAL)")
    
    # 2. User interactions
    for query, products in user_int.items():
        query_norm = query.strip().lower()
        if query_norm not in merged:
            merged[query_norm].update(products)
            print(f"   ✅ '{query}': {len(products)} productos (user_interactions)")
        else:
            before = len(merged[query_norm])
            merged[query_norm].update(products)
            added = len(merged[query_norm]) - before
            if added > 0:
                print(f"   ➕ '{query}': +{added} productos adicionales")
    
    # Convertir a lista
    result = {query: list(products) for query, products in merged.items()}
    
    # Guardar
    gt_dir = Path("data/ground_truth")
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    gt_file = gt_dir / "ground_truth.jsonl"
    
    # Backup
    if gt_file.exists():
        backup = gt_dir / "ground_truth_backup_fixed.jsonl"
        with open(gt_file, 'r') as f:
            content = f.read()
        with open(backup, 'w') as f:
            f.write(content)
        print(f"\n💾 Backup: {backup}")
    
    # Guardar nuevo
    with open(gt_file, 'w', encoding='utf-8') as f:
        for query, products in sorted(result.items()):
            entry = {
                'query': query,
                'relevant_products': products
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Ground truth guardado: {gt_file}")
    print(f"   • {len(result)} queries únicas")
    print(f"   • {sum(len(p) for p in result.values())} productos relevantes totales")
    
    # Resumen
    print(f"\n📋 Resumen por query:")
    for query, products in sorted(result.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"   • '{query}': {len(products)} productos")
    
    return gt_file

def main():
    print("=" * 80)
    print("🔧 FIX COMPLETO: ARREGLANDO IDs Y GROUND TRUTH")
    print("=" * 80)
    
    # 1. Cargar sistema
    system = load_system()
    if not system:
        return
    
    # 2. Crear mapas
    id_to_product, title_to_ids, category_title_to_id = build_product_maps(system)
    
    # 3. Procesar ground truth real
    gt_real = process_ground_truth_real()
    
    # 4. Procesar user interactions
    user_int = process_user_interactions(id_to_product, title_to_ids, system)
    
    # 5. Merge y guardar
    if gt_real or user_int:
        gt_file = merge_and_save(gt_real, user_int)
        
        print("\n" + "=" * 80)
        print("✅ FIX COMPLETADO")
        print("=" * 80)
        print(f"\n🔄 Ahora ejecuta:")
        print(f"   python main.py experimento")
    else:
        print("\n❌ No se encontraron datos para consolidar")
        print("   Verifica que existen:")
        print("   • data/interactions/ground_truth_REAL.json")
        print("   • data/interactions/user_interactions.jsonl")

if __name__ == "__main__":
    main()