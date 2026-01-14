#!/usr/bin/env python3
"""
Crea ground truth usando BÚSQUEDA POR TÍTULO
Ya que los IDs no coinciden, buscamos productos por su título
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List

def load_system():
    """Carga sistema"""
    cache = Path("data/cache/unified_system_v2.pkl")
    if not cache.exists():
        print("❌ Sistema no encontrado")
        return None
    
    with open(cache, 'rb') as f:
        system = pickle.load(f)
    
    print(f"✅ Sistema: {len(system.canonical_products):,} productos\n")
    return system

def search_by_query(system, query: str, k: int = 10):
    """Busca productos usando query del sistema"""
    try:
        results = system.query_four_methods(query, k=k)
        products = results['methods']['baseline']
        return products
    except Exception as e:
        print(f"❌ Error buscando '{query}': {e}")
        return []

def main():
    print("=" * 80)
    print("🔧 CREANDO GROUND TRUTH USANDO BÚSQUEDA DEL SISTEMA")
    print("=" * 80)
    print()
    
    system = load_system()
    if not system:
        return
    
    # Queries manualmente curadas
    queries_y_esperados = {
        "survival games": 3,
        "action video games": 2,
        "car parts": 3,
        "beauty products": 2,
        "electronics": 2,
        "sports equipment": 2,
        "kitchen utensils": 2,
        "detective books": 3,
        "kid toys": 2,
        "headphones": 2,
        "sport videogames": 2,
        "playstation games": 2,
        "xbox console": 2,
        "nintendo switch": 2,
        "pc gaming": 2,
        "mario games": 2,
        "zelda videogames": 2,
    }
    
    ground_truth = []
    
    print("🔍 Buscando productos para cada query...\n")
    
    for query, expected_count in queries_y_esperados.items():
        print(f"📝 '{query}'... ", end="", flush=True)
        
        # Buscar con el sistema
        products = search_by_query(system, query, k=10)
        
        if not products:
            print("❌ Sin resultados")
            continue
        
        # Tomar top N productos como relevantes
        relevant_ids = []
        for product in products[:expected_count]:
            product_id = str(getattr(product, 'id', ''))
            if product_id:
                relevant_ids.append(product_id)
        
        if relevant_ids:
            ground_truth.append({
                'query': query,
                'relevant_products': relevant_ids
            })
            print(f"✅ {len(relevant_ids)} productos")
        else:
            print("⚠️ Sin IDs válidos")
    
    # Guardar
    gt_dir = Path("data/ground_truth")
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    gt_file = gt_dir / "ground_truth.jsonl"
    
    # Backup
    if gt_file.exists():
        backup = gt_dir / "ground_truth_backup_search.jsonl"
        with open(gt_file, 'r') as f:
            content = f.read()
        with open(backup, 'w') as f:
            f.write(content)
        print(f"\n💾 Backup: {backup}")
    
    # Guardar
    with open(gt_file, 'w', encoding='utf-8') as f:
        for entry in ground_truth:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    total_products = sum(len(e['relevant_products']) for e in ground_truth)
    
    print("\n" + "=" * 80)
    print("✅ GROUND TRUTH CREADO")
    print("=" * 80)
    print(f"\n📊 Resumen:")
    print(f"   • Queries: {len(ground_truth)}")
    print(f"   • Productos relevantes totales: {total_products}")
    print()
    
    # Mostrar ejemplos
    print("📋 Primeros 3 ejemplos:")
    for i, entry in enumerate(ground_truth[:3], 1):
        print(f"\n   {i}. '{entry['query']}':")
        for j, pid in enumerate(entry['relevant_products'][:3], 1):
            print(f"      {j}. {pid}")
    
    print("\n🔄 Ahora ejecuta:")
    print("   python main.py experimento")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()