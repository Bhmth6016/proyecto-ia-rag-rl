# diagnose_ids.py
#!/usr/bin/env python3
import json
from pathlib import Path

def diagnose_id_mismatch():
    print("🔍 DIAGNÓSTICO DE ID MISMATCH")
    print("="*60)
    
    # 1. Revisar ground truth
    gt_file = Path("data/interactions/ground_truth_REAL.json")
    if gt_file.exists():
        with open(gt_file, 'r') as f:
            ground_truth = json.load(f)
        
        print("\n📊 GROUND TRUTH:")
        for query, ids in ground_truth.items():
            print(f"  '{query[:30]}...' → {len(ids)} productos: {ids[:3]}")
    
    # 2. Revisar interacciones
    interactions_file = Path("data/interactions/real_interactions.jsonl")
    if interactions_file.exists():
        print("\n📝 INTERACCIONES:")
        unique_ids = set()
        with open(interactions_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data.get('interaction_type') == 'click':
                    product_id = data.get('context', {}).get('product_id')
                    query = data.get('context', {}).get('query', '')
                    position = data.get('context', {}).get('position', 0)
                    if product_id:
                        unique_ids.add(product_id)
                        print(f"  Click: '{query[:20]}...' → {product_id} (pos {position})")
        
        print(f"\n  Total IDs únicos en clicks: {len(unique_ids)}")
        print(f"  IDs: {list(unique_ids)}")
    
    # 3. Revisar productos canonizados
    cache_file = Path("data/cache/unified_system_v2.pkl")
    if cache_file.exists():
        import pickle
        with open(cache_file, 'rb') as f:
            system = pickle.load(f)
        
        print("\n📦 PRODUCTOS CANONIZADOS:")
        print(f"  Total: {len(system.canonical_products)}")
        
        # Mostrar algunos IDs de ejemplo
        print("  Ejemplos de IDs (primeros 5):")
        for i, prod in enumerate(system.canonical_products[:5]):
            if hasattr(prod, 'id'):
                print(f"    {i+1}. {prod.id}")
        
        # Crear set de todos los IDs
        all_ids = set()
        for prod in system.canonical_products:
            if hasattr(prod, 'id'):
                all_ids.add(prod.id)
        
        print(f"\n  Total IDs únicos en sistema: {len(all_ids)}")
        
        # 4. Verificar match
        if 'unique_ids' in locals() and unique_ids:
            matches = unique_ids.intersection(all_ids)
            print(f"\n✅ MATCH ENCONTRADO: {len(matches)}/{len(unique_ids)} IDs coinciden")
            if matches:
                print(f"   IDs coincidentes: {list(matches)}")
            else:
                print("   ⚠️ NINGÚN ID COINCIDE!")
                print("\n   Posibles problemas:")
                print("   1. Los productos clickeados no están en la base de datos")
                print("   2. Los IDs son diferentes (ej: 'B0012345' vs 'B00123456')")
                print("   3. Los productos fueron filtrados durante canonicalización")

if __name__ == "__main__":
    diagnose_id_mismatch()