# debug_ranking_comparison.py - VERSIÓN DEFINITIVA
"""
Compara rankings de todos los métodos
"""

from src.unified_system_v2 import UnifiedSystemV2
import json

system = UnifiedSystemV2.load_from_cache()

# Cargar ground truth
print("📂 Cargando ground truth...")
with open('data/interactions/ground_truth_REAL.json', 'r') as f:
    gt_data = json.load(f)

# El formato es: {"query": [product_ids]}
print(f"   Formato: Dict de queries → product IDs")
print(f"   Queries totales: {len(gt_data)}\n")

query = '"pc gaming"'  # Con comillas porque así está en el ground truth
print(f"Query: {query}")
print("=" * 70)

# Buscar relevantes
relevant_ids = set(gt_data.get(query, []))
if relevant_ids:
    print(f"Productos relevantes: {relevant_ids}\n")
else:
    print(f"⚠️  No hay ground truth para esta query\n")
    print("Queries disponibles:")
    for q in list(gt_data.keys())[:5]:
        print(f"  - {q}")
    print()

# Baseline
print("🔵 BASELINE:")
baseline_results = system._process_query_baseline(query, k=20)
for i, prod in enumerate(baseline_results[:10]):
    prod_id = getattr(prod, 'id', 'N/A')
    title = getattr(prod, 'title', 'N/A')[:50]
    is_relevant = "✅" if prod_id in relevant_ids else "  "
    print(f"  {i+1}. {is_relevant} [{prod_id}] {title}")

# Todos los métodos
all_results = system.query_four_methods(query, k=20)

if 'methods' in all_results:
    methods_dict = all_results['methods']
    
    # NER-Enhanced
    if 'ner_enhanced' in methods_dict:
        print("\n🟢 NER-ENHANCED:")
        ner_results = methods_dict['ner_enhanced']
        for i, prod in enumerate(ner_results[:10]):
            prod_id = getattr(prod, 'id', 'N/A')
            title = getattr(prod, 'title', 'N/A')[:50]
            attrs = getattr(prod, 'ner_attributes', {})
            is_relevant = "✅" if prod_id in relevant_ids else "  "
            print(f"  {i+1}. {is_relevant} [{prod_id}] {title}")
            if attrs:
                print(f"       NER: {list(attrs.keys())}")
    
    # RLHF
    rlhf_key = next((k for k in ['rlhf', 'rl', 'rl_ranker'] if k in methods_dict), None)
    if rlhf_key:
        print(f"\n🔴 RLHF (key='{rlhf_key}'):")
        rlhf_results = methods_dict[rlhf_key]
        for i, prod in enumerate(rlhf_results[:10]):
            prod_id = getattr(prod, 'id', 'N/A')
            title = getattr(prod, 'title', 'N/A')[:50]
            is_relevant = "✅" if prod_id in relevant_ids else "  "
            print(f"  {i+1}. {is_relevant} [{prod_id}] {title}")
    
    # Full Hybrid
    if 'full_hybrid' in methods_dict:
        print("\n🟣 FULL HYBRID:")
        hybrid_results = methods_dict['full_hybrid']
        for i, prod in enumerate(hybrid_results[:10]):
            prod_id = getattr(prod, 'id', 'N/A')
            title = getattr(prod, 'title', 'N/A')[:50]
            is_relevant = "✅" if prod_id in relevant_ids else "  "
            print(f"  {i+1}. {is_relevant} [{prod_id}] {title}")
    
    # Comparaciones
    print("\n📊 COMPARACIÓN DE RANKINGS:")
    baseline_ids = [getattr(p, 'id', '') for p in baseline_results[:10]]
    
    comparisons = [
        ('NER-Enhanced', 'ner_enhanced'),
        ('RLHF', rlhf_key),
        ('Full Hybrid', 'full_hybrid')
    ]
    
    for name, key in comparisons:
        if key and key in methods_dict:
            method_results = methods_dict[key]
            method_ids = [getattr(p, 'id', '') for p in method_results[:10]]
            
            if baseline_ids == method_ids:
                print(f"\n{name} vs Baseline: ❌ IDÉNTICOS")
            else:
                different = sum(1 for i in range(10) if baseline_ids[i] != method_ids[i])
                print(f"\n{name} vs Baseline: ✅ DIFERENTES ({different}/10 posiciones)")

else:
    print(f"\n❌ Estructura inesperada: {list(all_results.keys())}")