# debug_rlhf.py - VERSIÓN DEFINITIVA
"""
Diagnóstico de RLHF
"""

from src.unified_system_v2 import UnifiedSystemV2

system = UnifiedSystemV2.load_from_cache()

rl_ranker = system.rl_ranker
print("=" * 70)
print("🔍 DIAGNÓSTICO RLHF")
print("=" * 70)

if hasattr(rl_ranker, 'feature_weights'):
    weights = rl_ranker.feature_weights
    print(f"\n📊 Features aprendidas: {len(weights)}")
    
    if len(weights) == 0:
        print("⚠️  No hay features aprendidas")
    else:
        # Top 10 pesos
        sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
        print("\n🔝 Top 10 pesos (absolutos):")
        for i, (feat, weight) in enumerate(sorted_weights[:10], 1):
            print(f"   {i}. {feat}: {weight:.4f}")
        
        # Estadísticas
        weight_values = list(weights.values())
        print(f"\n📈 Estadísticas:")
        print(f"   Max peso: {max(weight_values):.4f}")
        print(f"   Min peso: {min(weight_values):.4f}")
        print(f"   Promedio: {sum(weight_values)/len(weight_values):.4f}")
        
        significant = sum(1 for w in weight_values if abs(w) > 0.1)
        print(f"   Pesos significativos (|w| > 0.1): {significant}/{len(weight_values)}")

print(f"\n🔧 Configuración:")
print(f"   Learning rate: {getattr(rl_ranker, 'learning_rate', 'N/A')}")
print(f"   Match/rating balance: {getattr(rl_ranker, 'match_rating_balance', 'N/A')}")
print(f"   Feedback count: {getattr(rl_ranker, 'feedback_count', 'N/A')}")

# Test con query
print("\n" + "=" * 70)
print("🧪 TEST CON QUERY: 'pc gaming'")
print("=" * 70)

query = "pc gaming"

# Baseline
baseline_results = system._process_query_baseline(query, k=10)
print(f"\n🔵 BASELINE:")
for i, prod in enumerate(baseline_results[:5], 1):
    title = getattr(prod, 'title', 'N/A')[:50]
    prod_id = getattr(prod, 'id', 'N/A')
    print(f"   {i}. [{prod_id}] {title}")

# RLHF - FIX: acceder correctamente a la estructura
try:
    all_results = system.query_four_methods(query, k=10)
    
    # La estructura es: {'query': ..., 'methods': {...}, ...}
    if 'methods' in all_results:
        methods = all_results['methods']
        print(f"\n🔍 Métodos disponibles: {list(methods.keys())}")
        
        # Intentar con diferentes nombres
        rlhf_key = None
        for key in ['rlhf', 'rl', 'rl_ranker']:
            if key in methods:
                rlhf_key = key
                break
        
        if rlhf_key:
            rlhf_results = methods[rlhf_key]
            print(f"\n🔴 RLHF (key='{rlhf_key}'):")
            for i, prod in enumerate(rlhf_results[:5], 1):
                title = getattr(prod, 'title', 'N/A')[:50]
                prod_id = getattr(prod, 'id', 'N/A')
                print(f"   {i}. [{prod_id}] {title}")
            
            # Comparar
            baseline_ids = [getattr(p, 'id', '') for p in baseline_results[:5]]
            rlhf_ids = [getattr(p, 'id', '') for p in rlhf_results[:5]]
            
            print("\n📊 COMPARACIÓN:")
            if baseline_ids == rlhf_ids:
                print("❌ Rankings IDÉNTICOS - RLHF NO funcionando")
                print("\n🔍 DIAGNÓSTICO:")
                print("   • RLHF tiene pesos aprendidos ✅")
                print("   • Pero no está cambiando el ranking ❌")
                print("\n💡 POSIBLES CAUSAS:")
                print("   1. Los pesos son demasiado pequeños")
                print("   2. Las features no matchean con los productos")
                print("   3. Bug en aplicación de pesos")
            else:
                print("✅ Rankings DIFERENTES")
                changes = sum(1 for i in range(5) if baseline_ids[i] != rlhf_ids[i])
                print(f"   {changes}/5 posiciones diferentes")
                
                for i in range(5):
                    if baseline_ids[i] != rlhf_ids[i]:
                        print(f"   Pos {i+1}: Baseline={baseline_ids[i]}, RLHF={rlhf_ids[i]}")
        else:
            print(f"❌ No se encontró método RLHF")
            print(f"   Métodos disponibles: {list(methods.keys())}")
    else:
        print(f"❌ Estructura inesperada")
        print(f"   Keys: {list(all_results.keys())}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()