"""
Evaluador final CORREGIDO que usa ground truth REAL
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def calcular_precision(ranked_ids, relevant_ids, k=5):
    """Calcula Precision@k"""
    if not relevant_ids or k == 0:
        return 0.0
    top_k = ranked_ids[:k]
    relevant_in_top_k = sum(1 for pid in top_k if pid in relevant_ids)
    return relevant_in_top_k / k

def main():
    print("\n" + "="*80)
    print("🎯 EVALUADOR FINAL - CON GROUND TRUTH REAL")
    print("="*80)
    
    # 1. Cargar sistema con RL
    system_path = Path("data/cache/unified_system_with_fixed_rl.pkl")
    if not system_path.exists():
        print("❌ Sistema no encontrado")
        return
    
    with open(system_path, 'rb') as f:
        system = pickle.load(f)
    
    print(f"✅ Sistema cargado: {len(system.canonical_products):,} productos")
    
    # 2. Cargar ground truth REAL
    ground_truth_files = [
        Path("data/interactions/real_ground_truth.json"),
        Path("data/interactions/sample_ground_truth.json"),
        Path("data/interactions/relevance_labels.json")
    ]
    
    ground_truth = {}
    for gt_file in ground_truth_files:
        if gt_file.exists():
            try:
                with open(gt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # Unificar ground truth
                    for query, ids in data.items():
                        if query not in ground_truth:
                            ground_truth[query] = []
                        for pid in ids:
                            if pid not in ground_truth[query]:
                                ground_truth[query].append(pid)
                    print(f"✅ Cargado: {gt_file.name} ({len(data)} queries)")
            except Exception as e:
                print(f"⚠️  Error cargando {gt_file.name}: {e}")
    
    if not ground_truth:
        print("❌ No se encontró ground truth")
        return
    
    print(f"📊 Total queries con ground truth: {len(ground_truth)}")
    
    # 3. Evaluar cada query
    resultados = []
    queries_evaluadas = 0
    
    for query, relevant_ids in list(ground_truth.items())[:15]:  # Máximo 15
        print(f"\n🔍 Query: '{query}'")
        print(f"   • Productos relevantes: {len(relevant_ids)}")
        
        try:
            # Obtener embedding
            query_embedding = system.canonicalizer.embedding_model.encode(
                query, normalize_embeddings=True
            )
            
            # Baseline: buscar y ordenar por similitud
            baseline_results = system.vector_store.search(query_embedding, k=30)
            
            if not baseline_results:
                print("   ⚠️  No hay resultados")
                continue
            
            # IDs baseline
            baseline_ids = [p.id for p in baseline_results[:10]]
            baseline_precision = calcular_precision(baseline_ids, relevant_ids, 5)
            
            # RL
            rl_ranker = system.rl_ranker
            if hasattr(rl_ranker, 'has_learned') and rl_ranker.has_learned:
                # Calcular scores baseline (similitud)
                baseline_scores = []
                for product in baseline_results:
                    if hasattr(product, 'similarity'):
                        baseline_scores.append(product.similarity)
                    else:
                        baseline_scores.append(0.5)
                
                # Aplicar RL
                rl_results = rl_ranker.rank_products(
                    baseline_results, query, baseline_scores
                )
                rl_ids = [p.id for p in rl_results[:10]]
                rl_precision = calcular_precision(rl_ids, relevant_ids, 5)
                
                # Verificar cambios
                changed = baseline_ids[:5] != rl_ids[:5]
                
                # Relevantes encontrados
                baseline_relevantes = sum(1 for pid in baseline_ids[:5] if pid in relevant_ids)
                rl_relevantes = sum(1 for pid in rl_ids[:5] if pid in relevant_ids)
                
                resultados.append({
                    'query': query,
                    'baseline_p@5': baseline_precision,
                    'rl_p@5': rl_precision,
                    'mejora': rl_precision - baseline_precision,
                    'mejora_porcentual': ((rl_precision - baseline_precision) / baseline_precision * 100) 
                                      if baseline_precision > 0 else 0,
                    'ranking_cambiado': changed,
                    'baseline_relevantes': baseline_relevantes,
                    'rl_relevantes': rl_relevantes,
                    'total_relevantes': len(relevant_ids)
                })
                
                queries_evaluadas += 1
                
                print(f"   📊 Baseline: P@5={baseline_precision:.3f} ({baseline_relevantes}/{len(relevant_ids)})")
                print(f"   🤖 RL:       P@5={rl_precision:.3f} ({rl_relevantes}/{len(relevant_ids)})")
                print(f"   📈 Mejora: {rl_precision - baseline_precision:+.3f}")
                
                if changed:
                    print(f"   🔀 Ranking ajustado")
                    # Mostrar cambios en top 3
                    for i in range(min(3, len(baseline_ids), len(rl_ids))):
                        if baseline_ids[i] != rl_ids[i]:
                            bl_product = next((p for p in baseline_results if p.id == baseline_ids[i]), None)
                            rl_product = next((p for p in rl_results if p.id == rl_ids[i]), None)
                            
                            bl_title = bl_product.title[:40] + "..." if bl_product and hasattr(bl_product, 'title') else "N/A"
                            rl_title = rl_product.title[:40] + "..." if rl_product and hasattr(rl_product, 'title') else "N/A"
                            
                            print(f"     Pos {i+1}: {bl_title} → {rl_title}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # 4. Resultados
    if resultados:
        df = pd.DataFrame(resultados)
        
        print("\n" + "="*80)
        print("📈 RESULTADOS FINALES - RL MEJORADO")
        print("="*80)
        
        # Métricas promedio
        avg_baseline = df['baseline_p@5'].mean()
        avg_rl = df['rl_p@5'].mean()
        avg_mejora = df['mejora'].mean()
        avg_mejora_pct = df['mejora_porcentual'].mean()
        
        print(f"\n📊 PRECISIÓN@5 PROMEDIO:")
        print(f"   • Baseline (solo FAISS): {avg_baseline:.3f}")
        print(f"   • RL mejorado:           {avg_rl:.3f}")
        print(f"   • Mejora absoluta:       {avg_mejora:+.3f}")
        print(f"   • Mejora porcentual:     {avg_mejora_pct:+.1f}%")
        
        # Efectividad
        total_relevantes = df['total_relevantes'].sum()
        baseline_total = df['baseline_relevantes'].sum()
        rl_total = df['rl_relevantes'].sum()
        
        print(f"\n🎯 EFECTIVIDAD EN RECUPERAR RELEVANTES:")
        print(f"   • Total relevantes: {total_relevantes}")
        print(f"   • Baseline recuperó: {baseline_total} ({baseline_total/total_relevantes*100:.1f}%)")
        print(f"   • RL recuperó:       {rl_total} ({rl_total/total_relevantes*100:.1f}%)")
        print(f"   • Diferencia:        {rl_total - baseline_total:+d} relevantes")
        
        # Análisis por queries
        mejoradas = len(df[df['mejora'] > 0])
        iguales = len(df[df['mejora'] == 0])
        empeoradas = len(df[df['mejora'] < 0])
        
        print(f"\n📈 DISTRIBUCIÓN:")
        print(f"   • Queries mejoradas:  {mejoradas}/{len(df)}")
        print(f"   • Queries iguales:    {iguales}/{len(df)}")
        print(f"   • Queries empeoradas: {empeoradas}/{len(df)}")
        
        if mejoradas > 0:
            top_mejoras = df.nlargest(3, 'mejora')
            print(f"\n🏆 TOP 3 MEJORAS:")
            for _, row in top_mejoras.iterrows():
                print(f"   • '{row['query'][:30]}...': {row['baseline_p@5']:.3f} → {row['rl_p@5']:.3f} "
                      f"(+{row['mejora']:.3f}, +{row['mejora_porcentual']:.1f}%)")
        
        # Estadísticas RL
        if hasattr(system.rl_ranker, 'get_stats'):
            rl_stats = system.rl_ranker.get_stats()
            print(f"\n🤖 ESTADÍSTICAS DEL RL:")
            print(f"   • Feedback procesado: {rl_stats.get('feedback_count', 0)}")
            print(f"   • Features aprendidas: {rl_stats.get('weights_count', 0)}")
            print(f"   • Ratio match/rating:  {rl_stats.get('match_vs_rating_ratio', 0):.2f}")
            
            if 'top_features' in rl_stats and rl_stats['top_features']:
                print(f"   🔝 TOP 5 FEATURES:")
                for i, (feature, weight) in enumerate(rl_stats['top_features'][:5], 1):
                    if 'match' in feature.lower():
                        symbol = "🎯"
                    elif 'rating' in feature.lower():
                        symbol = "⭐"
                    elif 'keyword' in feature.lower():
                        symbol = "🔑"
                    else:
                        symbol = "📊"
                    
                    print(f"     {symbol} {feature:25} {weight:7.3f}")
        
        # Guardar
        output_file = "resultados_finales_corregidos.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Resultados guardados en: {output_file}")
        
        # Conclusión
        print(f"\n💡 CONCLUSIÓN:")
        if avg_mejora_pct > 5:
            print(f"  ✅ ¡EXCELENTE! RL mejora {avg_mejora_pct:+.1f}% sobre baseline")
            print(f"  ✨ El aprendizaje por refuerzo FUNCIONA correctamente")
        elif avg_mejora_pct > 0:
            print(f"  ✅ Mejora moderada: {avg_mejora_pct:+.1f}%")
            print(f"  📈 El RL está aprendiendo pero puede mejorar con más feedback")
        elif avg_mejora_pct > -5:
            print(f"  ⚠️  Sin mejora significativa: {avg_mejora_pct:+.1f}%")
            print(f"  🔧 Considerar ajustar parámetros de aprendizaje")
        else:
            print(f"  ❌ Degradación: {avg_mejora_pct:+.1f}%")
            print(f"  🛑 Revisar el entrenamiento del RL")
    
    print("\n" + "="*80)
    print("✅ EVALUACIÓN COMPLETADA")
    print("="*80)

if __name__ == "__main__":
    main()