# evaluador_avanzado_rlhf.py
"""
Evaluador AVANZADO para RLHF - Mide mejora REAL en posicionamiento
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.WARNING)

def calcular_metricas_avanzadas(ranked_ids, relevant_ids, k=10):
    """Calcula métricas avanzadas de ranking"""
    if not relevant_ids:
        return {
            'precision@k': 0,
            'recall@k': 0,
            'mrr': 0,
            'avg_position': float('inf'),
            'ndcg@k': 0,
            'relevant_in_top_k': 0
        }
    
    # Precision@k
    top_k = ranked_ids[:k]
    relevant_in_top_k = [pid for pid in top_k if pid in relevant_ids]
    precision_at_k = len(relevant_in_top_k) / k if top_k else 0
    
    # Recall@k
    recall_at_k = len(relevant_in_top_k) / len(relevant_ids)
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0
    for i, pid in enumerate(ranked_ids):
        if pid in relevant_ids:
            mrr = 1.0 / (i + 1)
            break
    
    # Average Position (cuanto más bajo, mejor)
    positions = []
    for pid in relevant_ids:
        if pid in ranked_ids:
            positions.append(ranked_ids.index(pid) + 1)  # +1 porque empieza en 0
    avg_position = np.mean(positions) if positions else float('inf')
    
    # NDCG@k
    dcg = 0.0
    for i, pid in enumerate(ranked_ids[:k]):
        if pid in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)  # i+2 porque log2(1)=0
    
    # Ideal DCG
    ideal_relevance = [1] * min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(len(ideal_relevance)))
    ndcg_at_k = dcg / idcg if idcg > 0 else 0
    
    return {
        'precision@k': precision_at_k,
        'recall@k': recall_at_k,
        'mrr': mrr,
        'avg_position': avg_position,
        'ndcg@k': ndcg_at_k,
        'relevant_in_top_k': len(relevant_in_top_k)
    }

def main():
    print("\n" + "="*80)
    print("🎯 EVALUADOR AVANZADO RLHF - MEJORA EN POSICIONAMIENTO")
    print("="*80)
    
    # 1. Cargar sistema con TRUE RLHF
    system_path = Path("data/cache/unified_system_with_TRUE_rl.pkl")
    if not system_path.exists():
        print("❌ Sistema con TRUE RLHF no encontrado")
        return
    
    with open(system_path, 'rb') as f:
        system = pickle.load(f)
    
    print(f"✅ Sistema TRUE RLHF cargado: {len(system.canonical_products):,} productos")
    
    # 2. Cargar ground truth REAL
    ground_truth_file = Path("data/interactions/ground_truth_REAL.json")
    
    if not ground_truth_file.exists():
        print("❌ No hay ground truth REAL")
        return
    
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    print(f"✅ Ground truth REAL: {len(ground_truth)} queries")
    
    # 3. Filtrar queries con datos
    valid_queries = {q: ids for q, ids in ground_truth.items() if len(ids) >= 1}
    print(f"📊 Queries válidas para evaluación: {len(valid_queries)}")
    
    # 4. Evaluar con métricas avanzadas
    resultados = []
    
    for query, relevant_ids in list(valid_queries.items())[:50]:  # Máximo 50
        try:
            # Obtener embedding
            query_embedding = system.canonicalizer.embedding_model.encode(
                query, normalize_embeddings=True
            )
            
            # Baseline
            baseline_results = system.vector_store.search(query_embedding, k=30)
            
            if not baseline_results:
                continue
            
            # Calcular scores baseline
            baseline_scores = []
            for product in baseline_results:
                if hasattr(product, 'content_embedding'):
                    prod_emb = product.content_embedding
                    prod_norm = prod_emb / np.linalg.norm(prod_emb)
                    query_norm = query_embedding / np.linalg.norm(query_embedding)
                    score = float(np.dot(query_norm, prod_norm))
                    baseline_scores.append(score)
                else:
                    baseline_scores.append(0.5)
            
            # MÉTODO 1: BASELINE
            baseline_ids = [p.id for p in baseline_results[:10]]
            baseline_metrics = calcular_metricas_avanzadas(baseline_ids, relevant_ids, k=5)
            
            # MÉTODO 2: TRUE RLHF
            rl_ids = baseline_ids[:]
            rl_metrics = baseline_metrics.copy()
            
            if hasattr(system, 'rl_ranker') and system.rl_ranker:
                rl_results = system.rl_ranker.rank_products(
                    baseline_results, query, baseline_scores
                )
                rl_ids = [p.id for p in rl_results[:10]]
                rl_metrics = calcular_metricas_avanzadas(rl_ids, relevant_ids, k=5)
            
            # Calcular mejoras
            mejora_posicion = baseline_metrics['avg_position'] - rl_metrics['avg_position']  # Positiva si mejora
            mejora_ndcg = rl_metrics['ndcg@k'] - baseline_metrics['ndcg@k']
            mejora_mrr = rl_metrics['mrr'] - baseline_metrics['mrr']
            
            # ¿Mejoró el posicionamiento?
            mejor_posicionamiento = mejora_posicion > 0  # Disminuyó la posición promedio
            
            # Guardar resultados
            resultados.append({
                'query': query,
                'total_relevantes': len(relevant_ids),
                
                # Baseline
                'baseline_p@5': baseline_metrics['precision@k'],
                'baseline_avg_pos': baseline_metrics['avg_position'],
                'baseline_ndcg': baseline_metrics['ndcg@k'],
                'baseline_mrr': baseline_metrics['mrr'],
                'baseline_found': baseline_metrics['relevant_in_top_k'],
                
                # RLHF
                'rlhf_p@5': rl_metrics['precision@k'],
                'rlhf_avg_pos': rl_metrics['avg_position'],
                'rlhf_ndcg': rl_metrics['ndcg@k'],
                'rlhf_mrr': rl_metrics['mrr'],
                'rlhf_found': rl_metrics['relevant_in_top_k'],
                
                # Mejoras
                'mejora_p@5': rl_metrics['precision@k'] - baseline_metrics['precision@k'],
                'mejora_posicion': mejora_posicion,  # Positiva = mejoró
                'mejora_ndcg': mejora_ndcg,
                'mejora_mrr': mejora_mrr,
                
                # Análisis
                'mejor_precision': rl_metrics['precision@k'] > baseline_metrics['precision@k'],
                'mejor_posicion': mejor_posicionamiento,
                'mejor_ndcg': mejora_ndcg > 0,
                'mejor_mrr': mejora_mrr > 0,
                
                # Cambios específicos
                'cambio_top1': baseline_ids[0] != rl_ids[0] if baseline_ids and rl_ids else False,
                'cambio_top3': baseline_ids[:3] != rl_ids[:3] if len(baseline_ids) >=3 and len(rl_ids) >= 3 else False,
            })
            
            # Mostrar si hay mejora en posicionamiento
            if abs(mejora_posicion) > 0.5:  # Cambio significativo en posición
                if mejora_posicion > 0:
                    print(f"   📈 '{query[:30]}...': Posición mejoró {baseline_metrics['avg_position']:.1f} → {rl_metrics['avg_position']:.1f} (-{mejora_posicion:.1f})")
                else:
                    print(f"   📉 '{query[:30]}...': Posición empeoró {baseline_metrics['avg_position']:.1f} → {rl_metrics['avg_position']:.1f} ({mejora_posicion:.1f})")
            
        except Exception as e:
            continue
    
    # 5. Resultados finales
    if resultados:
        df = pd.DataFrame(resultados)
        
        print("\n" + "="*80)
        print("📈 RESULTADOS AVANZADOS - TRUE RLHF")
        print("="*80)
        
        # Métricas promedio
        print(f"\n📊 MÉTRICAS PROMEDIO:")
        print(f"   Método           P@5     AvgPos   NDCG@5  MRR     Found")
        print(f"   {'-'*60}")
        print(f"   Baseline         {df['baseline_p@5'].mean():.3f}   {df['baseline_avg_pos'].mean():.1f}     {df['baseline_ndcg'].mean():.3f}   {df['baseline_mrr'].mean():.3f}   {df['baseline_found'].sum()}/{df['total_relevantes'].sum()}")
        print(f"   TRUE RLHF        {df['rlhf_p@5'].mean():.3f}   {df['rlhf_avg_pos'].mean():.1f}     {df['rlhf_ndcg'].mean():.3f}   {df['rlhf_mrr'].mean():.3f}   {df['rlhf_found'].sum()}/{df['total_relevantes'].sum()}")
        print(f"   {'-'*60}")
        
        # Mejoras promedio
        print(f"\n📈 MEJORA PROMEDIO:")
        print(f"   • Precision@5:    {df['mejora_p@5'].mean():+.3f}")
        print(f"   • Posición avg:   {df['mejora_posicion'].mean():+.1f} (↓ es mejor)")
        print(f"   • NDCG@5:         {df['mejora_ndcg'].mean():+.3f}")
        print(f"   • MRR:            {df['mejora_mrr'].mean():+.3f}")
        
        # Análisis de qué mejoró
        print(f"\n📊 ¿QUÉ MEJORÓ CON RLHF?")
        print(f"   • Precision:      {df['mejor_precision'].sum()}/{len(df)} ({df['mejor_precision'].sum()/len(df)*100:.1f}%)")
        print(f"   • Posicionamiento:{df['mejor_posicion'].sum()}/{len(df)} ({df['mejor_posicion'].sum()/len(df)*100:.1f}%)")
        print(f"   • NDCG:           {df['mejor_ndcg'].sum()}/{len(df)} ({df['mejor_ndcg'].sum()/len(df)*100:.1f}%)")
        print(f"   • MRR:            {df['mejor_mrr'].sum()}/{len(df)} ({df['mejor_mrr'].sum()/len(df)*100:.1f}%)")
        
        # Cambios específicos
        print(f"\n🎯 CAMBIOS EN RANKING:")
        print(f"   • Top 1 cambió:   {df['cambio_top1'].sum()}/{len(df)} ({df['cambio_top1'].sum()/len(df)*100:.1f}%)")
        print(f"   • Top 3 cambió:   {df['cambio_top3'].sum()}/{len(df)} ({df['cambio_top3'].sum()/len(df)*100:.1f}%)")
        
        # Análisis RLHF
        if hasattr(system, 'rl_ranker') and system.rl_ranker:
            stats = system.rl_ranker.get_stats()
            
            print(f"\n🤖 ANÁLISIS RLHF:")
            print(f"   • Feedback: {stats['feedback_count']}")
            print(f"   • Ratio match/rating: {stats.get('match_vs_rating_ratio', 0):.2f}")
            
            # Distribución de features
            if 'feature_type_distribution' in stats:
                print(f"   📊 Distribución de aprendizaje:")
                for tipo, peso in stats['feature_type_distribution'].items():
                    print(f"      • {tipo}: {peso:.2f}")
        
        # CONCLUSIÓN DETALLADA
        print(f"\n💡 ANÁLISIS DETALLADO:")
        
        mejora_posicion_promedio = df['mejora_posicion'].mean()
        mejora_ndcg_promedio = df['mejora_ndcg'].mean()
        
        if mejora_posicion_promedio > 0.5:
            print(f"  🎉 ¡EXCELENTE! RLHF MEJORA SIGNIFICATIVAMENTE EL POSICIONAMIENTO")
            print(f"  ✅ Los productos relevantes suben {mejora_posicion_promedio:.1f} posiciones en promedio")
        elif mejora_posicion_promedio > 0.2:
            print(f"  ✅ MEJORA MODERADA EN POSICIONAMIENTO: +{mejora_posicion_promedio:.1f} posiciones")
            print(f"  📈 RLHF está aprendiendo a priorizar productos relevantes")
        elif mejora_posicion_promedio > 0:
            print(f"  ⚠️  MEJORA MÍNIMA EN POSICIONAMIENTO: +{mejora_posicion_promedio:.1f} posiciones")
            print(f"  🔧 RLHF está aprendiendo pero necesita más datos")
        elif mejora_ndcg_promedio > 0.05:
            print(f"  ✅ RLHF MEJORA LA CALIDAD DEL RANKING (NDCG: +{mejora_ndcg_promedio:.3f})")
            print(f"  📊 Aunque no cambie posiciones, el orden es mejor")
        else:
            print(f"  ⚠️  SIN MEJORA SIGNIFICATIVA EN MÉTRICAS AVANZADAS")
            print(f"  🔧 Considerar: 1) Más datos, 2) Ajustar balance match/rating")
        
        # Guardar
        output_file = "resultados_AVANZADOS_RLHF.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Resultados guardados en: {output_file}")
    
    else:
        print("\n❌ No se pudieron evaluar queries")
    
    print("\n" + "="*80)
    print("✅ EVALUACIÓN AVANZADA COMPLETADA")
    print("="*80)

if __name__ == "__main__":
    main()