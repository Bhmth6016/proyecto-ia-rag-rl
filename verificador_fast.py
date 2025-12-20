# verificador_fast.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Configurar paths
current_dir = Path().cwd()
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

print("="*80)
print("📊 EVALUADOR RÁPIDO - CON CACHÉ")
print("="*80)

def main():
    try:
        # Cargar sistema optimizado
        from src.main_optimized import OptimizedRAGRLSystem
        
        print("\n⚡ Inicializando sistema con caché...")
        start_time = datetime.now()
        
        system = OptimizedRAGRLSystem('config/config.yaml', use_cache=True)
        system.initialize_with_cache(force_reload=False)
        
        init_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Sistema listo en {init_time:.1f}s")
        print(f"📊 Productos: {len(system.canonical_products):,}")
        
        # Cargar interacciones previas si existen
        print("\n📖 Buscando interacciones previas...")
        interactions_file = Path("data/interactions/real_interactions.jsonl")
        
        if interactions_file.exists():
            relevance = {}
            with open(interactions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        interaction = json.loads(line.strip())
                        if interaction.get('interaction_type') == 'click':
                            query = interaction['context'].get('query')
                            product_id = interaction['context'].get('product_id')
                            if query and product_id:
                                if query not in relevance:
                                    relevance[query] = []
                                if product_id not in relevance[query]:
                                    relevance[query].append(product_id)
                    except json.JSONDecodeError:
                        continue
            
            if relevance:
                print(f"✅ {len(relevance)} queries con clicks encontradas")
                queries_to_test = list(relevance.keys())[:10]
                
                print(f"\n🔍 Evaluando {len(queries_to_test)} queries...")
                
                # Evaluar modos
                modes = [
                    ("Baseline", "baseline"),
                    ("RAG+Features", "with_features"), 
                    ("RAG+RLHF", "with_rlhf")
                ]
                
                resultados = {nombre: [] for nombre, _ in modes}
                
                for query_idx, query in enumerate(queries_to_test):
                    print(f"\n   Query {query_idx+1}/{len(queries_to_test)}: '{query}'")
                    
                    for mode_name, mode in modes:
                        try:
                            # Procesar query
                            response = system.process_query(query, use_rlhf=(mode=="with_rlhf"))
                            
                            if response.get('success'):
                                ranked_products = [p.get('id') for p in response.get('products', [])]
                                relevant_ids = relevance.get(query, [])
                                
                                if relevant_ids:
                                    # Calcular métricas
                                    top_5 = ranked_products[:5]
                                    relevant_in_top_5 = [pid for pid in top_5 if pid in relevant_ids]
                                    precision_at_5 = len(relevant_in_top_5) / 5.0 if top_5 else 0
                                    
                                    recall_at_5 = len(relevant_in_top_5) / len(relevant_ids) if relevant_ids else 0
                                    
                                    mrr = 0
                                    for i, pid in enumerate(ranked_products[:10]):
                                        if pid in relevant_ids:
                                            mrr = 1.0 / (i + 1)
                                            break
                                    
                                    resultados[mode_name].append({
                                        'precision@5': precision_at_5,
                                        'recall@5': recall_at_5,
                                        'mrr': mrr,
                                        'has_ground_truth': True
                                    })
                                    
                                    print(f"     ✅ {mode_name}: P@5={precision_at_5:.3f}, R@5={recall_at_5:.3f}")
                                    
                        except Exception as e:
                            print(f"     ❌ {mode_name}: Error")
                
                # Calcular estadísticas
                print("\n" + "="*80)
                print("📋 RESULTADOS")
                print("="*80)
                
                tabla = []
                for mode_name, metrics_list in resultados.items():
                    if metrics_list:
                        valid_metrics = [m for m in metrics_list if m.get('has_ground_truth', False)]
                        if valid_metrics:
                            precision_scores = [m.get('precision@5', 0) for m in valid_metrics]
                            recall_scores = [m.get('recall@5', 0) for m in valid_metrics]
                            mrr_scores = [m.get('mrr', 0) for m in valid_metrics]
                            
                            tabla.append({
                                'Modo': mode_name,
                                'Queries': len(valid_metrics),
                                'Precision@5_mean': np.mean(precision_scores),
                                'Recall@5_mean': np.mean(recall_scores),
                                'MRR_mean': np.mean(mrr_scores)
                            })
                
                if tabla:
                    df = pd.DataFrame(tabla)
                    print(df.to_string(index=False))
                    
                    # Guardar resultados
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                    df.to_csv(f'resultados_rapidos_{timestamp}.csv', index=False, encoding='utf-8')
                    print(f"\n💾 Resultados guardados: resultados_rapidos_{timestamp}.csv")
                    
                    # Crear gráfica simple
                    plt.figure(figsize=(10, 6))
                    modos = [r['Modo'] for r in tabla]
                    precision = [r['Precision@5_mean'] for r in tabla]
                    
                    plt.bar(modos, precision, color=['blue', 'green', 'red'])
                    plt.title('Precision@5 por Modo - Evaluación Rápida')
                    plt.ylabel('Precision@5')
                    plt.ylim(0, 1)
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(f'grafica_rapida_{timestamp}.png', dpi=300)
                    print(f"📈 Gráfica guardada: grafica_rapida_{timestamp}.png")
                    
                else:
                    print("❌ No hay métricas válidas")
                
            else:
                print("⚠️  No hay interacciones con clicks")
                print("\n💡 Instrucciones:")
                print("   1. Ejecuta: python run_fast_interactive.py")
                print("   2. Haz búsquedas y clicks")
                print("   3. Vuelve a ejecutar este evaluador")
        
        else:
            print("❌ No hay archivo de interacciones")
            print("\n💡 Primero ejecuta el sistema interactivo")
        
        print("\n" + "="*80)
        print("🎉 EVALUACIÓN COMPLETADA")
        print("="*80)
        
    except ImportError as e:
        print(f"\n❌ Error de importación: {e}")
        print("💡 Asegúrate de que los módulos están disponibles")
    except Exception as e:
        print(f"\n❌ Error general: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()