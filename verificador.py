# verificador.py - VERSIÓN CORREGIDA
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import traceback

# Configurar paths
current_dir = Path().cwd()
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

print("="*80)
print("📊 EVALUACIÓN COMPARATIVA DE 3 MODOS - VERSIÓN CORREGIDA")
print("="*80)

def load_all_products():
    """Carga TODOS los productos de todos los archivos"""
    all_products = []
    data_dir = current_dir / "data" / "raw"
    
    files_to_load = [
        "meta_Automotive_10000.jsonl",
        "meta_Beauty_and_Personal_Care_10000.jsonl",
        "meta_Books_10000.jsonl",
        "meta_Clothing_Shoes_and_Jewelry_10000.jsonl", 
        "meta_Electronics_10000.jsonl",
        "meta_Home_and_Kitchen_10000.jsonl",
        "meta_Sports_and_Outdoors_10000.jsonl",
        "meta_Toys_and_Games_10000.jsonl",
        "meta_Video_Games_10000.jsonl"
    ]
    
    print(f"\n📂 Buscando archivos en: {data_dir}")
    
    for filename in files_to_load:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"   📄 Procesando {filename}...")
            count = 0
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            product = json.loads(line.strip())
                            if isinstance(product, dict):
                                # Añadir identificador si no existe
                                if 'id' not in product:
                                    product['id'] = f"{filename}_{count}"
                                all_products.append(product)
                                count += 1
                        except json.JSONDecodeError:
                            continue
                print(f"   ✅ {count} productos cargados")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"   ⚠️  Archivo no encontrado: {filename}")
    
    return all_products

try:
    # 1. Cargar TODO el dataset CORRECTAMENTE
    print("\n1️⃣  CARGANDO DATASET COMPLETO...")
    raw_products = load_all_products()
    print(f"   ✅ {len(raw_products):,} productos cargados en total")
    
    if len(raw_products) == 0:
        print("\n❌ ERROR: No se pudieron cargar productos")
        print("💡 Asegúrate de que los archivos JSONL están en data/raw/")
        sys.exit(1)
    
    # 2. Inicializar sistema
    print("\n2️⃣  INICIALIZANDO SISTEMA...")
    from src.main import RAGRLSystem
    
    system = RAGRLSystem('config/config.yaml')
    system.initialize_system(raw_products)
    print(f"   ✅ Sistema con {len(system.canonical_products):,} productos canonizados")
    
    # 3. Verificar si existe el interaction logger
    print("\n3️⃣  VERIFICANDO LOGS DE INTERACCIÓN...")
    
    # Crear logger simple si no existe
    class SimpleInteractionLogger:
        def __init__(self):
            self.interactions = []
            
        def get_relevance_labels(self):
            # Simular algunos clicks para testing
            return {
                "car parts": ["prod_1", "prod_3", "prod_5"],
                "led lights": ["prod_2", "prod_4"],
                "wireless headphones": ["prod_1", "prod_2", "prod_6"],
                "laptop for programming": ["prod_3"],
                "running shoes": ["prod_1", "prod_2"]
            }
    
    interaction_logger = SimpleInteractionLogger()
    relevance = interaction_logger.get_relevance_labels()
    queries_con_clicks = list(relevance.keys())
    
    print(f"   📝 {len(queries_con_clicks)} queries con clicks simulados")
    
    # 4. Evaluar cada modo
    print("\n4️⃣  EVALUANDO 3 MODOS...")
    modes = [
        ("Baseline", "baseline"),
        ("RAG+Features", "with_features"), 
        ("RAG+RLHF", "with_rlhf")
    ]
    
    resultados = {nombre: [] for nombre, _ in modes}
    
    for query_idx, query in enumerate(queries_con_clicks[:10]):  # 10 queries máximo
        print(f"\n   🔍 Query {query_idx+1}/{min(10, len(queries_con_clicks))}: '{query}'")
        
        for mode_name, mode in modes:
            try:
                # Procesar query
                response = system._process_query_mode(query, mode)
                
                if response.get('success'):
                    # Extraer productos rankeados
                    ranked_products = [p.get('id') for p in response.get('products', [])]
                    
                    # Calcular métricas simples
                    relevant_ids = relevance.get(query, [])
                    if relevant_ids:
                        # Precision@5
                        top_5 = ranked_products[:5]
                        relevant_in_top_5 = [pid for pid in top_5 if pid in relevant_ids]
                        precision_at_5 = len(relevant_in_top_5) / 5.0 if top_5 else 0
                        
                        # Recall@5
                        recall_at_5 = len(relevant_in_top_5) / len(relevant_ids) if relevant_ids else 0
                        
                        # MRR
                        mrr = 0
                        for i, pid in enumerate(ranked_products[:10]):
                            if pid in relevant_ids:
                                mrr = 1.0 / (i + 1)
                                break
                        
                        metrics = {
                            'precision@5': precision_at_5,
                            'recall@5': recall_at_5,
                            'mrr': mrr,
                            'has_ground_truth': True,
                            'ranked_count': len(ranked_products)
                        }
                        
                        resultados[mode_name].append(metrics)
                        print(f"     ✅ {mode_name}: P@5={precision_at_5:.3f}, R@5={recall_at_5:.3f}, MRR={mrr:.3f}")
                        
                    else:
                        print(f"     ⚠️  {mode_name}: No hay ground truth")
                        
            except Exception as e:
                print(f"     ❌ {mode_name}: Error - {str(e)[:50]}...")
    
    # 5. Calcular estadísticas
    print("\n5️⃣  CALCULANDO ESTADÍSTICAS...")
    
    tabla_comparativa = []
    for mode_name, metrics_list in resultados.items():
        if metrics_list:
            valid_metrics = [m for m in metrics_list if m.get('has_ground_truth', False)]
            
            if valid_metrics:
                precision_scores = [m.get('precision@5', 0) for m in valid_metrics]
                recall_scores = [m.get('recall@5', 0) for m in valid_metrics]
                mrr_scores = [m.get('mrr', 0) for m in valid_metrics]
                
                tabla_comparativa.append({
                    'Modo': mode_name,
                    'Queries': len(valid_metrics),
                    'Precision@5_mean': np.mean(precision_scores),
                    'Precision@5_std': np.std(precision_scores),
                    'Recall@5_mean': np.mean(recall_scores),
                    'MRR_mean': np.mean(mrr_scores)
                })
    
    # 6. Mostrar resultados
    print("\n" + "="*80)
    print("📋 TABLA COMPARATIVA - RESULTADOS")
    print("="*80)
    
    if tabla_comparativa:
        df = pd.DataFrame(tabla_comparativa)
        
        # Calcular mejoras
        if 'Baseline' in df['Modo'].values:
            baseline_precision = df[df['Modo'] == 'Baseline']['Precision@5_mean'].values[0]
            
            for idx, row in df.iterrows():
                if row['Modo'] != 'Baseline':
                    mejora = ((row['Precision@5_mean'] - baseline_precision) / baseline_precision) * 100
                    df.at[idx, 'Mejora_vs_Baseline'] = f"{mejora:+.1f}%"
                else:
                    df.at[idx, 'Mejora_vs_Baseline'] = 'Baseline'
        
        print(df.to_string(index=False))
        
        # Guardar resultados
        df.to_csv('resultados_comparativos.csv', index=False, encoding='utf-8')
        print(f"\n💾 Resultados guardados: resultados_comparativos.csv")
        
        # 7. Crear gráfica simple
        print("\n6️⃣  CREANDO GRÁFICA...")
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            modos = [r['Modo'] for r in tabla_comparativa]
            precision_means = [r['Precision@5_mean'] for r in tabla_comparativa]
            precision_stds = [r['Precision@5_std'] for r in tabla_comparativa]
            
            x = np.arange(len(modos))
            ax.bar(x, precision_means, yerr=precision_stds, capsize=10, 
                  color=['blue', 'green', 'red'], alpha=0.7)
            
            ax.set_xlabel('Modo de Ranking')
            ax.set_ylabel('Precision@5')
            ax.set_title('Comparación de Precision@5 entre Modos')
            ax.set_xticks(x)
            ax.set_xticklabels(modos, rotation=45, ha='right')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('grafica_precision.png', dpi=300)
            print("✅ Gráfica guardada: grafica_precision.png")
            
            # Mostrar gráfica
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Error creando gráfica: {e}")
        
        # 8. Resumen ejecutivo
        print("\n" + "="*80)
        print("📈 RESUMEN EJECUTIVO")
        print("="*80)
        
        if len(tabla_comparativa) >= 2:
            baseline = next((r for r in tabla_comparativa if r['Modo'] == 'Baseline'), None)
            features = next((r for r in tabla_comparativa if r['Modo'] == 'RAG+Features'), None)
            rlhf = next((r for r in tabla_comparativa if r['Modo'] == 'RAG+RLHF'), None)
            
            if baseline and features:
                mejora_features = ((features['Precision@5_mean'] - baseline['Precision@5_mean']) / baseline['Precision@5_mean']) * 100
                print(f"\n1. RAG+Features mejora Precision@5 en {mejora_features:+.1f}% sobre Baseline")
            
            if baseline and rlhf:
                mejora_rlhf = ((rlhf['Precision@5_mean'] - baseline['Precision@5_mean']) / baseline['Precision@5_mean']) * 100
                print(f"2. RAG+RLHF mejora Precision@5 en {mejora_rlhf:+.1f}% sobre Baseline")
                
                if features:
                    mejora_adicional = mejora_rlhf - mejora_features
                    print(f"3. RLHF aporta {mejora_adicional:+.1f}% adicional sobre Features")
        
        print(f"\n📊 Métricas calculadas sobre {tabla_comparativa[0]['Queries']} queries")
        print(f"📈 Todos los productos cargados: {len(raw_products):,}")
        
    else:
        print("❌ No hay métricas válidas para comparar")
        print("💡 El sistema necesita procesar queries correctamente")
        
except ImportError as e:
    print(f"\n❌ ERROR DE IMPORTACIÓN: {e}")
    print(f"💡 Asegúrate de que todos los módulos están disponibles")
    traceback.print_exc()
    
except Exception as e:
    print(f"\n❌ ERROR GENERAL: {e}")
    traceback.print_exc()

print("\n" + "="*80)
print("🎉 EVALUACIÓN COMPLETADA")
print("="*80)