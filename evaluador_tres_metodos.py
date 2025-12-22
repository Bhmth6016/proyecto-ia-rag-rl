"""
Evaluador de 3 métodos: Baseline, RAG+Features, RAG+RL
"""
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import seaborn as sns

# Configuración matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ThreeMethodEvaluator:
    def __init__(self):
        self.system = None
        self.ground_truth = {}
        self.results = []
        
    def load_system(self):
        """Carga sistema con RL"""
        system_path = Path("data/cache/unified_system_with_fixed_rl.pkl")
        with open(system_path, 'rb') as f:
            self.system = pickle.load(f)
        print(f"✅ Sistema cargado: {len(self.system.canonical_products):,} productos")
    
    def load_ground_truth(self):
        """Carga ground truth"""
        gt_file = Path("data/interactions/realistic_ground_truth.json")
        if gt_file.exists():
            with open(gt_file, 'r', encoding='utf-8') as f:
                self.ground_truth = json.load(f)
        print(f"📊 Ground truth: {len(self.ground_truth)} queries")
    
    def method_baseline(self, query, k=30):
        """Método 1: Baseline (solo similitud coseno)"""
        query_embedding = self.system.canonicalizer.embedding_model.encode(
            query, normalize_embeddings=True
        )
        results = self.system.vector_store.search(query_embedding, k=k)
        return results
    
    def method_features(self, query, k=30):
        """Método 2: RAG + Features heurísticas"""
        # Primero retrieval
        baseline_results = self.method_baseline(query, k)
        
        # Aplicar ranking por features heurísticas
        # 1. Rating (40% peso)
        # 2. Match título (40% peso)
        # 3. Precio disponible (10%)
        # 4. Reviews count (10%)
        
        scored_products = []
        for product in baseline_results:
            score = 0.0
            
            # Rating (0-5 → 0-1)
            if hasattr(product, 'rating') and product.rating:
                try:
                    rating_score = float(product.rating) / 5.0
                    score += rating_score * 0.4
                except:
                    pass
            
            # Match con título
            if hasattr(product, 'title'):
                query_lower = query.lower()
                title_lower = product.title.lower()
                query_words = set(query_lower.split())
                title_words = set(title_lower.split())
                match_count = len(query_words.intersection(title_words))
                
                if query_words:
                    match_ratio = match_count / len(query_words)
                    score += match_ratio * 0.4
            
            # Precio disponible
            if hasattr(product, 'price') and product.price:
                score += 0.1
            
            # Reviews count (simulado)
            if hasattr(product, 'review_count') and product.review_count:
                score += 0.1
            
            scored_products.append((score, product))
        
        # Ordenar
        scored_products.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored_products]
    
    def method_rl(self, query, k=30):
        """Método 3: RAG + RL"""
        # Baseline primero
        baseline_results = self.method_baseline(query, k)
        
        # Aplicar RL si ha aprendido
        if (hasattr(self.system, 'rl_ranker') and 
            hasattr(self.system.rl_ranker, 'has_learned') and 
            self.system.rl_ranker.has_learned):
            
            # Calcular scores baseline
            baseline_scores = []
            for product in baseline_results:
                if hasattr(product, 'similarity'):
                    baseline_scores.append(product.similarity)
                else:
                    baseline_scores.append(0.5)
            
            # Aplicar RL
            return self.system.rl_ranker.rank_products(
                baseline_results, query, baseline_scores
            )
        
        return baseline_results
    
    def evaluate_query(self, query, relevant_ids):
        """Evalúa una query con los 3 métodos"""
        try:
            # Método 1: Baseline
            baseline_results = self.method_baseline(query, 20)
            baseline_top5 = [p.id for p in baseline_results[:5]]
            
            # Método 2: Features
            features_results = self.method_features(query, 20)
            features_top5 = [p.id for p in features_results[:5]]
            
            # Método 3: RL
            rl_results = self.method_rl(query, 20)
            rl_top5 = [p.id for p in rl_results[:5]]
            
            # Calcular métricas
            metrics = {
                'query': query,
                'total_relevant': len(relevant_ids)
            }
            
            for method_name, top5 in [
                ('baseline', baseline_top5),
                ('features', features_top5),
                ('rl', rl_top5)
            ]:
                # Precision@5
                relevant_in_top5 = sum(1 for pid in top5 if pid in relevant_ids)
                precision = relevant_in_top5 / 5.0 if top5 else 0.0
                
                # Recall@5
                recall = relevant_in_top5 / len(relevant_ids) if relevant_ids else 0.0
                
                # NDCG@5
                ndcg = self.calculate_ndcg(top5, relevant_ids, 5)
                
                metrics.update({
                    f'{method_name}_precision': precision,
                    f'{method_name}_recall': recall,
                    f'{method_name}_ndcg': ndcg,
                    f'{method_name}_relevant': relevant_in_top5
                })
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error evaluando '{query}': {e}")
            return None
    
    def calculate_ndcg(self, ranked_ids, relevant_ids, k=5):
        """Calcula NDCG@k"""
        if not relevant_ids:
            return 0.0
        
        # DCG
        dcg = 0.0
        for i, pid in enumerate(ranked_ids[:k]):
            if pid in relevant_ids:
                dcg += 1.0 / np.log2(i + 2)
        
        # IDCG (ranking ideal)
        ideal_relevance = [1] * min(len(relevant_ids), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(len(ideal_relevance)))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def run_evaluation(self, max_queries=15):
        """Ejecuta evaluación completa"""
        print("\n🔬 EVALUANDO 3 MÉTODOS DE RANKING")
        print("="*80)
        
        self.load_system()
        self.load_ground_truth()
        
        queries_to_evaluate = list(self.ground_truth.items())[:max_queries]
        
        for query, relevant_ids in queries_to_evaluate:
            print(f"\n🔍 Query: '{query}'")
            print(f"   • Productos relevantes: {len(relevant_ids)}")
            
            metrics = self.evaluate_query(query, relevant_ids)
            if metrics:
                self.results.append(metrics)
                
                # Mostrar resultados
                print(f"   📊 Baseline:  P@5={metrics['baseline_precision']:.3f}, R@5={metrics['baseline_recall']:.3f}")
                print(f"   ⚙️  Features:  P@5={metrics['features_precision']:.3f}, R@5={metrics['features_recall']:.3f}")
                print(f"   🤖 RL:        P@5={metrics['rl_precision']:.3f}, R@5={metrics['rl_recall']:.3f}")
                
                # Comparar
                if metrics['rl_precision'] > metrics['baseline_precision']:
                    print(f"   🎯 RL MEJORA: +{(metrics['rl_precision']-metrics['baseline_precision'])/metrics['baseline_precision']*100:.1f}%")
        
        if self.results:
            self.analyze_results()
            self.generate_plots()
    
    def analyze_results(self):
        """Analiza y muestra resultados"""
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("📈 RESULTADOS COMPARATIVOS - 3 MÉTODOS")
        print("="*80)
        
        # Métricas promedio
        metrics_summary = []
        for method in ['baseline', 'features', 'rl']:
            avg_precision = df[f'{method}_precision'].mean()
            avg_recall = df[f'{method}_recall'].mean()
            avg_ndcg = df[f'{method}_ndcg'].mean()
            
            metrics_summary.append({
                'Método': method.upper(),
                'Precision@5': f"{avg_precision:.3f}",
                'Recall@5': f"{avg_recall:.3f}",
                'NDCG@5': f"{avg_ndcg:.3f}",
                'Relevantes recuperados': f"{df[f'{method}_relevant'].sum()}/{df['total_relevant'].sum()}",
                '% Recuperados': f"{df[f'{method}_relevant'].sum()/df['total_relevant'].sum()*100:.1f}%"
            })
        
        # Crear tabla
        summary_df = pd.DataFrame(metrics_summary)
        print("\n📊 MÉTRICAS PROMEDIO:")
        print(summary_df.to_string(index=False))
        
        # Mejoras relativas
        baseline_avg = df['baseline_precision'].mean()
        features_avg = df['features_precision'].mean()
        rl_avg = df['rl_precision'].mean()
        
        print(f"\n📈 MEJORAS RELATIVAS vs BASELINE:")
        print(f"   • Features:  {((features_avg - baseline_avg) / baseline_avg * 100):+.1f}%")
        print(f"   • RL:        {((rl_avg - baseline_avg) / baseline_avg * 100):+.1f}%")
        
        # Análisis por query
        print(f"\n🔍 ANÁLISIS POR QUERY:")
        print(f"   • Queries evaluadas: {len(df)}")
        print(f"   • Queries donde RL mejora: {len(df[df['rl_precision'] > df['baseline_precision']])}")
        print(f"   • Queries donde Features mejora: {len(df[df['features_precision'] > df['baseline_precision']])}")
        print(f"   • Queries donde ambos mejoran: {len(df[(df['rl_precision'] > df['baseline_precision']) & (df['features_precision'] > df['baseline_precision'])])}")
        
        # Guardar resultados
        output_file = "resultados_tres_metodos.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Resultados detallados guardados en: {output_file}")
        
        return df
    
    def generate_plots(self):
        """Genera gráficas comparativas"""
        df = pd.DataFrame(self.results)
        
        # Preparar datos para gráficas
        methods = ['Baseline', 'Features', 'RL']
        precision_means = [
            df['baseline_precision'].mean(),
            df['features_precision'].mean(),
            df['rl_precision'].mean()
        ]
        
        # Gráfica 1: Precision@5 por método
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, precision_means, color=['#3498db', '#2ecc71', '#e74c3c'])
        plt.title('Precisión@5 Promedio por Método de Ranking', fontsize=14, fontweight='bold')
        plt.ylabel('Precisión@5', fontsize=12)
        plt.ylim(0, 1.0)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, precision_means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('precision_por_metodo.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Gráfica 2: Comparación por query
        plt.figure(figsize=(12, 8))
        
        # Preparar datos
        queries = df['query'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
        x = np.arange(len(queries))
        width = 0.25
        
        plt.bar(x - width, df['baseline_precision'], width, label='Baseline', color='#3498db')
        plt.bar(x, df['features_precision'], width, label='Features', color='#2ecc71')
        plt.bar(x + width, df['rl_precision'], width, label='RL', color='#e74c3c')
        
        plt.xlabel('Queries', fontsize=12)
        plt.ylabel('Precisión@5', fontsize=12)
        plt.title('Comparación de Precisión@5 por Query', fontsize=14, fontweight='bold')
        plt.xticks(x, queries, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('precision_por_query.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Gráfica 3: Heatmap de mejoras
        plt.figure(figsize=(8, 6))
        
        improvement_data = pd.DataFrame({
            'Query': queries,
            'Features vs Baseline': df['features_precision'] - df['baseline_precision'],
            'RL vs Baseline': df['rl_precision'] - df['baseline_precision']
        })
        
        # Crear pivot para heatmap
        pivot_data = improvement_data.set_index('Query')
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   center=0, cbar_kws={'label': 'Mejora en Precisión'})
        plt.title('Mejora Relativa vs Baseline por Query', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('heatmap_mejoras.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n📊 Gráficas generadas:")
        print("   • precision_por_metodo.png - Comparación global")
        print("   • precision_por_query.png - Comparación por query")
        print("   • heatmap_mejoras.png - Heatmap de mejoras")
        print("   • resultados_tres_metodos.csv - Datos completos")

def main():
    print("\n" + "="*80)
    print("🎯 EVALUADOR DE 3 MÉTODOS: Baseline vs Features vs RL")
    print("="*80)
    
    evaluator = ThreeMethodEvaluator()
    evaluator.run_evaluation(max_queries=15)
    
    print("\n" + "="*80)
    print("✅ EVALUACIÓN COMPLETA")
    print("="*80)

if __name__ == "__main__":
    main()