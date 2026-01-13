# experimento_completo_4_metodos.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import random
import logging
from datetime import datetime
import sys
import traceback
def setup_directories():
    directories = [
        'data/cache',
        'results',
        'logs',
        'data/interactions',
        'data/backups'
    ]
    
    created_dirs = []
    for dir_path in directories:
        path = Path(dir_path)
        try:
            path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(str(path))
        except Exception as e:
            print(f"  ✗ Error creando {dir_path}: {e}")
    
    return created_dirs

def setup_logging():
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"experimento_{timestamp}.log"
    
    try:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        app_logger = logging.getLogger(__name__)
        app_logger.info(f" Log iniciado: {log_file}")
        
        return app_logger
    except Exception as e:
        print(f"  Error configurando logging: {e}")
        print("  Usando logging básico...")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
        logger = logging.getLogger(__name__)
        logger.warning("Logging alternativo activado")
        return logger

def check_dependencies():
    required = ['numpy', 'pandas', 'faiss']
    optional = ['scipy', 'tqdm', 'sentence_transformers']
    missing_required = []
    missing_optional = []
    
    print(" Verificando dependencias...")
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"  ✗ {package} (REQUERIDO)")
    
    for package in optional:
        try:
            __import__(package)
            print(f"  ✓ {package} (opcional)")
        except ImportError:
            missing_optional.append(package)
            print(f"    {package} (opcional, faltante)")
    
    if missing_required:
        print(f"\n Paquetes REQUERIDOS faltantes: {', '.join(missing_required)}")
        print(f" Ejecutar: pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n  Paquetes opcionales faltantes: {', '.join(missing_optional)}")
        print(" Para funcionalidad completa: pip install scipy tqdm sentence-transformers")
    
    return True
print("\n" + "="*60)
print(" INICIALIZANDO EXPERIMENTO - 4 MÉTODOS")
print("="*60)

print("\n Creando estructura de directorios...")
dirs = setup_directories()
print(f"    Directorios creados/verificados: {len(dirs)}")

print("\n Configurando sistema de logging...")
logger = setup_logging()

print("\n Verificando dependencias de Python...")
deps_ok = check_dependencies()

if not deps_ok:
    print("\n No se pueden continuar sin las dependencias requeridas")
    sys.exit(1)

print("\n" + "="*60)
print(" SISTEMA INICIALIZADO CORRECTAMENTE")
print("="*60 + "\n")

def load_ground_truth() -> Dict[str, List[str]]:
    gt_file = Path("data/interactions/ground_truth_REAL.json")
    
    if not gt_file.exists():
        logger.error(f" Ground truth no encontrado: {gt_file}")
        logger.info("   Ejecuta primero: python main.py interactivo")
        sys.exit(1)
    
    with open(gt_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    logger.info(f" Ground truth cargado: {len(ground_truth)} queries")
    
    total_relevantes = sum(len(ids) for ids in ground_truth.values())
    logger.info(f"   • Productos relevantes totales: {total_relevantes}")
    
    return ground_truth

def split_train_test_stratified(ground_truth: Dict, test_size: float = 0.25, 
                               seed: int = 42) -> Tuple[List[str], List[str]]:
    random.seed(seed)
    
    queries_by_count = {}
    for query, ids in ground_truth.items():
        count = len(ids)
        if count not in queries_by_count:
            queries_by_count[count] = []
        queries_by_count[count].append(query)
    
    train_queries = []
    test_queries = []
    
    for count, queries in queries_by_count.items():
        random.shuffle(queries)
        split_idx = int(len(queries) * (1 - test_size))
        
        train_queries.extend(queries[:split_idx])
        test_queries.extend(queries[split_idx:])
    
    random.shuffle(train_queries)
    random.shuffle(test_queries)
    
    logger.info(f" Split creado: {len(train_queries)} train, {len(test_queries)} test")
    
    overlap = set(train_queries) & set(test_queries)
    if overlap:
        logger.warning(f"  Overlap detectado: {len(overlap)} queries")
    
    return train_queries, test_queries

def load_or_create_system_v2() -> Any:
    try:
        from src.unified_system_v2 import UnifiedSystemV2
    except ImportError as e:
        logger.error(f" No se pudo importar UnifiedSystemV2: {e}")
        logger.error("   Asegúrate de que el archivo src/unified_system_v2.py existe")
        sys.exit(1)
    
    system_cache = Path("data/cache/unified_system_v2.pkl")
    
    if system_cache.exists():
        logger.info(" Cargando sistema V2 desde cache...")
        try:
            system = UnifiedSystemV2.load_from_cache()
            
            if system:
                logger.info(f" Sistema cargado: {len(system.canonical_products):,} productos")
                logger.info(f"   • RLHF: {' Disponible' if system.rl_ranker else '❌ No disponible'}")
                logger.info(f"   • NER: {' Disponible' if system.ner_ranker else '❌ No disponible'}")
                return system
            else:
                logger.warning("  Sistema V2 no encontrado en cache, creando nuevo...")
        except Exception as e:
            logger.error(f" Error cargando sistema desde cache: {e}")
            logger.warning("  Creando nuevo sistema...")
    
    logger.info(" Creando nuevo sistema V2...")
    system = UnifiedSystemV2()
    
    try:
        success = system.initialize_with_ner(
            limit=100000,  # Límite para experimentos rápidos
            use_cache=True,
            use_zero_shot=True
        )
        
        if not success:
            logger.error(" Error inicializando sistema V2")
            sys.exit(1)
        
        return system
    except Exception as e:
        logger.error(f" Error al inicializar sistema: {e}")
        traceback.print_exc()
        sys.exit(1)

def train_rlhf_on_system(system, train_queries: List[str]) -> bool:
    interactions_file = Path("data/interactions/real_interactions.jsonl")
    
    if not interactions_file.exists():
        logger.warning("  No hay interacciones para entrenar RLHF")
        return False
    
    logger.info(f" Entrenando RLHF con {len(train_queries)} queries de entrenamiento...")
    
    try:
        success = system.train_rlhf_with_queries(train_queries, interactions_file)
        
        if success:
            logger.info(" RLHF entrenado exitosamente")
            
            if hasattr(system, 'rl_ranker') and system.rl_ranker:
                stats = system.rl_ranker.get_stats()
                logger.info(f"   • Feedback procesado: {stats.get('feedback_count', 0)}")
                logger.info(f"   • Features aprendidas: {stats.get('weights_count', 0)}")
        else:
            logger.warning("  RLHF no pudo ser entrenado (pocos datos?)")
        
        return success
    except Exception as e:
        logger.error(f" Error entrenando RLHF: {e}")
        traceback.print_exc()
        return False

def calculate_ranking_metrics(ranked_ids: List[str], relevant_ids: List[str], 
                            k: int = 5) -> Dict[str, float]:
    if not relevant_ids or k == 0:
        return {'mrr': 0.0, 'precision@k': 0.0, 'recall@k': 0.0, 'ndcg@k': 0.0}
    
    mrr = 0.0
    for i, pid in enumerate(ranked_ids):
        if pid in relevant_ids:
            mrr = 1.0 / (i + 1)
            break
    
    top_k = ranked_ids[:k]
    relevant_in_top = [pid for pid in top_k if pid in relevant_ids]
    precision = len(relevant_in_top) / k if k > 0 else 0.0
    
    recall = len(relevant_in_top) / len(relevant_ids)
    
    dcg = 0.0
    for i, pid in enumerate(top_k):
        if pid in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)
    
    ideal_relevance = [1] * min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(len(ideal_relevance)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    
    return {
        'mrr': mrr,
        'precision@k': precision,
        'recall@k': recall,
        'ndcg@k': ndcg,
        'found': len(relevant_in_top)
    }

def evaluate_method_on_query(system, method: str, query: str, 
                           relevant_ids: List[str], k: int = 5) -> Dict[str, Any]:
    try:
        results = system.query_four_methods(query, k=k*2)
        method_results = results['methods'].get(method, [])
        
        if not method_results:
            return {
                'mrr': 0.0,
                'precision@k': 0.0,
                'recall@k': 0.0,
                'ndcg@k': 0.0,
                'found': 0,
                'success': False
            }
        
        ranked_ids = []
        for product in method_results[:k]:
            product_id = getattr(product, 'id', None)
            if product_id:
                ranked_ids.append(product_id)
        
        metrics = calculate_ranking_metrics(ranked_ids, relevant_ids, k)
        metrics['success'] = True
        
        return metrics
        
    except Exception as e:
        logger.error(f" Error evaluando {method} en '{query}': {e}")
        return {
            'mrr': 0.0,
            'precision@k': 0.0,
            'recall@k': 0.0,
            'ndcg@k': 0.0,
            'found': 0,
            'success': False
        }

def run_statistical_analysis(results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    try:
        from scipy import stats
        
        tests = {}
        baseline_key = 'baseline'
        
        if baseline_key not in results or len(results[baseline_key]) < 3:
            logger.warning("  Insuficientes datos para análisis estadístico")
            return {}
        
        baseline_metrics = [r['mrr'] for r in results[baseline_key] if 'mrr' in r]
        
        for method in ['ner_enhanced', 'rlhf', 'full_hybrid']:
            if method not in results or len(results[method]) < 3:
                continue
            
            method_metrics = [r['mrr'] for r in results[method] if 'mrr' in r]
            
            if len(baseline_metrics) != len(method_metrics):
                logger.warning(f"  Número de muestras diferente para {method}")
                continue
            
            if len(baseline_metrics) > 2:
                t_stat, p_value = stats.ttest_rel(baseline_metrics, method_metrics)
                
                baseline_mean = np.mean(baseline_metrics)
                method_mean = np.mean(method_metrics)
                mean_diff = method_mean - baseline_mean
                
                pooled_std = np.sqrt((np.std(baseline_metrics)**2 + np.std(method_metrics)**2) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
                
                percent_improvement = (mean_diff / baseline_mean * 100) if baseline_mean > 0 else 0
                
                tests[method] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'cohens_d': float(cohens_d),
                    'mean_improvement': float(mean_diff),
                    'percent_improvement': float(percent_improvement),
                    'baseline_mean': float(baseline_mean),
                    'method_mean': float(method_mean),
                    'n_samples': len(baseline_metrics)
                }
        
        return tests
        
    except ImportError:
        logger.warning("  SciPy no instalado. Skipping tests estadísticos.")
        logger.info("   Instalar: pip install scipy")
        return {}
    except Exception as e:
        logger.error(f" Error en análisis estadístico: {e}")
        return {}
class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):  
            return int(o)
        if isinstance(o, np.floating):  
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (bool, np.bool_)):
            return bool(o)
        if pd.isna(o):
            return None
        return super().default(o)

def save_results(results: Dict[str, Any], summary: Dict[str, Any], 
                tests: Dict[str, Any], train_queries: List[str], 
                test_queries: List[str]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_file = Path(f"results/experimento_4_metodos_{timestamp}.json")
    
    save_data = {
        'metadata': {
            'timestamp': timestamp,
            'train_queries_count': len(train_queries),
            'test_queries_count': len(test_queries),
            'split_ratio': '75/25',
            'seed': 42
        },
        'summary': {},
        'statistical_tests': {},
        'train_queries': train_queries,
        'test_queries': test_queries
    }
    
    def normalize_json_value(value: Any) -> Any:
        if isinstance(value, np.floating):  
            return float(value)
        if isinstance(value, np.integer):   
            return int(value)
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if pd.isna(value):
            return None
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value
    
    for method, stats in summary.items():
        save_data['summary'][method] = {
            key: normalize_json_value(value)
            for key, value in stats.items()
        }
    
    for method, test in tests.items():
        save_data['statistical_tests'][method] = {
            key: normalize_json_value(value)
            for key, value in test.items()
        }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, cls=EnhancedJSONEncoder)
    
    logger.info(f" Resultados JSON: {json_file}")
    
    csv_file = Path(f"results/experimento_4_metodos_{timestamp}.csv")
    
    rows = []
    methods = ['baseline', 'ner_enhanced', 'rlhf', 'full_hybrid']
    
    for method in methods:
        if method in results:
            for i, metrics in enumerate(results[method]):
                rows.append({
                    'method': method,
                    'query_idx': i,
                    'mrr': normalize_json_value(metrics.get('mrr', 0)),
                    'precision@5': normalize_json_value(metrics.get('precision@k', 0)),
                    'recall@5': normalize_json_value(metrics.get('recall@k', 0)),
                    'ndcg@5': normalize_json_value(metrics.get('ndcg@k', 0)),
                    'found': normalize_json_value(metrics.get('found', 0)),
                    'success': normalize_json_value(metrics.get('success', False))
                })
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        logger.info(f" Resultados CSV: {csv_file}")
    
    txt_file = Path(f"results/resumen_{timestamp}.txt")
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RESUMEN DEL EXPERIMENTO - 4 MÉTODOS DE RANKING\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(" MÉTRICAS PROMEDIO (MRR):\n")
        f.write("-" * 50 + "\n")
        for method in methods:
            if method in summary:
                mrr_mean = normalize_json_value(summary[method]['mrr_mean'])
                mrr_std = normalize_json_value(summary[method]['mrr_std'])
                f.write(f"{method.replace('_', ' ').title():20} {mrr_mean:.4f} ± {mrr_std:.4f}\n")
        
        f.write("\n TESTS ESTADÍSTICOS (vs Baseline):\n")
        f.write("-" * 50 + "\n")
        for method in ['ner_enhanced', 'rlhf', 'full_hybrid']:
            if method in tests:
                p_value = normalize_json_value(tests[method].get('p_value', 1.0))
                significant = normalize_json_value(tests[method].get('significant', False))
                percent_improvement = normalize_json_value(tests[method].get('percent_improvement', 0.0))
                cohens_d = normalize_json_value(tests[method].get('cohens_d', 0.0))
                
                sig = " SIGNIFICATIVO" if significant else "  NO SIGNIFICATIVO"
                f.write(f"{method.replace('_', ' ').title():20} p={p_value:.4f} {sig}\n")
                f.write(f"                     Mejora: {percent_improvement:+.2f}% (d={cohens_d:.3f})\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONCLUSIONES\n")
        f.write("=" * 80 + "\n\n")
        
        valid_methods = {k: v for k, v in summary.items() if v.get('mrr_mean', 0) > 0}
        if valid_methods:
            best_method = max(valid_methods.items(), key=lambda x: normalize_json_value(x[1]['mrr_mean']))[0]
            
            if 'baseline' in summary and normalize_json_value(summary['baseline']['mrr_mean']) > 0:
                baseline_mrr = normalize_json_value(summary['baseline']['mrr_mean'])
                best_mrr = normalize_json_value(summary[best_method]['mrr_mean'])
                improvement = ((best_mrr / baseline_mrr) - 1) * 100
                
                f.write(f" MEJOR MÉTODO: {best_method.replace('_', ' ').title()}\n")
                f.write(f"   • MRR Baseline: {baseline_mrr:.4f}\n")
                f.write(f"   • MRR Mejor:    {best_mrr:.4f}\n")
                f.write(f"   • Mejora:       {improvement:+.2f}%\n\n")
                
                is_significant = best_method in tests and normalize_json_value(tests[best_method].get('significant', False))
                
                if improvement > 5 and is_significant:
                    f.write(" ¡MEJORA SIGNIFICATIVA Y RELEVANTE!\n")
                    f.write("   El sistema funciona correctamente.\n")
                elif improvement > 0:
                    f.write("  MEJORA PEQUEÑA\n")
                    f.write("   Considera recolectar más datos.\n")
                else:
                    f.write(" SIN MEJORA\n")
                    f.write("   Revisa la implementación.\n")
            else:
                f.write(f" MEJOR MÉTODO: {best_method.replace('_', ' ').title()}\n")
                f.write(f"   • MRR: {normalize_json_value(summary[best_method]['mrr_mean']):.4f}\n")
                f.write("   • Baseline no funcionó (MRR=0)\n")
        else:
            f.write(" NINGÚN MÉTODO FUNCIONÓ CORRECTAMENTE\n")
            f.write("   • Todos los métodos tuvieron MRR=0\n")
            f.write("   • Verifica los errores en los logs\n")
    
    logger.info(f" Resumen: {txt_file}")
    
    return json_file, csv_file, txt_file


def main():
    try:
        print("\n" + "="*80)
        print(" EXPERIMENTO COMPLETO: 4 MÉTODOS DE RANKING")
        print("="*80)
        
        logger.info(" Cargando ground truth...")
        ground_truth = load_ground_truth()
        
        logger.info("  Creando split train/test estratificado...")
        train_queries, test_queries = split_train_test_stratified(
            ground_truth, test_size=0.25, seed=42
        )
        
        max_test_queries = min(30, len(test_queries))
        test_queries = test_queries[:max_test_queries]
        
        print("\n DATASET:")
        print(f"   • Total queries: {len(ground_truth)}")
        print(f"   • Train queries: {len(train_queries)}")
        print(f"   • Test queries: {len(test_queries)}")
        
        logger.info(" Cargando sistema V2...")
        system = load_or_create_system_v2()
        
        print("\n SISTEMA:")
        print(f"   • Productos: {len(system.canonical_products):,}")
        print(f"   • RLHF: {' Disponible' if hasattr(system, 'rl_ranker') and system.rl_ranker else ' No entrenado'}")
        print(f"   • NER: {' Disponible' if hasattr(system, 'ner_ranker') and system.ner_ranker else ' No disponible'}")
        
        train_rlhf_on_system(system, train_queries)
        
        print("\n EVALUANDO 4 MÉTODOS...")
        
        methods = ['baseline', 'ner_enhanced', 'rlhf', 'full_hybrid']
        results = {method: [] for method in methods}
        
        for i, query in enumerate(test_queries, 1):
            relevant_ids = ground_truth.get(query, [])
            
            if not relevant_ids:
                continue
            
            if i % 5 == 0 or i == 1 or i == len(test_queries):
                print(f"   [{i}/{len(test_queries)}] '{query[:40]}...'")
            
            for method in methods:
                metrics = evaluate_method_on_query(system, method, query, relevant_ids, k=5)
                results[method].append(metrics)
        
        print("\n CALCULANDO RESULTADOS...")
        
        summary = {}
        for method in methods:
            if method in results and results[method]:
                method_results = [r for r in results[method] if r.get('success', False)]
                
                if method_results:
                    df = pd.DataFrame(method_results)
                    summary[method] = {
                        'mrr_mean': float(df['mrr'].mean()),
                        'mrr_std': float(df['mrr'].std()),
                        'precision_mean': float(df['precision@k'].mean()),
                        'recall_mean': float(df['recall@k'].mean()),
                        'ndcg_mean': float(df['ndcg@k'].mean()),
                        'total_found': int(df['found'].sum()),
                        'n_queries': len(method_results),
                        'success_rate': float(len(method_results) / len(results[method]))
                    }
                else:
                    summary[method] = {
                        'mrr_mean': 0.0,
                        'mrr_std': 0.0,
                        'precision_mean': 0.0,
                        'recall_mean': 0.0,
                        'ndcg_mean': 0.0,
                        'total_found': 0,
                        'n_queries': 0,
                        'success_rate': 0.0
                    }
        
        tests = run_statistical_analysis(results)
        
        print("\n" + "="*80)
        print(" RESULTADOS FINALES")
        print("="*80)
        
        print(f"\n{'Método':<20} {'MRR':<8} {'P@5':<8} {'R@5':<8} {'NDCG@5':<8} {'Found':<8}")
        print("-" * 70)
        
        for method in methods:
            if method in summary:
                stats = summary[method]
                print(f"{method.replace('_', ' ').title():<20} "
                      f"{stats['mrr_mean']:.4f}  "
                      f"{stats['precision_mean']:.4f}  "
                      f"{stats['recall_mean']:.4f}  "
                      f"{stats['ndcg_mean']:.4f}  "
                      f"{stats['total_found']:>6}")
        
        if tests:
            print("\n SIGNIFICANCIA ESTADÍSTICA (vs Baseline)")
            print("-" * 50)
            
            for method in ['ner_enhanced', 'rlhf', 'full_hybrid']:
                if method in tests:
                    test = tests[method]
                    sig = "Significante" if test.get('significant', False) else "No significante"
                    print(f"{method.replace('_', ' ').title():20} "
                          f"p={test.get('p_value', 1.0):.4f} {sig} "
                          f"Mejora: {test.get('percent_improvement', 0.0):+.2f}%")
        
        logger.info(" Guardando resultados...")
        json_file, csv_file, txt_file = save_results(results, summary, tests, 
                                                    train_queries, test_queries)
        
        print("\n" + "="*80)
        print(" CONCLUSIONES Y RECOMENDACIONES")
        print("="*80)
        
        if 'full_hybrid' in summary and summary['full_hybrid']['mrr_mean'] > 0:
            if 'baseline' in summary and summary['baseline']['mrr_mean'] > 0:
                baseline_mrr = summary['baseline']['mrr_mean']
                hybrid_mrr = summary['full_hybrid']['mrr_mean']
                improvement = ((hybrid_mrr / baseline_mrr) - 1) * 100
                
                print("\n FULL HYBRID vs BASELINE:")
                print(f"   • Baseline MRR:  {baseline_mrr:.4f}")
                print(f"   • Hybrid MRR:    {hybrid_mrr:.4f}")
                print(f"   • Mejora:        {improvement:+.2f}%")
                
                is_significant = 'full_hybrid' in tests and tests['full_hybrid'].get('significant', False)
                
                if improvement > 5 and is_significant:
                    print("\n ¡EXCELENTE! MEJORA SIGNIFICATIVA (>5%)")
                    print("   • Tu sistema funciona correctamente")
                    print("   • RLHF y NER están aportando valor")
                    print("   • Puedes proceder con paper IEEE")
                elif improvement > 0:
                    print("\n  MEJORA PEQUEÑA ({improvement:+.2f}%)")
                    print("   • Recomendado: Recolectar más datos")
                else:
                    print("\n SIN MEJORA ({improvement:+.2f}%)")
                    print("   • Posibles causas:")
                    print("     1. Baseline demasiado bueno")
                    print("     2. Insuficientes datos de entrenamiento")
        
        print("\n PRÓXIMOS PASOS:")
        print(f"1. Revisar resultados: {txt_file}")
        if 'full_hybrid' in summary and summary['full_hybrid']['mrr_mean'] > 0.1:
            print("2. Si MRR > 0.1: ¡Prepara paper!")
        else:
            print("2. Si MRR < 0.1: Recolectar más feedback")
        print("3. Ejecutar: python main.py interactivo (más clicks)")
        
        print("\n EXPERIMENTO COMPLETADO")
        print("   • Archivos guardados en: results/")
        
        logger.info(" Experimento completado exitosamente")
        
    except KeyboardInterrupt:
        print("\n\n  Experimento interrumpido por el usuario")
        logger.warning("Experimento interrumpido por el usuario")
    except Exception as e:
        logger.error(f" Error en experimento: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()