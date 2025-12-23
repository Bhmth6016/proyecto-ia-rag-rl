# optimizar_rlhf.py
"""
Optimización de RLHF - Encuentra los mejores parámetros
"""
import json
import pickle
import numpy as np
from pathlib import Path
from src.ranking.rl_ranker_fixed import RLHFRankerFixed

def evaluar_parametros(learning_rate, match_rating_balance):
    """Evalúa un conjunto de parámetros"""
    print(f"\n🔬 Probando: LR={learning_rate}, Balance={match_rating_balance}")
    
    # Cargar sistema
    system_path = Path("data/cache/unified_system.pkl")
    with open(system_path, 'rb') as f:
        system = pickle.load(f)
    
    # Crear RL ranker con parámetros
    rl_ranker = RLHFRankerFixed(
        learning_rate=learning_rate,
        match_rating_balance=match_rating_balance
    )
    
    # Cargar interacciones
    interactions_file = Path("data/interactions/real_interactions.jsonl")
    products_by_id = {p.id: p for p in system.canonical_products if hasattr(p, 'id')}
    
    # Entrenar
    trained = 0
    with open(interactions_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                interaction = json.loads(line.strip())
                if interaction.get('interaction_type') == 'click':
                    context = interaction.get('context', {})
                    query = context.get('query', '').strip()
                    product_id = context.get('product_id')
                    position = context.get('position', 1)
                    
                    if not query or not product_id:
                        continue
                    
                    product = products_by_id.get(product_id)
                    if product:
                        # Reward basado en posición
                        if position == 1:
                            reward = 0.8
                        elif position <= 3:
                            reward = 1.0
                        elif position <= 10:
                            reward = 1.5
                        else:
                            reward = 2.0
                        
                        rl_ranker.learn_from_human_feedback(product, query, position, reward)
                        trained += 1
                        
                        if trained >= 50:  # Limitar para prueba rápida
                            break
            except:
                continue
    
    # Evaluar rápidamente
    stats = rl_ranker.get_stats()
    ratio = stats.get('match_vs_rating_ratio', 0)
    
    print(f"   → Ratio match/rating: {ratio:.2f}")
    print(f"   → Features aprendidas: {stats['weights_count']}")
    
    # Puntuación: queremos ratio entre 0.5 y 3.0
    if 0.5 <= ratio <= 3.0:
        score = 10.0 - abs(ratio - 1.5)  # Mejor cerca de 1.5
    elif ratio > 3.0:
        score = 3.0 / ratio  # Penalizar ratio alto
    else:
        score = ratio / 0.5  # Penalizar ratio bajo
    
    return {
        'learning_rate': learning_rate,
        'match_rating_balance': match_rating_balance,
        'ratio': ratio,
        'score': score,
        'features': stats['weights_count']
    }

def main():
    print("\n" + "="*80)
    print("⚙️  OPTIMIZADOR DE PARÁMETROS RLHF")
    print("="*80)
    
    # Parámetros a probar
    learning_rates = [0.2, 0.3, 0.4, 0.5, 0.6]
    balances = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    resultados = []
    
    for lr in learning_rates:
        for balance in balances:
            resultado = evaluar_parametros(lr, balance)
            resultados.append(resultado)
    
    # Ordenar por score
    resultados.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n" + "="*80)
    print("🏆 MEJORES PARÁMETROS ENCONTRADOS")
    print("="*80)
    
    print(f"\n📊 TOP 5 CONFIGURACIONES:")
    for i, res in enumerate(resultados[:5], 1):
        print(f"{i}. LR={res['learning_rate']}, Balance={res['match_rating_balance']}")
        print(f"   → Ratio: {res['ratio']:.2f}, Score: {res['score']:.2f}, Features: {res['features']}")
        print()
    
    # Recomendación
    mejor = resultados[0]
    print(f"🎯 RECOMENDACIÓN:")
    print(f"   Usar: learning_rate={mejor['learning_rate']}, match_rating_balance={mejor['match_rating_balance']}")
    print(f"   Ratio esperado: {mejor['ratio']:.2f}")
    
    # Crear archivo de configuración
    config = {
        'optimal_parameters': {
            'learning_rate': mejor['learning_rate'],
            'match_rating_balance': mejor['match_rating_balance'],
            'expected_ratio': mejor['ratio'],
            'score': mejor['score']
        },
        'all_results': resultados
    }
    
    with open("optimizacion_rlhf.json", "w") as f:
        import json
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Resultados guardados en: optimizacion_rlhf.json")
    
    print("\n" + "="*80)
    print("✅ OPTIMIZACIÓN COMPLETADA")
    print("="*80)

if __name__ == "__main__":
    main()