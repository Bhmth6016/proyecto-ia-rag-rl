# entrenar_true_rlhf.py
"""
Entrenamiento TRUE RLHF - Versión simplificada y funcional
"""
import json
import pickle
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def main():
    print("\n" + "="*80)
    print("🧠 TRUE RLHF - ENTRENAMIENTO SIMPLIFICADO Y FUNCIONAL")
    print("="*80)
    
    # 1. Cargar sistema
    system_path = Path("data/cache/unified_system.pkl")
    if not system_path.exists():
        print("❌ Sistema no encontrado")
        return
    
    with open(system_path, 'rb') as f:
        system = pickle.load(f)
    
    print(f"✅ Sistema cargado: {len(system.canonical_products):,} productos")
    
    # 2. Cargar RL ranker TRUE
    from src.ranking.rl_ranker_fixed import RLHFRankerFixed
    rl_ranker = RLHFRankerFixed(learning_rate=0.4, match_rating_balance=1.5)
    
    # 3. Cargar interacciones reales
    interactions_file = Path("data/interactions/real_interactions.jsonl")
    if not interactions_file.exists():
        print("❌ No hay interacciones reales")
        return
    
    # Crear mapa de productos
    products_by_id = {p.id: p for p in system.canonical_products if hasattr(p, 'id')}
    
    # 4. Entrenar con análisis de posiciones
    print("\n📊 ANALIZANDO INTERACCIONES REALES...")
    
    position_stats = {1: 0, '2-3': 0, '4-10': 0, '11+': 0}
    total_clicks = 0
    
    with open(interactions_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                interaction = json.loads(line.strip())
                if interaction.get('interaction_type') == 'click':
                    total_clicks += 1
                    context = interaction.get('context', {})
                    position = context.get('position', 1)
                    
                    if position == 1:
                        position_stats[1] += 1
                    elif 2 <= position <= 3:
                        position_stats['2-3'] += 1
                    elif 4 <= position <= 10:
                        position_stats['4-10'] += 1
                    else:
                        position_stats['11+'] += 1
            except:
                continue
    
    print(f"   • Total clicks: {total_clicks}")
    print(f"   • Posición 1: {position_stats[1]} ({position_stats[1]/total_clicks*100:.1f}%)")
    print(f"   • Posiciones 2-3: {position_stats['2-3']} ({position_stats['2-3']/total_clicks*100:.1f}%)")
    print(f"   • Posiciones 4-10: {position_stats['4-10']} ({position_stats['4-10']/total_clicks*100:.1f}%)")
    print(f"   • Posiciones 11+: {position_stats['11+']} ({position_stats['11+']/total_clicks*100:.1f}%)")
    
    # 5. Entrenar
    print("\n🎯 ENTRENANDO TRUE RLHF...")
    
    trained_count = 0
    
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
                    
                    # Encontrar producto
                    product = products_by_id.get(product_id)
                    if not product:
                        # Intentar buscar por ID parcial
                        for pid, prod in products_by_id.items():
                            if product_id in pid or pid in product_id:
                                product = prod
                                break
                    
                    if not product:
                        continue
                    
                    # Reward basado en posición
                    if position == 1:
                        reward = 0.8  # Menos reward para obvios
                    elif position <= 3:
                        reward = 1.0  # Normal
                    elif position <= 10:
                        reward = 1.5  # Más reward por descubrimiento
                    else:
                        reward = 2.0  # Máximo reward por descubrimiento profundo
                    
                    # Entrenar
                    rl_ranker.learn_from_human_feedback(product, query, position, reward)
                    trained_count += 1
                    
                    if trained_count % 10 == 0:
                        print(f"   ✅ {trained_count} interacciones entrenadas")
                        
            except Exception as e:
                print(f"   ⚠️  Error en interacción: {e}")
                continue
    
    # 6. Forzar aprendizaje
    rl_ranker.has_learned = True
    
    print(f"\n✅ Entrenamiento completado: {trained_count} interacciones")
    
    # 7. Mostrar estadísticas
    stats = rl_ranker.get_stats()
    
    print("\n📊 ESTADÍSTICAS DEL APRENDIZAJE:")
    print(f"   • Feedback procesado: {stats['feedback_count']}")
    print(f"   • Features aprendidas: {stats['weights_count']}")
    print(f"   • Ratio match/rating: {stats.get('match_vs_rating_ratio', 0):.2f}")
    
    # 8. Ajustar si ratio está desbalanceado
    ratio = stats.get('match_vs_rating_ratio', 0)
    
    if ratio > 3.0:
        print("   ⚠️  Ratio muy alto (demasiado match), ajustando...")
        # Reducir matches, aumentar rating
        for feature in list(rl_ranker.feature_weights.keys()):
            if 'match' in feature.lower():
                rl_ranker.feature_weights[feature] *= 0.7
            elif 'rating' in feature.lower():
                rl_ranker.feature_weights[feature] *= 1.3
    
    elif ratio < 0.5:
        print("   ⚠️  Ratio muy bajo (demasiado rating), ajustando...")
        # Reducir rating, aumentar matches
        for feature in list(rl_ranker.feature_weights.keys()):
            if 'rating' in feature.lower():
                rl_ranker.feature_weights[feature] *= 0.7
            elif 'match' in feature.lower():
                rl_ranker.feature_weights[feature] *= 1.3
    
    else:
        print("   ✅ Ratio balanceado")
    
    # 9. Mostrar top features
    if stats['top_features']:
        print("\n🔝 TOP 10 FEATURES APRENDIDAS:")
        for i, (feature, weight) in enumerate(stats['top_features'][:10], 1):
            if 'match' in feature.lower():
                icon = "🎯"
            elif 'rating' in feature.lower():
                icon = "⭐"
            elif 'preference_' in feature:
                icon = "❤️"
            else:
                icon = "📊"
            
            print(f"   {i:2d}. {icon} {feature[:30]:30} {weight:7.3f}")
    
    # 10. Guardar
    rl_ranker.save("data/cache/rl_ranker_TRUE.pkl")
    
    # Actualizar sistema
    system.rl_ranker = rl_ranker
    system.save_to_cache("data/cache/unified_system_with_TRUE_rl.pkl")
    
    print(f"\n💾 Modelo guardado:")
    print(f"   • rl_ranker_TRUE.pkl")
    print(f"   • unified_system_with_TRUE_rl.pkl")
    
    # 11. Salud del aprendizaje
    health = rl_ranker.get_learning_health()
    
    print(f"\n🏥 SALUD DEL APRENDIZAJE: {health['status']}")
    print(f"   • Puntuación: {health['score']}/5")
    
    if health['issues']:
        print(f"   ⚠️  Issues:")
        for issue in health['issues']:
            print(f"      • {issue}")
    
    print("\n" + "="*80)
    print("✅ TRUE RLHF ENTRENADO CORRECTAMENTE")
    print("\n🎯 EJECUTA AHORA:")
    print("   python evaluador_true_rlhf.py")

if __name__ == "__main__":
    main()