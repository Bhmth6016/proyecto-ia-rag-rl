# train_rl_simple.py
"""
Entrenamiento RL simple y efectivo
"""
import json
import pickle
import numpy as np
from pathlib import Path
from src.ranking.rl_ranker_fixed import RLHFRankerFixed

def main():
    print("\n" + "="*80)
    print("🧠 ENTRENAMIENTO RL SIMPLE Y EFECTIVO")
    print("="*80)
    
    # 1. Cargar sistema
    system_path = Path("data/cache/unified_system.pkl")
    if not system_path.exists():
        print("❌ Sistema no encontrado")
        return
    
    with open(system_path, 'rb') as f:
        system = pickle.load(f)
    
    print(f"✅ Sistema cargado: {len(system.canonical_products):,} productos")
    
    # 2. Crear RL ranker
    rl_ranker = RLHFRankerFixed(learning_rate=0.2)
    
    # 3. Cargar interacciones
    interactions_file = Path("data/interactions/real_interactions.jsonl")
    if not interactions_file.exists():
        print("❌ No hay interacciones para entrenar")
        print("💡 Ejecuta primero el sistema interactivo y haz clicks")
        return
    
    # Crear mapa de productos
    products_by_id = {p.id: p for p in system.canonical_products if hasattr(p, 'id')}
    
    # 4. Entrenar
    print("\n📚 Entrenando con interacciones...")
    
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
                        continue
                    
                    # Entrenar
                    rl_ranker.learn_from_feedback(product, query, position)
                    
            except:
                continue
    
    # 5. Verificar balance
    stats = rl_ranker.get_stats()
    print(f"\n📊 Estadísticas del entrenamiento:")
    print(f"   • Feedback procesado: {stats['feedback_count']}")
    print(f"   • Features aprendidas: {stats['weights_count']}")
    print(f"   • Ratio match/rating: {stats.get('match_vs_rating_ratio', 0):.2f}")
    
    if stats.get('match_vs_rating_ratio', 0) < 2.0:
        print("   ⚠️  RL está dando mucho peso a rating vs match")
        print("   🔧 Ajustando manualmente pesos...")
        
        # Ajustar pesos manualmente
        for feature in list(rl_ranker.feature_weights.keys()):
            if 'rating' in feature.lower():
                rl_ranker.feature_weights[feature] *= 0.3  # Reducir rating
            elif 'match' in feature.lower():
                rl_ranker.feature_weights[feature] *= 1.5  # Aumentar match
    
    # 6. Guardar
    rl_ranker.save("data/cache/rl_ranker_trained_fixed.pkl")
    
    # Actualizar sistema
    system.rl_ranker = rl_ranker
    system.save_to_cache("data/cache/unified_system_with_fixed_rl.pkl")
    
    print("\n✅ RL entrenado y guardado correctamente")
    print("🎯 Ahora ejecuta: python evaluador_simple_final.py")

if __name__ == "__main__":
    main()