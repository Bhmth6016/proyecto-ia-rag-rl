# entrenar_true_rlhf.py
"""
Entrenamiento TRUE RLHF - Versión simplificada y funcional
"""
import json
import pickle
import numpy as np
from pathlib import Path
import logging
import sys
import os

# Añadir el directorio src al path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("\n" + "="*80)
    print("🧠 TRUE RLHF - ENTRENAMIENTO SIMPLIFICADO Y FUNCIONAL")
    print("="*80)
    
    # 1. Cargar sistema
    system_path = Path("data/cache/unified_system.pkl")
    if not system_path.exists():
        print("❌ Sistema no encontrado. Ejecutar primero:")
        print("   python -m src.unified_system")
        return
    
    try:
        with open(system_path, 'rb') as f:
            system = pickle.load(f)
        
        print(f"✅ Sistema cargado: {len(system.canonical_products):,} productos")
    except Exception as e:
        print(f"❌ Error cargando sistema: {e}")
        return
    
    # 2. Crear RL ranker TRUE
    try:
        from src.ranking.rl_ranker_fixed import RLHFRankerFixed
        rl_ranker = RLHFRankerFixed(learning_rate=0.4, match_rating_balance=1.5)
        print("✅ RL ranker inicializado")
    except ImportError as e:
        print(f"❌ Error importando RLHFRankerFixed: {e}")
        return
    
    # 3. Cargar interacciones reales
    interactions_file = Path("data/interactions/real_interactions.jsonl")
    if not interactions_file.exists():
        print("❌ No hay interacciones reales. Ejecutar primero:")
        print("   python simulador_realista.py")
        return
    
    # Crear mapa de productos
    products_by_id = {}
    if hasattr(system, 'canonical_products'):
        for p in system.canonical_products:
            if hasattr(p, 'id'):
                products_by_id[p.id] = p
    
    print(f"📦 Mapa de productos creado: {len(products_by_id):,} productos")
    
    # 4. Analizar interacciones
    print("\n📊 ANALIZANDO INTERACCIONES REALES...")
    
    position_stats = {1: 0, '2-3': 0, '4-10': 0, '11+': 0}
    total_clicks = 0
    interaction_types = {}
    
    try:
        with open(interactions_file, 'r', encoding='utf-8') as f:
            lines = list(f)
            print(f"   • Total líneas: {len(lines)}")
            
            for line in lines:
                try:
                    interaction = json.loads(line.strip())
                    interaction_type = interaction.get('interaction_type', 'unknown')
                    
                    # Contar tipos de interacción
                    if interaction_type not in interaction_types:
                        interaction_types[interaction_type] = 0
                    interaction_types[interaction_type] += 1
                    
                    if interaction_type == 'click':
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
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
    except Exception as e:
        print(f"❌ Error leyendo interacciones: {e}")
        return
    
    print(f"   • Total clicks: {total_clicks}")
    if total_clicks > 0:
        print(f"   • Posición 1: {position_stats[1]} ({position_stats[1]/total_clicks*100:.1f}%)")
        print(f"   • Posiciones 2-3: {position_stats['2-3']} ({position_stats['2-3']/total_clicks*100:.1f}%)")
        print(f"   • Posiciones 4-10: {position_stats['4-10']} ({position_stats['4-10']/total_clicks*100:.1f}%)")
        print(f"   • Posiciones 11+: {position_stats['11+']} ({position_stats['11+']/total_clicks*100:.1f}%)")
    
    print("\n📝 TIPOS DE INTERACCIÓN:")
    for itype, count in interaction_types.items():
        print(f"   • {itype}: {count}")
    
    if total_clicks == 0:
        print("❌ No hay clicks para entrenar")
        return
    
    # 5. Entrenar
    print("\n🎯 ENTRENANDO TRUE RLHF...")
    
    trained_count = 0
    failed_count = 0
    
    try:
        with open(interactions_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    interaction = json.loads(line.strip())
                    if interaction.get('interaction_type') == 'click':
                        context = interaction.get('context', {})
                        query = context.get('query', '').strip()
                        product_id = context.get('product_id')
                        position = context.get('position', 1)
                        
                        if not query or not product_id:
                            failed_count += 1
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
                            failed_count += 1
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
                        
                        # Extraer features del producto
                        product_features = {}
                        if hasattr(system, 'feature_engineer') and system.feature_engineer:
                            try:
                                # Simular query features para extracción
                                dummy_query_features = {'query_text': query}
                                product_features = system.feature_engineer.extract_product_features(
                                    product, dummy_query_features
                                )
                            except:
                                # Si falla, usar atributos básicos
                                product_features = {
                                    'price': getattr(product, 'price', 0.0),
                                    'rating': getattr(product, 'rating', 0.0),
                                    'match_score': 0.5
                                }
                        
                        # Entrenar
                        if hasattr(rl_ranker, 'learn_from_human_feedback'):
                            rl_ranker.learn_from_human_feedback(product, query, position, reward)
                        elif hasattr(rl_ranker, 'learn_from_feedback'):
                            rl_ranker.learn_from_feedback(
                                query_features={'query_text': query},
                                selected_product_id=product_id,
                                reward=reward,
                                context=context,
                                product_features=product_features
                            )
                        
                        trained_count += 1
                        
                        if trained_count % 10 == 0:
                            print(f"   ✅ {trained_count} interacciones entrenadas")
                            
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 5:  # Mostrar solo primeros 5 errores
                        print(f"   ⚠️  Error en línea {line_num}: {e}")
                    continue
    
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. Marcar como aprendido
    rl_ranker.has_learned = True
    
    print(f"\n📊 RESUMEN ENTRENAMIENTO:")
    print(f"   • Entrenados exitosamente: {trained_count}")
    print(f"   • Fallados: {failed_count}")
    print(f"   • Total procesados: {total_clicks}")
    
    if trained_count == 0:
        print("❌ No se pudo entrenar con ninguna interacción")
        return
    
    # 7. Mostrar estadísticas
    stats = {}
    if hasattr(rl_ranker, 'get_stats'):
        stats = rl_ranker.get_stats()
    elif hasattr(rl_ranker, 'get_learning_stats'):
        stats = rl_ranker.get_learning_stats()
    
    print("\n📊 ESTADÍSTICAS DEL APRENDIZAJE:")
    print(f"   • Feedback procesado: {stats.get('feedback_count', trained_count)}")
    print(f"   • Features aprendidas: {stats.get('weights_count', len(getattr(rl_ranker, 'feature_weights', {})))}")
    print(f"   • Ratio match/rating: {stats.get('match_vs_rating_ratio', 0):.2f}")
    
    # 8. Ajustar si ratio está desbalanceado
    ratio = stats.get('match_vs_rating_ratio', 0)
    
    if hasattr(rl_ranker, 'feature_weights'):
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
    if 'top_features' in stats and stats['top_features']:
        print("\n🔝 TOP 10 FEATURES APRENDIDAS:")
        for i, (feature, weight) in enumerate(stats['top_features'][:10], 1):
            if 'match' in feature.lower():
                icon = "🎯"
            elif 'rating' in feature.lower():
                icon = "⭐"
            elif 'preference_' in feature:
                icon = "❤️"
            elif 'position' in feature.lower():
                icon = "📊"
            else:
                icon = "📈"
            
            print(f"   {i:2d}. {icon} {feature[:30]:30} {weight:7.3f}")
    elif hasattr(rl_ranker, 'feature_weights') and rl_ranker.feature_weights:
        # Ordenar manualmente
        sorted_features = sorted(rl_ranker.feature_weights.items(), 
                                key=lambda x: abs(x[1]), reverse=True)[:10]
        print("\n🔝 TOP 10 FEATURES APRENDIDAS:")
        for i, (feature, weight) in enumerate(sorted_features, 1):
            if 'match' in feature.lower():
                icon = "🎯"
            elif 'rating' in feature.lower():
                icon = "⭐"
            elif 'preference_' in feature:
                icon = "❤️"
            elif 'position' in feature.lower():
                icon = "📊"
            else:
                icon = "📈"
            
            print(f"   {i:2d}. {icon} {feature[:30]:30} {weight:7.3f}")
    
    # 10. Guardar RL ranker
    save_dir = Path("data/cache")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        rl_ranker_path = save_dir / "rl_ranker_TRUE.pkl"
        with open(rl_ranker_path, 'wb') as f:
            pickle.dump(rl_ranker, f)
        print(f"\n💾 RL ranker guardado: {rl_ranker_path}")
    except Exception as e:
        print(f"❌ Error guardando RL ranker: {e}")
    
    # 11. Actualizar sistema y guardar
    try:
        system.rl_ranker = rl_ranker
        system_path_with_rl = save_dir / "unified_system_with_TRUE_rl.pkl"
        with open(system_path_with_rl, 'wb') as f:
            pickle.dump(system, f)
        print(f"💾 Sistema actualizado guardado: {system_path_with_rl}")
    except Exception as e:
        print(f"❌ Error guardando sistema actualizado: {e}")
    
    # 12. Salud del aprendizaje
    if hasattr(rl_ranker, 'get_learning_health'):
        health = rl_ranker.get_learning_health()
        print(f"\n🏥 SALUD DEL APRENDIZAJE: {health.get('status', 'UNKNOWN')}")
        print(f"   • Puntuación: {health.get('score', 0)}/5")
        
        if health.get('issues'):
            print(f"   ⚠️  Issues:")
            for issue in health['issues'][:3]:  # Mostrar solo primeros 3
                print(f"      • {issue}")
    else:
        # Salud básica manual
        print(f"\n🏥 SALUD DEL APRENDIZAJE:")
        print(f"   • Interacciones entrenadas: {trained_count}")
        print(f"   • Has learned: {rl_ranker.has_learned}")
        print(f"   • Features count: {len(getattr(rl_ranker, 'feature_weights', {}))}")
    
    # 13. Instrucciones para evaluación
    print("\n" + "="*80)
    print("✅ TRUE RLHF ENTRENADO CORRECTAMENTE")
    print("="*80)
    print("\n📋 PRÓXIMOS PASOS:")
    print("1. Evaluar el modelo entrenado:")
    print("   python evaluador_avanzado_rlhf.py")
    print("\n2. Probar en tiempo real:")
    print("   python -m src.unified_system")

if __name__ == "__main__":
    main()