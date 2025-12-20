# run_interactive_with_logging.py
#!/usr/bin/env python
"""
Sistema interactivo con LOGGING COMPLETO de todas las interacciones
"""
import sys
from pathlib import Path
import logging
import json
from datetime import datetime

# Configurar paths
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Reducir logs para mejor visibilidad
logging.basicConfig(level=logging.WARNING)

from src.main import RAGRLSystem
from src.data.loader import load_raw_products

class InteractiveSystemWithLogging:
    def __init__(self):
        print("\n" + "="*80)
        print("🚀 SISTEMA INTERACTIVO CON LOGGING (PARA EVALUACIÓN)")
        print("="*80)
        
        # Configurar logging de interacciones
        self.interactions_file = Path("data/interactions/real_interactions.jsonl")
        self.interactions_file.parent.mkdir(parents=True, exist_ok=True)
        self.interactions = []
        
        print(f"\n📁 Las interacciones se guardarán en: {self.interactions_file}")
        print("💡 Todos los clicks quedarán registrados para evaluación posterior")
        
        # Cargar TODOS los datos
        print("\n📥 Cargando dataset completo...")
        try:
            raw_products = load_raw_products(limit=None)  # Sin límite
            print(f"✅ {len(raw_products):,} productos cargados")
        except Exception as e:
            print(f"❌ Error cargando productos: {e}")
            raw_products = []
        
        # Inicializar sistema
        print("🔧 Inicializando sistema...")
        self.system = RAGRLSystem('config/config.yaml')
        
        if raw_products:
            self.system.initialize_system(raw_products)
        
        # Estado de la sesión
        self.session_id = f"eval_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_mode = "with_rlhf"
        self.last_query = None
        self.last_results = None
        self.interaction_count = 0
        
        print(f"\n📝 Session ID: {self.session_id}")
        print(f"📊 Modo actual: {self.current_mode}")
        print(f"🎯 Objetivo: Hacer CLICKS para crear ground truth REAL")
        print("\n💡 Escribe 'help' para ver comandos")
        
    def log_interaction(self, interaction_type, context, details=None):
        """Guarda una interacción en el archivo JSONL"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'interaction_type': interaction_type,
            'context': context,
            'details': details or {}
        }
        
        # Guardar en memoria
        self.interactions.append(interaction)
        
        # Guardar en archivo (append mode)
        try:
            with open(self.interactions_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(interaction, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️  Error guardando interacción: {e}")
        
        self.interaction_count += 1
        
    def run(self):
        """Bucle principal interactivo"""
        print("\n🎮 ¡COMIENZA LA EVALUACIÓN! Haz clicks para crear datos reales")
        print("   Cada click se guardará para evaluar RLHF después")
        
        while True:
            try:
                user_input = input("\n👉 ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "exit":
                    self.save_and_exit()
                    break
                
                elif user_input.lower() == "help":
                    self.show_help()
                
                elif user_input.lower().startswith("query "):
                    query_text = user_input[6:]
                    if query_text:
                        self.handle_query(query_text)
                    else:
                        print("❌ Debes proporcionar un texto de búsqueda")
                        print("   Ejemplo: query car parts")
                
                elif user_input.lower().startswith("mode "):
                    mode = user_input[5:]
                    self.handle_mode_change(mode)
                
                elif user_input.lower().startswith("click "):
                    position = user_input[6:]
                    self.handle_click(position)
                
                elif user_input.lower() == "stats":
                    self.show_stats()
                
                elif user_input.lower() == "evaluate":
                    self.run_evaluation_now()
                
                elif user_input.lower() == "reset":
                    self.reset_interactions()
                
                else:
                    # Si no es un comando, asumir que es una query
                    self.handle_query(user_input)
                    
            except KeyboardInterrupt:
                self.save_and_exit()
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_help(self):
        """Muestra ayuda"""
        print("\n" + "="*80)
        print("COMANDOS DE EVALUACIÓN:")
        print("="*80)
        print("  [texto]                 - Buscar (ej: 'car parts')")
        print("  query [texto]           - Buscar con comando explícito")
        print("  mode [baseline|features|rlhf] - Cambiar modo")
        print("  click [número]          - Click en producto (ESTO GUARDA DATOS REALES)")
        print("  evaluate                - Evaluar ahora con los clicks guardados")
        print("  stats                   - Ver estadísticas de interacciones")
        print("  reset                   - Limpiar interacciones guardadas")
        print("  exit                    - Guardar y salir")
        print("\n💡 IMPORTANTE: Haz CLICKS para crear datos de evaluación reales")
        print("   Cada click se guarda automáticamente")
        print("="*80)
    
    def handle_query(self, query_text: str):
        """Procesa una query y guarda la interacción"""
        print(f"\n🔍 Buscando: '{query_text}'")
        print(f"   Modo: {self.current_mode}")
        
        # Procesar query
        response = self.system._process_query_mode(query_text, self.current_mode)
        
        if response.get('success'):
            products = response.get('products', [])
            
            # Loggear la query (sin click)
            self.log_interaction(
                interaction_type='query',
                context={
                    'query': query_text,
                    'mode': self.current_mode,
                    'results_count': len(products)
                },
                details={
                    'top_products': [p.get('title', '')[:50] for p in products[:3]]
                }
            )
            
            print(f"\n📦 Resultados ({len(products)} productos):")
            print("-" * 80)
            
            # Mostrar resultados con scores
            for i, product in enumerate(products[:10], 1):
                title = product.get('title', 'Sin título')
                category = product.get('category', 'N/A')
                price = product.get('price', 0)
                rating = product.get('rating', 0)
                score = product.get('similarity_score', 0)
                
                # Truncar título
                if len(title) > 50:
                    title = title[:47] + "..."
                
                print(f"  {i:2d}. {title}")
                
                # Formatear
                price_str = f"${price:7.2f}" if isinstance(price, (int, float)) else "$    N/A"
                rating_str = f"{rating:4.1f}" if isinstance(rating, (int, float)) else " N/A"
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "0.0000"
                print(f"      📂 {category:20} 💰 {price_str} ⭐ {rating_str} 📊 {score_str}")
            
            print("-" * 80)
            
            # Guardar para clicks posteriores
            self.last_query = query_text
            self.last_results = products
            
            print(f"🎯 IMPORTANTE: Usa 'click [número]' para guardar feedback REAL")
            print(f"   Ejemplo: 'click 1' para seleccionar el primer producto")
            
        else:
            print(f"❌ Error: {response.get('error')}")
    
    def handle_mode_change(self, mode: str):
        """Cambia el modo de funcionamiento"""
        mode_map = {
            "baseline": "baseline",
            "features": "with_features",
            "rlhf": "with_rlhf"
        }
        
        if mode in mode_map:
            old_mode = self.current_mode
            self.current_mode = mode_map[mode]
            print(f"✅ Modo cambiado: {old_mode} → {mode} ({self.current_mode})")
            
            # Loggear cambio de modo
            self.log_interaction(
                interaction_type='mode_change',
                context={
                    'old_mode': old_mode,
                    'new_mode': self.current_mode
                }
            )
        else:
            print(f"❌ Modo no válido. Usa: baseline, features, rlhf")
    
    def handle_click(self, position_str: str):
        """Registra un click REAL y aplica aprendizaje RL"""
        if not hasattr(self, 'last_results') or not self.last_results:
            print("❌ Primero ejecuta una búsqueda")
            return
        
        try:
            position = int(position_str) - 1
            
            if 0 <= position < len(self.last_results):
                product = self.last_results[position]
                
                print(f"\n🎯 CLICK REGISTRADO en producto {position + 1}:")
                print(f"   📛 {product.get('title', 'Sin título')[:60]}")
                print(f"   🆔 ID: {product.get('id', 'N/A')}")
                print(f"   📊 Score: {product.get('similarity_score', 0):.4f}")
                print(f"   🔍 Query: '{self.last_query}'")
                
                # LOG IMPORTANTE: Guardar el click REAL
                self.log_interaction(
                    interaction_type='click',
                    context={
                        'query': self.last_query,
                        'product_id': product.get('id'),
                        'position': position + 1,
                        'mode': self.current_mode,
                        'product_score': product.get('similarity_score', 0)
                    },
                    details={
                        'product_title': product.get('title', ''),
                        'product_category': product.get('category', '')
                    }
                )
                
                # Aplicar aprendizaje RL (solo si estamos en modo RLHF)
                if self.current_mode == "with_rlhf":
                    print(f"   🧠 Aplicando aprendizaje RL...")
                    try:
                        feedback_data = {
                            'interaction_type': 'click',
                            'context': {
                                'query': self.last_query,
                                'product_id': product.get('id'),
                                'position': position + 1,
                                'product_score': product.get('similarity_score', 0)
                            }
                        }
                        
                        learning_result = self.system.process_feedback(feedback_data)
                        if learning_result.get('success'):
                            print(f"   ✅ RL aprendió de este feedback")
                            print(f"   📈 Feedback procesado #{self.interaction_count}")
                        else:
                            print(f"   ⚠️  RL no pudo aprender: {learning_result.get('error', 'Unknown')}")
                    except Exception as e:
                        print(f"   ⚠️  Error en RL: {e}")
                
                print(f"✅ Click guardado para evaluación futura")
                print(f"📊 Total clicks en esta sesión: {self.interaction_count}")
                
                # Mostrar progreso de evaluación
                if self.interaction_count >= 3:
                    print(f"🎯 ¡Excelente! Con {self.interaction_count} clicks ya puedes evaluar")
                    print(f"💡 Usa 'evaluate' para ver resultados de mejora RL")
                
            else:
                print(f"❌ Posición inválida. Usa 1-{len(self.last_results)}")
                
        except ValueError:
            print("❌ Posición debe ser un número")
    
    def show_stats(self):
        """Muestra estadísticas de las interacciones"""
        print("\n📊 ESTADÍSTICAS DE LA SESIÓN:")
        print("-" * 40)
        print(f"   Sesión: {self.session_id}")
        print(f"   Modo actual: {self.current_mode}")
        print(f"   Total interacciones: {self.interaction_count}")
        
        # Contar clicks por tipo
        click_count = len([i for i in self.interactions if i['interaction_type'] == 'click'])
        query_count = len([i for i in self.interactions if i['interaction_type'] == 'query'])
        
        print(f"   • Queries ejecutadas: {query_count}")
        print(f"   • Clicks registrados: {click_count}")
        
        # Estadísticas de RL
        if hasattr(self.system, 'rl_ranker'):
            rl_stats = self.system.rl_ranker.get_learning_stats()
            print(f"   🧠 APRENDIZAJE RL:")
            print(f"      Aprendido: {'Sí' if rl_stats.get('has_learned') else 'No'}")
            print(f"      Feedback recibido: {rl_stats.get('feedback_count', 0)}")
        
        # Queries con clicks
        if click_count > 0:
            queries_with_clicks = {}
            for interaction in self.interactions:
                if interaction['interaction_type'] == 'click':
                    query = interaction['context'].get('query')
                    if query:
                        queries_with_clicks[query] = queries_with_clicks.get(query, 0) + 1
            
            print(f"\n   📝 Queries con clicks ({len(queries_with_clicks)}):")
            for query, count in sorted(queries_with_clicks.items())[:5]:
                print(f"      • '{query[:30]}...': {count} clicks")
        
        print("-" * 40)
        print(f"💾 Archivo: {self.interactions_file}")
        print(f"   Tamaño: {self.interactions_file.stat().st_size if self.interactions_file.exists() else 0} bytes")
    
    def run_evaluation_now(self):
        """Ejecuta evaluación inmediata con los datos guardados"""
        print("\n" + "="*80)
        print("📊 EVALUACIÓN AUTOMÁTICA CON DATOS REALES")
        print("="*80)
        
        if self.interaction_count < 3:
            print(f"⚠️  Necesitas al menos 3 clicks para evaluar")
            print(f"   Clicks actuales: {self.interaction_count}")
            print(f"   Haz más búsquedas y clicks con 'click [número]'")
            return
        
        # Extraer ground truth de los clicks
        relevance_labels = {}
        for interaction in self.interactions:
            if interaction['interaction_type'] == 'click':
                query = interaction['context'].get('query')
                product_id = interaction['context'].get('product_id')
                
                if query and product_id:
                    if query not in relevance_labels:
                        relevance_labels[query] = []
                    if product_id not in relevance_labels[query]:
                        relevance_labels[query].append(product_id)
        
        print(f"\n📝 Ground truth extraído:")
        print(f"   • {len(relevance_labels)} queries con clicks")
        print(f"   • {sum(len(v) for v in relevance_labels.values())} productos relevantes")
        
        # Guardar ground truth para evaluación posterior
        ground_truth_file = Path("data/interactions/relevance_labels_real.json")
        with open(ground_truth_file, 'w', encoding='utf-8') as f:
            json.dump(relevance_labels, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Ground truth guardado en: {ground_truth_file}")
        
        # Ejecutar evaluación automática
        self.run_automatic_evaluation(relevance_labels)
    
    def run_automatic_evaluation(self, relevance_labels):
        """Ejecuta evaluación automática"""
        print("\n🔬 EJECUTANDO EVALUACIÓN DE 3 MODOS...")
        
        queries_to_test = list(relevance_labels.keys())[:5]  # Máximo 5 queries
        modes = [
            ("Baseline", "baseline"),
            ("RAG+Features", "with_features"), 
            ("RAG+RLHF", "with_rlhf")
        ]
        
        resultados = {nombre: [] for nombre, _ in modes}
        
        for query_idx, query in enumerate(queries_to_test):
            print(f"\n   🔍 Query {query_idx+1}/{len(queries_to_test)}: '{query}'")
            
            for mode_name, mode in modes:
                try:
                    response = self.system._process_query_mode(query, mode)
                    
                    if response.get('success'):
                        # Extraer productos rankeados
                        ranked_products = [p.get('id') for p in response.get('products', [])]
                        
                        # Calcular métricas
                        relevant_ids = relevance_labels.get(query, [])
                        
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
                                'relevant_found': len(relevant_in_top_5)
                            }
                            
                            resultados[mode_name].append(metrics)
                            print(f"     ✅ {mode_name}: ", end="")
                            print(f"P@5={precision_at_5:.3f}, ", end="")
                            print(f"R@5={recall_at_5:.3f}, ", end="")
                            print(f"MRR={mrr:.3f}")
                            
                        else:
                            print(f"     ⚠️  {mode_name}: Sin ground truth para esta query")
                            
                except Exception as e:
                    print(f"     ❌ {mode_name}: Error - {str(e)[:50]}")
        
        # Mostrar resumen
        print("\n" + "="*80)
        print("📈 RESUMEN DE EVALUACIÓN")
        print("="*80)
        
        for mode_name, metrics_list in resultados.items():
            if metrics_list:
                valid_metrics = [m for m in metrics_list if m.get('has_ground_truth', False)]
                if valid_metrics:
                    precision_scores = [m.get('precision@5', 0) for m in valid_metrics]
                    avg_precision = sum(precision_scores) / len(precision_scores)
                    
                    print(f"\n{mode_name}:")
                    print(f"   • Queries evaluadas: {len(valid_metrics)}")
                    print(f"   • Precision@5 promedio: {avg_precision:.3f}")
                    print(f"   • Rango Precision@5: {min(precision_scores):.3f} - {max(precision_scores):.3f}")
        
        # Calcular mejoras
        if 'Baseline' in resultados and 'RAG+RLHF' in resultados:
            baseline_scores = [m.get('precision@5', 0) for m in resultados['Baseline'] if m.get('has_ground_truth', False)]
            rlhf_scores = [m.get('precision@5', 0) for m in resultados['RAG+RLHF'] if m.get('has_ground_truth', False)]
            
            if baseline_scores and rlhf_scores:
                baseline_avg = sum(baseline_scores) / len(baseline_scores)
                rlhf_avg = sum(rlhf_scores) / len(rlhf_scores)
                
                if baseline_avg > 0:
                    mejora = ((rlhf_avg - baseline_avg) / baseline_avg) * 100
                    print(f"\n🎯 MEJORA RLHF vs BASELINE: {mejora:+.1f}%")
                    
                    if mejora > 0:
                        print(f"✅ ¡RLHF MEJORA EL SISTEMA BASADO EN FEEDBACK REAL!")
                    else:
                        print(f"⚠️  RLHF no muestra mejora aún")
        
        print("\n💡 Para evaluación completa, ejecuta después: python verificador_final.py")
    
    def reset_interactions(self):
        """Limpia las interacciones guardadas"""
        print("\n⚠️  ¿Estás seguro de limpiar TODAS las interacciones?")
        print("   Esto borrará todos los clicks guardados.")
        confirm = input("   Escribe 'SI' para confirmar: ").strip().upper()
        
        if confirm == "SI":
            self.interactions = []
            self.interaction_count = 0
            
            if self.interactions_file.exists():
                self.interactions_file.unlink()
            
            print("✅ Interacciones limpiadas")
        else:
            print("❌ Cancelado")
    
    def save_and_exit(self):
        """Guarda todo y sale"""
        print("\n💾 Guardando todas las interacciones...")
        
        # Guardar resumen
        summary_file = Path(f"data/interactions/summary_{self.session_id}.json")
        summary = {
            'session_id': self.session_id,
            'total_interactions': self.interaction_count,
            'click_count': len([i for i in self.interactions if i['interaction_type'] == 'click']),
            'query_count': len([i for i in self.interactions if i['interaction_type'] == 'query']),
            'modes_used': list(set(i['context'].get('mode', 'unknown') for i in self.interactions)),
            'timestamp_end': datetime.now().isoformat()
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✅ {self.interaction_count} interacciones guardadas")
        print(f"📄 Resumen en: {summary_file}")
        print(f"📊 Datos completos en: {self.interactions_file}")
        print("\n👋 ¡Adiós! Ejecuta 'python verificador_final.py' para evaluación completa")

if __name__ == "__main__":
    print("\n🚀 INICIANDO SISTEMA DE EVALUACIÓN CON LOGGING")
    print("   TODOS LOS CLICKS SE GUARDARÁN PARA EVALUACIÓN REAL\n")
    
    try:
        # Crear directorios necesarios
        Path("data/interactions").mkdir(parents=True, exist_ok=True)
        
        system = InteractiveSystemWithLogging()
        system.run()
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()