# run_fast_interactive.py
#!/usr/bin/env python
"""
Sistema interactivo RÁPIDO - Usa caché para cargar en segundos
"""
import sys
from pathlib import Path
import logging

# Configurar paths
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Reducir logs para mejor visibilidad
logging.basicConfig(level=logging.WARNING)

from src.main_optimized import OptimizedRAGRLSystem
import json
from datetime import datetime

class FastInteractiveSystem:
    def __init__(self, use_cache=True):
        print("\n" + "="*80)
        print("🚀 SISTEMA INTERACTIVO RÁPIDO")
        print("="*80)
        print(f"📦 Usando caché: {'SÍ' if use_cache else 'NO'}")
        
        # Inicializar sistema OPTIMIZADO
        print("\n⚡ Inicializando sistema (con caché)...")
        start_time = datetime.now()
        
        self.system = OptimizedRAGRLSystem('config/config.yaml', use_cache=use_cache)
        
        # Inicialización RÁPIDA
        success = self.system.initialize_with_cache(force_reload=False)
        
        if not success:
            print("❌ Error inicializando sistema")
            return
        
        init_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Sistema listo en {init_time:.1f} segundos")
        
        # Estado
        self.session_id = f"fast_session_{datetime.now().strftime('%H%M%S')}"
        self.current_mode = "with_rlhf"
        self.last_query = None
        self.last_results = None
        self.interaction_count = 0
        
        print(f"\n📊 Estadísticas:")
        print(f"   • Productos: {len(self.system.canonical_products):,}")
        print(f"   • Modo: {self.current_mode}")
        print(f"   • Caché: {'CARGADO' if self.system.cache_loaded else 'NUEVO'}")
        print(f"\n💡 Comandos: help, query [texto], mode [nombre], click [número], exit")
    
    def run(self):
        """Bucle principal"""
        while True:
            try:
                user_input = input("\n👉 ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "exit":
                    # Guardar estado antes de salir
                    self.system.save_rl_state()
                    print("\n💾 Estado RL guardado en caché")
                    print("👋 ¡Hasta luego!")
                    break
                
                elif user_input.lower() == "help":
                    self.show_help()
                
                elif user_input.lower().startswith("query "):
                    query_text = user_input[6:]
                    self.handle_query(query_text)
                
                elif user_input.lower().startswith("mode "):
                    mode = user_input[5:]
                    self.handle_mode_change(mode)
                
                elif user_input.lower().startswith("click "):
                    position = user_input[6:]
                    self.handle_click(position)
                
                elif user_input.lower() == "save":
                    self.save_snapshot()
                
                elif user_input.lower() == "stats":
                    self.show_stats()
                
                elif user_input.lower() == "evaluate":
                    self.run_evaluation()
                
                else:
                    # Query implícita
                    self.handle_query(user_input)
                    
            except KeyboardInterrupt:
                print("\n\n💾 Guardando estado RL...")
                self.system.save_rl_state()
                print("👋 ¡Hasta luego!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_help(self):
        """Muestra ayuda"""
        print("\n" + "="*80)
        print("COMANDOS RÁPIDOS (SISTEMA OPTIMIZADO):")
        print("="*80)
        print("  query [texto]     - Buscar productos")
        print("  mode [nombre]     - Cambiar modo (baseline/features/rlhf)")
        print("  click [número]    - Click en producto (guarda feedback)")
        print("  save              - Guardar snapshot del sistema")
        print("  stats             - Ver estadísticas")
        print("  evaluate          - Evaluar rendimiento")
        print("  help              - Mostrar esta ayuda")
        print("  exit              - Guardar y salir")
        print("\n💡 El sistema usa caché para cargar 90K productos en segundos")
        print("💡 Cada click se guarda automáticamente")
        print("="*80)
    
    def handle_query(self, query_text: str):
        """Procesa una query"""
        print(f"\n🔍 Buscando: '{query_text}'")
        print(f"   Modo: {self.current_mode}")
        
        response = self.system.process_query(query_text, use_rlhf=(self.current_mode=="with_rlhf"))
        
        if response.get('success'):
            products = response.get('products', [])
            
            print(f"\n📦 Resultados ({len(products)} productos):")
            print("-" * 80)
            
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
            
            # Guardar para clicks
            self.last_query = query_text
            self.last_results = products
            
            print(f"🎯 Usa 'click [número]' para guardar feedback")
            
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
        else:
            print(f"❌ Modo no válido. Usa: baseline, features, rlhf")
    
    def handle_click(self, position_str: str):
        """Registra un click"""
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
                print(f"   🔍 Query: '{self.last_query}'")
                
                # Aplicar aprendizaje RL
                if self.current_mode == "with_rlhf":
                    print(f"   🧠 Aplicando aprendizaje RL...")
                    
                    feedback_data = {
                        'interaction_type': 'click',
                        'context': {
                            'query': self.last_query,
                            'product_id': product.get('id'),
                            'position': position + 1
                        }
                    }
                    
                    result = self.system.process_feedback(feedback_data)
                    if result.get('success'):
                        print(f"   ✅ RL aprendió de este feedback")
                        print(f"   💾 Estado RL guardado en caché")
                    else:
                        print(f"   ⚠️  Error en RL: {result.get('error', 'Unknown')}")
                
                self.interaction_count += 1
                print(f"✅ Click procesado (total: {self.interaction_count})")
                
                if self.interaction_count >= 5:
                    print(f"🎯 ¡Excelente! Con {self.interaction_count} clicks puedes evaluar")
                    print(f"💡 Usa 'evaluate' para ver resultados")
                
            else:
                print(f"❌ Posición inválida. Usa 1-{len(self.last_results)}")
                
        except ValueError:
            print("❌ Posición debe ser un número")
    
    def save_snapshot(self):
        """Guarda snapshot del sistema"""
        snapshot_path = self.system.save_snapshot(f"interactive_{self.session_id}")
        print(f"\n💾 Snapshot guardado:")
        print(f"   • Directorio: {snapshot_path}")
        print(f"   • Productos: {len(self.system.canonical_products):,}")
        print(f"   • Interacciones: {self.interaction_count}")
        print(f"   • Modo RL: {self.current_mode}")
    
    def show_stats(self):
        """Muestra estadísticas"""
        print("\n📊 ESTADÍSTICAS DEL SISTEMA:")
        print("-" * 40)
        print(f"   Sesión: {self.session_id}")
        print(f"   Modo actual: {self.current_mode}")
        print(f"   Productos: {len(self.system.canonical_products):,}")
        print(f"   Clicks: {self.interaction_count}")
        print(f"   Caché: {'CARGADO' if self.system.cache_loaded else 'NUEVO'}")
        
        # Estadísticas RL
        if hasattr(self.system, 'rl_ranker'):
            rl_stats = self.system.rl_ranker.get_learning_stats()
            print(f"\n   🧠 APRENDIZAJE RL:")
            print(f"      Aprendido: {'Sí' if rl_stats.get('has_learned') else 'No'}")
            print(f"      Feedback recibido: {rl_stats.get('feedback_count', 0)}")
        
        print("-" * 40)
    
    def run_evaluation(self):
        """Ejecuta evaluación simple"""
        if self.interaction_count < 3:
            print(f"⚠️  Necesitas al menos 3 clicks para evaluar")
            print(f"   Clicks actuales: {self.interaction_count}")
            return
        
        print("\n" + "="*80)
        print("📊 EVALUACIÓN SIMPLE")
        print("="*80)
        
        # Aquí podrías integrar tu código de evaluación
        print("\n🎯 Usa estos comandos para evaluación completa:")
        print("   1. Guarda snapshot: save")
        print("   2. Ejecuta evaluador: python verificador_final.py")
        print("\n💡 Con el snapshot guardado, el evaluador cargará rápido")

if __name__ == "__main__":
    print("\n🚀 INICIANDO SISTEMA INTERACTIVO RÁPIDO")
    print("   Cargará desde caché si está disponible\n")
    
    try:
        system = FastInteractiveSystem(use_cache=True)
        system.run()
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()