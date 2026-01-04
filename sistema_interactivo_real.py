# sistema_interactivo_real.py
"""
SISTEMA INTERACTIVO REAL para obtener feedback REAL de usuario
Versión completa y bien estructurada
"""
import json
import pickle
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SistemaInteractivoReal:
    """Sistema completo para obtener feedback REAL de usuarios"""
    
    def __init__(self):
        print("\n" + "="*80)
        print("🚀 SISTEMA INTERACTIVO REAL - PARA DATOS REALES")
        print("="*80)
        
        # Configurar paths
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.cache_dir = self.data_dir / "cache"
        self.interactions_dir = self.data_dir / "interactions"
        
        # Crear directorios
        self.interactions_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivos de datos
        self.interactions_file = self.interactions_dir / "real_interactions.jsonl"
        self.ground_truth_file = self.interactions_dir / "ground_truth_REAL.json"
        
        # Inicializar
        self.system = None
        self.canonical_products = []
        self.current_query = None
        self.current_results = None
        self.interaction_count = 0
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Cargar sistema
        self.cargar_sistema()
        
        print(f"\n📝 Session ID: {self.session_id}")
        print(f"📊 Productos cargados: {len(self.canonical_products):,}")
        print(f"💾 Interacciones se guardarán en: {self.interactions_file}")
        print("\n💡 COMANDOS: query [texto], click [número], stats, help, exit")
    
    def cargar_sistema(self):
        """Carga el sistema desde caché o lo crea si no existe"""
        print("\n🔧 Cargando sistema...")
        
        # Verificar si existe sistema en caché
        system_cache = self.cache_dir / "unified_system.pkl"
        
        if system_cache.exists():
            print("📥 Cargando desde caché...")
            try:
                with open(system_cache, 'rb') as f:
                    self.system = pickle.load(f)
                
                # Extraer componentes necesarios
                if hasattr(self.system, 'canonical_products'):
                    self.canonical_products = self.system.canonical_products
                    print(f"✅ Sistema cargado: {len(self.canonical_products):,} productos")
                
                if hasattr(self.system, 'canonicalizer'):
                    print("✅ Canonicalizer cargado")
                
                if hasattr(self.system, 'vector_store'):
                    print("✅ Vector store cargado")
                    
                if hasattr(self.system, 'rl_ranker'):
                    print("✅ RL Ranker cargado")
                
                return True
                
            except Exception as e:
                print(f"❌ Error cargando caché: {e}")
        
        # Si no hay caché, cargar desde raw
        print("⚠️  No hay caché, cargando desde archivos raw...")
        return self.cargar_desde_raw()
    
    def cargar_desde_raw(self):
        """Carga productos desde archivos raw"""
        try:
            # Importar dinámicamente para evitar errores
            from src.data.loader import load_raw_products
            from src.data.canonicalizer import ProductCanonicalizer
            from src.data.vector_store import ImmutableVectorStore
            
            print("📥 Cargando productos raw...")
            raw_products = load_raw_products(limit=100000)
            
            if not raw_products:
                print("❌ No se pudieron cargar productos raw")
                return False
            
            print(f"📦 {len(raw_products):,} productos raw cargados")
            
            # Canonizar
            print("🔧 Canonizando productos...")
            canonicalizer = ProductCanonicalizer()
            self.canonical_products = canonicalizer.batch_canonicalize(raw_products)
            
            print(f"✅ {len(self.canonical_products):,} productos canonizados")
            
            # Crear objeto sistema mínimo
            class MinimalSystem:
                def __init__(self, products, canonicalizer):
                    self.canonical_products = products
                    self.canonicalizer = canonicalizer
                    self.vector_store = None
                    self.rl_ranker = None
            
            self.system = MinimalSystem(self.canonical_products, canonicalizer)
            
            # Construir vector store
            print("📚 Construyendo vector store...")
            self.system.vector_store = ImmutableVectorStore(dimension=384)
            self.system.vector_store.build_index(self.canonical_products)
            
            print("✅ Sistema inicializado correctamente")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando desde raw: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def buscar_productos(self, query_text: str, k: int = 20):
        """Busca productos usando el vector store"""
        if not self.system or not hasattr(self.system, 'canonicalizer'):
            print("❌ Sistema no inicializado correctamente")
            return []
        
        try:
            # Obtener embedding de la query
            query_embedding = self.system.canonicalizer.embedding_model.encode(
                query_text, normalize_embeddings=True
            )
            
            # Buscar en vector store
            if hasattr(self.system, 'vector_store') and self.system.vector_store:
                results = self.system.vector_store.search(query_embedding, k=k)
                
                # Añadir similitud a cada producto
                query_norm = query_embedding / np.linalg.norm(query_embedding)
                for product in results:
                    if hasattr(product, 'content_embedding'):
                        prod_embedding = product.content_embedding
                        prod_norm = prod_embedding / np.linalg.norm(prod_embedding)
                        product.similarity = float(np.dot(query_norm, prod_norm))
                    else:
                        product.similarity = 0.0
                
                return results
            else:
                print("❌ Vector store no disponible")
                return []
                
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            return []
    
    def guardar_interaccion(self, tipo: str, contexto: Dict[str, Any]):
        """Guarda una interacción REAL en el archivo"""
        interaccion = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'interaction_type': tipo,
            'context': contexto
        }
        
        try:
            with open(self.interactions_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(interaccion, ensure_ascii=False) + '\n')
            
            self.interaction_count += 1
            return True
            
        except Exception as e:
            print(f"❌ Error guardando interacción: {e}")
            return False
    
    def procesar_query(self, query_text: str):
        """Procesa una query y muestra resultados"""
        print(f"\n🔍 Buscando: '{query_text}'")
        
        # Buscar productos
        results = self.buscar_productos(query_text, k=100)
        
        if not results:
            print("❌ No se encontraron resultados")
            return
        
        print(f"📦 {len(results)} resultados encontrados")
        print("-" * 100)
        
        # Mostrar resultados
        productos_mostrados = []
        for i, product in enumerate(results[:100], 1):
            # Extraer información
            titulo = getattr(product, 'title', 'Sin título')
            categoria = getattr(product, 'category', 'N/A')
            precio = getattr(product, 'price', 0)
            rating = getattr(product, 'rating', 0)
            similarity = getattr(product, 'similarity', 0)
            
            # Truncar
            if len(titulo) > 60:
                titulo_display = titulo[:57] + "..."
            else:
                titulo_display = titulo
            
            # Formatear
            precio_str = f"${precio:.2f}" if precio else "$  N/A"
            rating_str = f"{rating:.1f}⭐" if rating else "N/A⭐"
            sim_str = f"{similarity:.3f}"
            
            print(f"{i:2d}. {titulo_display}")
            print(f"    📂 {categoria:20} 💰 {precio_str:10} {rating_str:8} 🔍 {sim_str}")
            print()
            
            # Guardar para clicks
            productos_mostrados.append({
                'id': getattr(product, 'id', f'prod_{i}'),
                'title': titulo,
                'position': i
            })
        
        print("-" * 100)
        print(f"🎯 Usa 'click [número]' para seleccionar productos RELEVANTES")
        print(f"   Ejemplo: 'click 1' para seleccionar el primer producto")
        
        # Guardar para referencia
        self.current_query = query_text
        self.current_results = productos_mostrados
        
        # Guardar interacción de query
        self.guardar_interaccion('query', {
            'query': query_text,
            'results_count': len(results),
            'timestamp': datetime.now().isoformat()
        })
    
    def procesar_click(self, posicion_str: str):
        """Procesa un click REAL en un producto"""
        if not self.current_query or not self.current_results:
            print("❌ Primero ejecuta una búsqueda con 'query [texto]'")
            return
        
        try:
            posicion = int(posicion_str) - 1
            
            if 0 <= posicion < len(self.current_results):
                producto = self.current_results[posicion]
                
                print(f"\n✅ CLICK REGISTRADO en posición {posicion + 1}")
                print(f"   📛 Producto: {producto['title'][:80]}...")
                print(f"   🆔 ID: {producto['id']}")
                print(f"   🔍 Query: '{self.current_query}'")
                print(f"   📊 Este producto fue considerado RELEVANTE para esta búsqueda")
                
                # Guardar interacción REAL
                self.guardar_interaccion('click', {
                    'query': self.current_query,
                    'product_id': producto['id'],
                    'position': posicion + 1,
                    'product_title': producto['title'],
                    'timestamp': datetime.now().isoformat(),
                    'is_relevant': True,
                    'feedback_type': 'explicit_click'
                })
                
                print(f"\n📈 Total clicks en esta sesión: {self.interaction_count}")
                
                # Sugerir entrenar RL si hay suficientes clicks
                if self.interaction_count >= 10:
                    print(f"\n💡 ¡Ya tienes {self.interaction_count} clicks!")
                
            else:
                print(f"❌ Posición inválida. Usa 1-{len(self.current_results)}")
                
        except ValueError:
            print("❌ Posición debe ser un número (ej: 'click 1')")
    
    def mostrar_estadisticas(self):
        """Muestra estadísticas de la sesión"""
        print("\n📊 ESTADÍSTICAS DE LA SESIÓN")
        print("-" * 50)
        print(f"   Sesión: {self.session_id}")
        print(f"   Total interacciones: {self.interaction_count}")
        print(f"   Archivo: {self.interactions_file}")
        
        # Contar clicks vs queries
        if self.interactions_file.exists():
            clicks = 0
            queries = 0
            
            with open(self.interactions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if data.get('interaction_type') == 'click':
                            clicks += 1
                        elif data.get('interaction_type') == 'query':
                            queries += 1
                    except:
                        continue
            
            print(f"   • Queries ejecutadas: {queries}")
            print(f"   • Clicks registrados: {clicks}")
            
            if clicks > 0:
                print(f"\n🎯 CONSEJO: Con {clicks} clicks ya puedes:")
        
        print("-" * 50)
    
    def mostrar_ayuda(self):
        """Muestra ayuda de comandos"""
        print("\n" + "="*80)
        print("📖 AYUDA - COMANDOS DEL SISTEMA INTERACTIVO")
        print("="*80)
        print("\n🎯 OBJETIVO: Obtener datos REALES de usuario para entrenar RL")
        print("   Cada CLICK que hagas se guardará como feedback REAL")
        print()
        print("📋 COMANDOS:")
        print("  query [texto]        - Buscar productos (ej: 'query car parts')")
        print("  click [número]       - Click en producto (GUARDA DATO REAL)")
        print("  stats                - Ver estadísticas")
        print("  help                 - Mostrar esta ayuda")
        print("  exit                 - Guardar y salir")
        print()
        print("💡 EJEMPLO DE USO:")
        print("  1. query car parts")
        print("  2. Revisa resultados")
        print("  3. click 1 (selecciona el más relevante)")
        print("  4. click 3 (selecciona otro relevante)")
        print("  5. Repite con diferentes búsquedas")
        print()
        print("🎓 RECOMENDACIONES:")
        print("  • Haz clicks en productos que realmente sean relevantes")
        print("  • Varía las búsquedas (car parts, beauty products, books, etc.)")
        print("  • Objetivo mínimo: 20-30 clicks para buen entrenamiento")
        print("="*80)
    
    def crear_ground_truth_automatico(self):
        """Crea ground truth automáticamente al salir"""
        if not self.interactions_file.exists() or self.interaction_count == 0:
            print("⚠️  No hay interacciones para crear ground truth")
            return
        
        print("\n📝 Creando ground truth REAL automáticamente...")
        
        ground_truth = {}
        total_clicks = 0
        
        with open(self.interactions_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('interaction_type') == 'click':
                        query = data.get('context', {}).get('query')
                        product_id = data.get('context', {}).get('product_id')
                        
                        if query and product_id:
                            if query not in ground_truth:
                                ground_truth[query] = []
                            if product_id not in ground_truth[query]:
                                ground_truth[query].append(product_id)
                                total_clicks += 1
                except:
                    continue
        
        if ground_truth:
            with open(self.ground_truth_file, 'w', encoding='utf-8') as f:
                json.dump(ground_truth, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Ground truth REAL creado:")
            print(f"   • {len(ground_truth)} queries con clicks")
            print(f"   • {total_clicks} productos relevantes totales")
            print(f"   • Guardado en: {self.ground_truth_file}")
        else:
            print("❌ No se pudieron extraer clicks para ground truth")
    
    def ejecutar(self):
        """Bucle principal del sistema interactivo"""
        print("\n🎮 ¡COMIENZA A OBTENER DATOS REALES!")
        print("   Cada CLICK que hagas será feedback REAL para entrenar RL")
        
        while True:
            try:
                # Prompt
                comando = input("\n👉 ").strip()
                
                if not comando:
                    continue
                
                elif comando.lower() == "exit":
                    self.crear_ground_truth_automatico()
                    print(f"\n👋 ¡Adiós! Sesión guardada.")
                    print(f"📊 Total interacciones: {self.interaction_count}")
                    print(f"💾 Archivo: {self.interactions_file}")
                    if self.ground_truth_file.exists():
                        print(f"🎯 Ground truth: {self.ground_truth_file}")
                    break
                
                elif comando.lower() == "help":
                    self.mostrar_ayuda()
                
                elif comando.lower() == "stats":
                    self.mostrar_estadisticas()
                
                elif comando.lower().startswith("query "):
                    query_text = comando[6:].strip()
                    if query_text:
                        self.procesar_query(query_text)
                    else:
                        print("❌ Debes proporcionar un texto de búsqueda")
                        print("   Ejemplo: query car parts")
                
                elif comando.lower().startswith("click "):
                    posicion = comando[6:].strip()
                    if posicion:
                        self.procesar_click(posicion)
                    else:
                        print("❌ Debes proporcionar una posición")
                        print("   Ejemplo: click 1")
                
                else:
                    # Si no es un comando reconocido, asumir que es una query
                    print(f"⚠️  Comando no reconocido. Asumiendo que es una búsqueda...")
                    self.procesar_query(comando)
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrumpido por usuario")
                self.crear_ground_truth_automatico()
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

def main():
    """Función principal"""
    print("\n🚀 INICIANDO SISTEMA INTERACTIVO REAL")
    print("   Versión: 1.0 - Para obtención de datos REALES")
    
    try:
        sistema = SistemaInteractivoReal()
        sistema.ejecutar()
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()