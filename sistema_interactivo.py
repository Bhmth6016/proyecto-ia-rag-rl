# sistema_interactivo.py
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SistemaInteractivoReal:
    def __init__(self):
        print("\n" + "="*80)
        print("SISTEMA INTERACTIVO REAL - PARA DATOS REALES")
        print("="*80)
        
        self.interactions_file = Path("data/interactions/real_interactions.jsonl")
        self.ground_truth_file = Path("data/interactions/ground_truth_REAL.json")
        
        # Crear directorios si no existen
        self.interactions_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.system = None
        self.canonical_products = []
        self.current_query = None
        self.current_results = None
        self.interaction_count = 0
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.cargar_sistema()
        
        print("\n OBJETIVO: Obtener 30+ clicks REALES para entrenar RLHF")
        print(f" Session ID: {self.session_id}")
        print(f" Productos cargados: {len(self.canonical_products):,}")
        print(f" Interacciones se guardarán en: {self.interactions_file}")
        print("\n COMANDOS: query [texto], click [número], stats, help, exit")
    
    def cargar_sistema(self):
        print("\nCargando sistema...")
        
        system_cache = Path("data/cache/unified_system_v2.pkl")
        
        if system_cache.exists():
            try:
                from src.unified_system_v2 import UnifiedSystemV2
                self.system = UnifiedSystemV2.load_from_cache()
                
                if self.system and hasattr(self.system, 'canonical_products'):
                    self.canonical_products = self.system.canonical_products
                    print(f" Sistema V2 cargado: {len(self.canonical_products):,} productos")
                    return True
            except Exception as e:
                print(f"  Error cargando sistema V2: {e}")
        
        print("  Sistema V2 no encontrado. Ejecuta primero:")
        print("   python main.py init")
        return False
    
    def buscar_productos(self, query_text: str, k: int = 20):
        if not self.system or not hasattr(self.system, 'canonicalizer'):
            print("Sistema no inicializado correctamente")
            return []
        
        try:
            query_embedding = self.system.canonicalizer.embedding_model.encode(
                query_text, normalize_embeddings=True
            )
            
            if hasattr(self.system, 'vector_store') and self.system.vector_store:
                results = self.system.vector_store.search(query_embedding, k=k)
                return results
            else:
                print("Vector store no disponible")
                return []
                
        except Exception as e:
            print(f"Error en búsqueda: {e}")
            return []
    
    def guardar_interaccion(self, tipo: str, contexto: Dict[str, Any]):
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
            print(f"Error guardando interacción: {e}")
            return False
    
    def procesar_query(self, query_text: str):
        print(f"\n Buscando: '{query_text}'")
        
        results = self.buscar_productos(query_text, k=20)
        
        if not results:
            print("No se encontraron resultados")
            return
        
        print(f" {len(results)} resultados encontrados")
        print("-" * 100)
        
        productos_mostrados = []
        for i, product in enumerate(results[:20], 1):
            titulo = getattr(product, 'title', 'Sin título')
            categoria = getattr(product, 'category', 'N/A')
            precio = getattr(product, 'price', 0)
            rating = getattr(product, 'rating', 0)
            
            if len(titulo) > 60:
                titulo_display = titulo[:57] + "..."
            else:
                titulo_display = titulo
            
            precio_str = f"${precio:.2f}" if precio else "$  N/A"
            rating_str = f"{rating:.1f}⭐" if rating else "N/A⭐"
            
            print(f"{i:2d}. {titulo_display}")
            print(f"    {categoria:20} {precio_str:10} {rating_str}")
            print()
            
            productos_mostrados.append({
                'id': getattr(product, 'id', f'prod_{i}'),
                'title': titulo,
                'position': i
            })
        
        print("-" * 100)
        print(" Usa 'click [número]' para seleccionar productos RELEVANTES")
        print("   Ejemplo: 'click 1' para seleccionar el primer producto")
        print("   Objetivo: 30+ clicks para buen entrenamiento RLHF")
        
        self.current_query = query_text
        self.current_results = productos_mostrados
        
        self.guardar_interaccion('query', {
            'query': query_text,
            'results_count': len(results),
            'timestamp': datetime.now().isoformat()
        })
    
    def procesar_click(self, posicion_str: str):
        if not self.current_query or not self.current_results:
            print("Primero ejecuta una búsqueda con 'query [texto]'")
            return
        
        try:
            posicion = int(posicion_str) - 1
            
            if 0 <= posicion < len(self.current_results):
                producto = self.current_results[posicion]
                
                print(f"\n CLICK REGISTRADO en posición {posicion + 1}")
                print(f"   Producto: {producto['title'][:80]}...")
                print(f"   ID: {producto['id']}")
                print(f"   Query: '{self.current_query}'")
                print("    Este producto fue considerado RELEVANTE para esta búsqueda")
                
                self.guardar_interaccion('click', {
                    'query': self.current_query,
                    'product_id': producto['id'],
                    'position': posicion + 1,
                    'product_title': producto['title'],
                    'timestamp': datetime.now().isoformat(),
                    'is_relevant': True,
                    'feedback_type': 'explicit_click'
                })
                
                print(f"\n Total clicks en esta sesión: {self.interaction_count}")
                
                if self.interaction_count >= 30:
                    print(f"\n ¡Ya tienes {self.interaction_count} clicks! Suficiente para entrenar RLHF.")
                    print("   Puedes ejecutar: python main.py experimento")
                elif self.interaction_count >= 10:
                    print(f"\n ¡Ya tienes {self.interaction_count} clicks! Sigue recolectando para mejor entrenamiento.")
                
            else:
                print(f"Posición inválida. Usa 1-{len(self.current_results)}")
                
        except ValueError:
            print("Posición debe ser un número (ej: 'click 1')")
    
    def mostrar_estadisticas(self):
        print("\n ESTADÍSTICAS DE LA SESIÓN")
        print("-" * 50)
        print(f"   Sesión: {self.session_id}")
        print(f"   Total interacciones: {self.interaction_count}")
        print(f"   Archivo: {self.interactions_file}")
        
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
                    except json.JSONDecodeError:
                        continue
            
            print(f"   Queries ejecutadas: {queries}")
            print(f"   Clicks registrados: {clicks}")
            
            if clicks > 0:
                print(f"\n Con {clicks} clicks puedes:")
                if clicks >= 30:
                    print("    Entrenar RLHF robustamente")
                    print("    Ejecutar experimento completo")
                elif clicks >= 20:
                    print("     Entrenar RLHF básicamente")
                    print("     Ejecutar experimento pequeño")
                else:
                    print("    Necesitas más datos (objetivo: 30+ clicks)")
        
        print("-" * 50)
    
    def mostrar_ayuda(self):
        print("\n" + "="*80)
        print(" AYUDA - COMANDOS DEL SISTEMA INTERACTIVO")
        print("="*80)
        print("\n OBJETIVO: Obtener datos REALES de usuario para entrenar RL")
        print("   Cada CLICK que hagas se guardará como feedback REAL")
        print()
        print(" COMANDOS:")
        print("  query [texto]        - Buscar productos (ej: 'query car parts')")
        print("  click [número]       - Click en producto (GUARDA DATO REAL)")
        print("  stats                - Ver estadísticas")
        print("  help                 - Mostrar esta ayuda")
        print("  exit                 - Guardar y salir")
        print()
        print(" EJEMPLO DE USO:")
        print("  1. query car parts")
        print("  2. Revisa resultados")
        print("  3. click 1 (selecciona el más relevante)")
        print("  4. click 3 (selecciona otro relevante)")
        print("  5. Repite con diferentes búsquedas")
        print()
        print(" RECOMENDACIONES:")
        print("  • Haz clicks en productos que realmente sean relevantes")
        print("  • Varía las búsquedas (car parts, beauty products, books, etc.)")
        print("  • Objetivo mínimo: 30 clicks para buen entrenamiento")
        print("="*80)
    
    def crear_ground_truth_automatico(self):
        if not self.interactions_file.exists() or self.interaction_count == 0:
            print("No hay interacciones para crear ground truth")
            return
        
        print("\n Creando ground truth REAL automáticamente...")
        
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
                except json.JSONDecodeError:
                    continue
        
        if ground_truth:
            with open(self.ground_truth_file, 'w', encoding='utf-8') as f:
                json.dump(ground_truth, f, indent=2, ensure_ascii=False)
            
            print(" Ground truth REAL creado:")
            print(f"    {len(ground_truth)} queries con clicks")
            print(f"    {total_clicks} productos relevantes totales")
            print(f"    Guardado en: {self.ground_truth_file}")
        else:
            print("No se pudieron extraer clicks para ground truth")
    
    def ejecutar(self):
        print("\n ¡COMIENZA A OBTENER DATOS REALES!")
        print("   Cada CLICK que hagas será feedback REAL para entrenar RLHF")
        print("   Objetivo: 30+ clicks para experimento robusto")
        
        while True:
            try:
                comando = input("\nuser: ").strip()
                
                if not comando:
                    continue
                
                elif comando.lower() == "exit":
                    self.crear_ground_truth_automatico()
                    print("\n ¡Adiós! Sesión guardada.")
                    print(f" Total interacciones: {self.interaction_count}")
                    print(f" Archivo: {self.interactions_file}")
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
                        print("Debes proporcionar un texto de búsqueda")
                        print("   Ejemplo: query car parts")
                
                elif comando.lower().startswith("click "):
                    posicion = comando[6:].strip()
                    if posicion:
                        self.procesar_click(posicion)
                    else:
                        print("Debes proporcionar una posición")
                        print("   Ejemplo: click 1")
                
                else:
                    print("Comando no reconocido. Asumiendo que es una búsqueda...")
                    self.procesar_query(comando)
                    
            except KeyboardInterrupt:
                print("\nInterrumpido por usuario")
                self.crear_ground_truth_automatico()
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    print("\n INICIANDO SISTEMA INTERACTIVO REAL")
    print("   Versión: 2.0 - Para obtención de datos REALES")
    
    try:
        sistema = SistemaInteractivoReal()
        if sistema.canonical_products:
            sistema.ejecutar()
        else:
            print("\n Sistema no cargado. Ejecuta primero:")
            print("   python main.py init")
    except Exception as e:
        print(f"\nError crítico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()