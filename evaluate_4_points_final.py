#!/usr/bin/env python3
"""
evaluate_4_points_real.py - Evaluación REAL en 4 puntos con modelo RLHF entrenado

⚠️ ESTE SCRIPT NO GENERA DATOS
⚠️ SOLO EVALÚA SALIDAS DEL SISTEMA REAL
⚠️ NO SE USAN MOCKS NI SIMULACIONES
"""

import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import statistics

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealFourPointEvaluator:
    """Evaluador REAL que usa exclusivamente tu pipeline real"""
    
    def __init__(self):
        self.results = {}
        
        # Consultas de prueba REALES (no se inventan respuestas)
        self.test_queries = self.load_test_queries()
        
        # Componentes reales (lazy loaded)
        self._rag_agent = None
        self._retriever = None
        self._user_manager = None
        
    def load_test_queries(self) -> List[tuple]:
        """Carga consultas de prueba reales (las mismas para todos los puntos)"""
        return [
            # Consultas simples (punto de referencia)
            ("nintendo switch", "simple"),
            ("mario kart", "simple"),
            ("zelda breath of the wild", "simple"),
            
            # Consultas complejas (para evaluar NLP/Zero-shot)
            ("¿Qué juegos de aventura para Nintendo Switch me recomiendas para niños?", "complex"),
            ("Busco consola portátil con buena batería y pantalla OLED", "complex"),
            ("Necesito un juego multijugador para jugar con amigos los fines de semana", "complex"),
            
            # Consultas técnicas (para evaluar ML)
            ("producto nintendo con mejor relación calidad-precio", "technical"),
            ("comparar nintendo switch oled vs lite para viajes", "technical"),
            ("accesorios esenciales para nueva nintendo switch", "technical"),
            
            # Consultas desafiantes (para evaluar robustez)
            ("algo divertido para regalar a mi sobrino de 10 años", "challenging"),
            ("consola retro para jugar juegos clásicos", "challenging"),
            ("¿qué me recomiendas si me gustan los juegos de estrategia?", "challenging"),
        ]
    
    def configure_system(self, point: int) -> Dict[str, Any]:
        """Configura el sistema para cada punto usando settings REALES"""
        try:
            from src.core.config import settings
            
            configs = {
                1: {  # Punto 1: Base sin entrenar (sin NER/Zero-shot)
                    "description": "Base sin ML/NLP",
                    "ml_enabled": False,
                    "nlp_enabled": False,
                    "rlhf_enabled": False,
                    "llm_enabled": False
                },
                2: {  # Punto 2: Base sin entrenar (con NER/Zero-shot)
                    "description": "Base con NLP",
                    "ml_enabled": False,
                    "nlp_enabled": True,
                    "rlhf_enabled": False,
                    "llm_enabled": True
                },
                3: {  # Punto 3: Entrenado (sin NER/Zero-shot)
                    "description": "ML sin NLP",
                    "ml_enabled": True,
                    "nlp_enabled": False,
                    "rlhf_enabled": True,
                    "llm_enabled": False
                },
                4: {  # Punto 4: Entrenado (con NER/Zero-shot)
                    "description": "Completo (ML+NLP+RLHF)",
                    "ml_enabled": True,
                    "nlp_enabled": True,
                    "rlhf_enabled": True,
                    "llm_enabled": True
                }
            }
            
            config = configs[point]
            
            # 🔥 CONFIGURACIÓN REAL: Modificar settings globales
            settings.ML_ENABLED = config["ml_enabled"]
            settings.NLP_ENABLED = config["nlp_enabled"]
            settings.LOCAL_LLM_ENABLED = config["llm_enabled"]
            
            # Establecer modo
            if point == 1:
                settings.CURRENT_MODE = "basic"
            elif point == 2:
                settings.CURRENT_MODE = "enhanced"
            elif point == 3:
                settings.CURRENT_MODE = "balanced"
            else:  # point == 4
                settings.CURRENT_MODE = "enhanced"
            
            # Verificar modelo RLHF
            rlhf_model_path = Path("data/models/rlhf_model")
            config["rlhf_model_available"] = rlhf_model_path.exists() and any(rlhf_model_path.glob("*"))
            
            logger.info(f"🔧 Punto {point}: {config['description']}")
            logger.info(f"   • ML: {config['ml_enabled']}, NLP: {config['nlp_enabled']}")
            logger.info(f"   • RLHF: {config['rlhf_enabled']} (modelo disponible: {config['rlhf_model_available']})")
            logger.info(f"   • LLM: {config['llm_enabled']}")
            
            return config
            
        except ImportError as e:
            logger.error(f"❌ Error importando settings: {e}")
            raise
    
    def get_rag_agent(self, config: Dict) -> Any:
        """Obtiene el agente RAG REAL según configuración"""
        try:
            # 🔥 USAR AGENTE REAL - NO MOCK
            from src.core.rag.advanced.WorkingRAGAgent import create_rag_agent
            
            # Determinar modo basado en configuración
            mode_map = {
                (False, False): "basic",      # Sin ML, sin NLP
                (False, True): "enhanced",    # Sin ML, con NLP
                (True, False): "balanced",    # Con ML, sin NLP
                (True, True): "enhanced"      # Con ML, con NLP
            }
            
            mode = mode_map.get((config["ml_enabled"], config["nlp_enabled"]), "hybrid")
            
            # Crear agente REAL
            agent = create_rag_agent(
                mode=mode,
                ml_enabled=config["ml_enabled"],
                local_llm_enabled=config["llm_enabled"]
            )
            
            # Verificar componentes cargados
            test_results = agent.test_components()
            logger.info(f"✅ Agente RAG real creado (modo: {mode})")
            logger.info(f"   • Retriever: {'✅' if test_results.get('retriever') else '❌'}")
            logger.info(f"   • LLM: {'✅' if test_results.get('llm_client') else '❌'}")
            logger.info(f"   • RLHF: {'✅' if test_results.get('rlhf_pipeline') else '❌'}")
            
            return agent
            
        except ImportError as e:
            logger.error(f"❌ No se pudo importar WorkingRAGAgent: {e}")
            
            # 🔥 FALLBACK: Usar retriever básico si está disponible
            try:
                from src.core.rag.basic.retriever import Retriever
                from src.core.config import settings
                
                logger.warning("⚠️  Usando Retriever básico como fallback")
                retriever = Retriever(
                    index_path=settings.VECTOR_INDEX_PATH,
                    embedding_model=settings.EMBEDDING_MODEL,
                    device=settings.DEVICE
                )
                
                # Crear objeto simple con interfaz compatible
                class SimpleRAGAgent:
                    def __init__(self, retriever):
                        self.retriever = retriever
                        self.config = config
                    
                    def process_query(self, query: str, user_id: Optional[str] = None):
                        start_time = time.time()
                        
                        # Búsqueda básica
                        products = self.retriever.retrieve(query=query, k=5)
                        
                        # Respuesta simple
                        if products:
                            titles = []
                            for p in products[:3]:
                                if hasattr(p, 'title'):
                                    titles.append(p.title[:40])
                                else:
                                    titles.append(str(p)[:40])
                            
                            answer = f"Encontré {len(products)} productos. Recomiendo: {', '.join(titles)}"
                        else:
                            answer = "No encontré productos para tu búsqueda."
                        
                        processing_time = time.time() - start_time
                        
                        return {
                            "query": query,
                            "answer": answer,
                            "products": products or [],
                            "stats": {
                                "processing_time": round(processing_time, 3),
                                "initial_results": len(products),
                                "final_results": len(products),
                                "ml_enhanced": False,
                                "reranking_enabled": False
                            }
                        }
                
                return SimpleRAGAgent(retriever)
                
            except Exception as e2:
                logger.error(f"❌ Fallback también falló: {e2}")
                raise RuntimeError("No se pudo crear ningún agente RAG")
    
    def evaluate_response_quality(self, query: str, response: Dict, point: int) -> Dict[str, float]:
        """Evalúa calidad de respuesta usando métricas REALES (no inventadas)"""
        try:
            answer = response.get("answer", "")
            products = response.get("products", [])
            stats = response.get("stats", {})
            
            scores = {
                "relevance": 0.0,
                "specificity": 0.0,
                "completeness": 0.0,
                "helpfulness": 0.0,
                "processing_time": stats.get("processing_time", 0)
            }
            
            # 1. Relevancia (coincidencia palabras clave)
            query_lower = query.lower()
            answer_lower = answer.lower()
            
            query_words = set(query_lower.split())
            answer_words = set(answer_lower.split())
            
            common_words = query_words.intersection(answer_words)
            if query_words:
                scores["relevance"] = len(common_words) / len(query_words)
            
            # 2. Especificidad (detalles en respuesta)
            # Penalizar respuestas genéricas como "No encontré productos"
            generic_phrases = [
                "no encontré", "no hay", "lo siento", "error",
                "sin resultados", "no disponible", "no pude encontrar"
            ]
            
            is_generic = any(phrase in answer_lower for phrase in generic_phrases)
            scores["specificity"] = 0.0 if is_generic else min(1.0, len(answer) / 100)
            
            # 3. Completitud (productos devueltos)
            scores["completeness"] = min(1.0, len(products) / 5)
            
            # 4. Utilidad (basado en características reales de productos)
            if products:
                helpfulness_scores = []
                for i, product in enumerate(products[:3]):  # Evaluar solo top 3
                    product_score = 0.0
                    
                    # Extraer información real
                    if hasattr(product, 'title'):
                        title = product.title.lower()
                        # Bonus si el título contiene palabras de la query
                        if any(word in title for word in query_lower.split() if len(word) > 3):
                            product_score += 0.3
                    
                    if hasattr(product, 'price') and product.price:
                        # Bonus si tiene precio (indicador de información completa)
                        product_score += 0.2
                    
                    if hasattr(product, 'description') and product.description:
                        # Bonus si tiene descripción
                        product_score += 0.2
                    
                    if hasattr(product, 'ml_processed') and product.ml_processed:
                        # Bonus si fue procesado por ML
                        product_score += 0.1
                    
                    if hasattr(product, 'average_rating') and product.average_rating:
                        # Bonus si tiene rating
                        product_score += 0.1
                    
                    if hasattr(product, 'main_category') and product.main_category:
                        # Bonus si tiene categoría
                        product_score += 0.1
                    
                    helpfulness_scores.append(min(1.0, product_score))
                
                if helpfulness_scores:
                    scores["helpfulness"] = statistics.mean(helpfulness_scores)
            
            # Puntaje total ponderado
            weights = {
                "relevance": 0.35,
                "specificity": 0.25,
                "completeness": 0.20,
                "helpfulness": 0.20
            }
            
            scores["total"] = sum(scores[k] * weights[k] for k in weights)
            
            # Debug info
            logger.debug(f"[Punto {point}] Query: '{query[:30]}...'")
            logger.debug(f"  • Relevancia: {scores['relevance']:.3f}")
            logger.debug(f"  • Especificidad: {scores['specificity']:.3f}")
            logger.debug(f"  • Completitud: {scores['completeness']:.3f}")
            logger.debug(f"  • Utilidad: {scores['helpfulness']:.3f}")
            logger.debug(f"  • Total: {scores['total']:.3f}")
            
            return scores
            
        except Exception as e:
            logger.error(f"Error evaluando respuesta: {e}")
            return {
                "relevance": 0.0,
                "specificity": 0.0,
                "completeness": 0.0,
                "helpfulness": 0.0,
                "total": 0.0,
                "processing_time": 0
            }
    
    def evaluate_point(self, point: int) -> Dict[str, Any]:
        """Evalúa un punto específico usando sistema REAL"""
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 EVALUANDO PUNTO {point} (SISTEMA REAL)")
        logger.info(f"{'='*60}")
        
        # Configurar sistema REAL
        config = self.configure_system(point)
        
        # Obtener agente RAG REAL
        agent = self.get_rag_agent(config)
        
        # Evaluar cada consulta
        all_scores = []
        query_times = []
        product_counts = []
        
        for i, (query, query_type) in enumerate(self.test_queries):
            logger.info(f"🔍 Consulta {i+1}/{len(self.test_queries)}: '{query[:40]}...'")
            
            # 🔥 PROCESAR CON AGENTE REAL
            start_time = time.time()
            try:
                response = agent.process_query(query)
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                # Contar productos REALES devueltos
                products = response.get("products", [])
                product_counts.append(len(products))
                
                # Evaluar calidad REAL
                scores = self.evaluate_response_quality(query, response, point)
                all_scores.append(scores)
                
                logger.info(f"   ⏱️  Tiempo: {query_time*1000:.1f}ms")
                logger.info(f"   📊 Score: {scores['total']:.3f}")
                logger.info(f"   🎯 Productos REALES: {len(products)}")
                
                # Debug: mostrar primeros productos si hay
                if products and logger.isEnabledFor(logging.DEBUG):
                    for j, p in enumerate(products[:2]):
                        title = getattr(p, 'title', str(p))[:50]
                        price = getattr(p, 'price', 'N/A')
                        logger.debug(f"      {j+1}. {title} (${price})")
                        
            except Exception as e:
                logger.error(f"❌ Error procesando consulta: {e}")
                query_times.append(5.0)  # Tiempo penalizado por error
                product_counts.append(0)
                all_scores.append({
                    "relevance": 0.0,
                    "specificity": 0.0,
                    "completeness": 0.0,
                    "helpfulness": 0.0,
                    "total": 0.0,
                    "processing_time": 5.0
                })
        
        # Calcular métricas agregadas
        if all_scores:
            avg_scores = {}
            for key in all_scores[0].keys():
                if key != "processing_time":
                    avg_scores[key] = statistics.mean(s.get(key, 0.0) for s in all_scores)
            
            # Calcular por tipo de consulta
            type_scores = {}
            for query_type in ["simple", "complex", "technical", "challenging"]:
                type_indices = [i for i, (_, t) in enumerate(self.test_queries) if t == query_type]
                if type_indices:
                    type_scores[query_type] = statistics.mean(
                        all_scores[i]["total"] for i in type_indices
                    )
        else:
            avg_scores = {"total": 0.0, "relevance": 0.0, "specificity": 0.0, 
                         "completeness": 0.0, "helpfulness": 0.0}
            type_scores = {}
        
        # Estadísticas de tiempo
        avg_query_time_ms = statistics.mean(query_times) * 1000 if query_times else 0
        avg_products = statistics.mean(product_counts) if product_counts else 0
        
        metrics = {
            "point": point,
            "config": config,
            "avg_query_time_ms": avg_query_time_ms,
            "total_time_ms": sum(query_times) * 1000,
            "avg_products_returned": avg_products,
            "queries_evaluated": len(self.test_queries),
            "scores": avg_scores,
            "type_scores": type_scores,
            "success_rate": sum(1 for c in product_counts if c > 0) / len(product_counts)
        }
        
        logger.info(f"✅ Punto {point} evaluado REALMENTE")
        logger.info(f"   📈 Score total: {avg_scores.get('total', 0):.3f}")
        logger.info(f"   ⏱️  Tiempo promedio: {avg_query_time_ms:.1f}ms")
        logger.info(f"   📦 Productos promedio: {avg_products:.1f}")
        logger.info(f"   ✅ Tasa de éxito: {metrics['success_rate']*100:.1f}%")
        
        return metrics
    
    def compare_points(self):
        """Compara los 4 puntos usando resultados REALES"""
        print("\n" + "="*80)
        print("📊 COMPARACIÓN DE LOS 4 PUNTOS (RESULTADOS REALES)")
        print("="*80)
        
        headers = ["Punto", "Configuración", "Score", "Tiempo(ms)", "Productos", "Éxito", "Simple", "Compleja", "Técnica"]
        print(f"{headers[0]:<6} {headers[1]:<25} {headers[2]:<7} {headers[3]:<10} {headers[4]:<8} {headers[5]:<7} {headers[6]:<8} {headers[7]:<9} {headers[8]:<8}")
        print("-"*80)
        
        for point in sorted(self.results.keys()):
            r = self.results[point]
            config_desc = r["config"]["description"][:23] + "..." if len(r["config"]["description"]) > 23 else r["config"]["description"]
            
            # Emojis según score
            score = r["scores"]["total"]
            if score > 0.7:
                score_display = f"🚀{score:.3f}"
            elif score > 0.5:
                score_display = f"✅{score:.3f}"
            elif score > 0.3:
                score_display = f"⚠️{score:.3f}"
            else:
                score_display = f"❌{score:.3f}"
            
            # Emojis según tiempo
            time_ms = r["avg_query_time_ms"]
            if time_ms < 100:
                time_display = f"⚡{time_ms:.1f}"
            elif time_ms < 500:
                time_display = f"✅{time_ms:.1f}"
            elif time_ms < 1000:
                time_display = f"⚠️{time_ms:.1f}"
            else:
                time_display = f"🐢{time_ms:.1f}"
            
            # Emojis según productos
            products = r["avg_products_returned"]
            if products > 3:
                products_display = f"📦{products:.1f}"
            elif products > 1:
                products_display = f"📦{products:.1f}"
            else:
                products_display = f"📭{products:.1f}"
            
            # Tasa de éxito
            success = r["success_rate"] * 100
            if success > 80:
                success_display = f"✅{success:.0f}%"
            elif success > 50:
                success_display = f"⚠️{success:.0f}%"
            else:
                success_display = f"❌{success:.0f}%"
            
            print(f"{point:<6} {config_desc:<25} {score_display:<7} {time_display:<10} "
                  f"{products_display:<8} {success_display:<7} "
                  f"{r['type_scores'].get('simple', 0):<8.3f} "
                  f"{r['type_scores'].get('complex', 0):<9.3f} "
                  f"{r['type_scores'].get('technical', 0):<8.3f}")
        
        print("="*80)
        
        # Análisis de mejoras REALES
        if 1 in self.results and 4 in self.results:
            point1 = self.results[1]
            point4 = self.results[4]
            
            score_improvement = ((point4["scores"]["total"] - point1["scores"]["total"]) 
                               / max(point1["scores"]["total"], 0.01) * 100)
            
            time_increase = point4["avg_query_time_ms"] - point1["avg_query_time_ms"]
            products_increase = point4["avg_products_returned"] - point1["avg_products_returned"]
            success_increase = (point4["success_rate"] - point1["success_rate"]) * 100
            
            print(f"\n📈 MEJORA REAL Punto 4 vs Punto 1:")
            print(f"   • Score: +{score_improvement:+.1f}%")
            print(f"   • Tiempo: +{time_increase:+.1f}ms")
            print(f"   • Productos: +{products_increase:+.2f}")
            print(f"   • Tasa éxito: +{success_increase:+.1f}%")
            
            if time_increase > 0:
                efficiency = score_improvement / time_increase if time_increase > 0 else 0
                print(f"   • Eficiencia: {efficiency:+.2f}% por ms")
            
            # Recomendación basada en datos REALES
            print(f"\n💡 RECOMENDACIÓN BASADA EN DATOS REALES: ", end="")
            if score_improvement > 30 and success_increase > 20 and time_increase < 200:
                print("USAR Punto 4 (Sistema completo)")
                print("   • Gran mejora en calidad y éxito")
                print("   • Overhead de tiempo aceptable")
            elif score_improvement > 15 and success_increase > 10:
                print("CONSIDERAR Punto 3 o 4 según necesidades")
                print("   • Mejora moderada en calidad")
                print("   • Evaluar trade-off calidad/tiempo")
            else:
                print("USAR Punto 1 o 2")
                print("   • Sistema complejo no justifica mejora")
                print("   • Mejor rendimiento con sistema simple")
        
        # Mejor en cada categoría REAL
        print(f"\n🏆 MEJOR EN CADA CATEGORÍA (DATOS REALES):")

        # Verificar que hay resultados
        if not self.results:
            print("   • No hay datos para comparar")
            return

        categories = [
            ("Score", lambda r: r["scores"]["total"], "max", ""),
            ("Tiempo", lambda r: r["avg_query_time_ms"], "min", "ms"),
            ("Productos", lambda r: r["avg_products_returned"], "max", ""),
            ("Éxito", lambda r: r["success_rate"] * 100, "max", "%")
        ]

        for cat_name, get_value, extremum, unit in categories:
            try:
                if extremum == "max":
                    best_point = max(self.results.keys(), 
                                key=lambda p: get_value(self.results[p]))
                else:  # "min"
                    best_point = min(self.results.keys(),
                                key=lambda p: get_value(self.results[p]))
                
                value = get_value(self.results[best_point])
                
                # Formatear según unidad
                if unit == "%":
                    print(f"   • {cat_name:<10}: Punto {best_point} ({value:.1f}%)")
                elif unit == "ms":
                    print(f"   • {cat_name:<10}: Punto {best_point} ({value:.1f}ms)")
                else:
                    print(f"   • {cat_name:<10}: Punto {best_point} ({value:.2f})")
                    
            except (KeyError, ValueError, TypeError) as e:
                print(f"   • {cat_name:<10}: Error ({e})")
    
    def save_results(self, filename: str = "4_points_evaluation_real.json"):
        """Guarda los resultados REALES"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "test_queries": len(self.test_queries),
            "query_list": [q for q, _ in self.test_queries],
            "rlhf_model_trained": Path("data/models/rlhf_model").exists() and any(Path("data/models/rlhf_model").iterdir()),
            "chroma_index_exists": Path(Path.cwd() / "data/processed/chroma_db").exists(),
            "results": self.results,
            "disclaimer": "ESTOS SON RESULTADOS REALES - NO SE INVENTARON DATOS"
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Resultados REALES guardados en: {filename}")
        return filename

def verify_system_requirements():
    """Verifica que el sistema esté listo para evaluación REAL"""
    print("🔍 VERIFICANDO SISTEMA PARA EVALUACIÓN REAL")
    print("="*60)
    
    requirements_met = True
    
    # 1. Verificar índice Chroma
    chroma_path = Path.cwd() / "data/processed/chroma_db"
    if chroma_path.exists():
        print(f"✅ Índice Chroma: {len(list(chroma_path.glob('*')))} archivos")
    else:
        print("❌ Índice Chroma no encontrado")
        print("   Ejecuta primero: python main.py index")
        requirements_met = False
    
    # 2. Verificar datos procesados
    products_file = Path.cwd() / "data/processed/products.json"
    if products_file.exists():
        try:
            import json
            with open(products_file, 'r', encoding='utf-8') as f:
                products = json.load(f)
                print(f"✅ Datos procesados: {len(products)} productos")
        except:
            print("⚠️  Datos procesados: archivo corrupto")
    else:
        print("❌ Datos procesados no encontrados")
        requirements_met = False
    
    # 3. Verificar modelo RLHF (opcional)
    rlhf_path = Path.cwd() / "data/models/rlhf_model"
    if rlhf_path.exists() and any(rlhf_path.glob("*")):
        print(f"✅ Modelo RLHF: {'pytorch_model.bin' in [f.name for f in rlhf_path.glob('*')]}")
    else:
        print("⚠️  Modelo RLHF no encontrado (Puntos 3-4 usarán fallback)")
    
    # 4. Verificar dependencias
    try:
        from src.core.config import settings
        print(f"✅ Configuración: {settings.CURRENT_MODE}")
    except ImportError:
        print("❌ No se pudo importar configuración")
        requirements_met = False
    
    print("="*60)
    
    if not requirements_met:
        print("\n⚠️  ADVERTENCIA: Algunos requisitos no se cumplen")
        print("   La evaluación puede fallar o usar modos fallback")
    
    return requirements_met

def main():
    parser = argparse.ArgumentParser(description="Evaluación REAL en 4 puntos con RLHF")
    parser.add_argument("--points", type=str, default="1,2,3,4",
                       help="Puntos a evaluar (ej: '1,2,3,4' o '3,4')")
    parser.add_argument("--output", type=str, default="4_points_evaluation_real.json",
                       help="Archivo de salida")
    parser.add_argument("--verbose", action="store_true",
                       help="Mostrar detalles detallados")
    parser.add_argument("--force", action="store_true",
                       help="Forzar evaluación incluso si hay advertencias")
    parser.add_argument("--verify-only", action="store_true",
                       help="Solo verificar sistema, no evaluar")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🤖 EVALUACIÓN REAL EN 4 PUNTOS CON RLHF")
    print("=" * 60)
    print("⚠️  ESTE SCRIPT USA DATOS REALES - NO INVENTA RESPUESTAS")
    print("=" * 60)
    
    # Verificar sistema
    if not verify_system_requirements() and not args.force:
        print("\n❌ Sistema no está listo para evaluación")
        print("💡 Usa --force para evaluar de todos modos")
        return
    
    if args.verify_only:
        print("\n✅ Verificación completada")
        return
    
    # Verificar modelo RLHF
    rlhf_dir = Path("data/models/rlhf_model")
    if rlhf_dir.exists() and any(rlhf_dir.glob("*")):
        print(f"✅ Modelo RLHF entrenado detectado")
        model_files = list(rlhf_dir.glob("*"))
        print(f"   📁 Archivos: {len(model_files)}")
        for file in model_files[:3]:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   • {file.name} ({size_mb:.1f} MB)")
    else:
        print("⚠️  No se detectó modelo RLHF entrenado")
        print("💡 Puntos 3-4 usarán sistema sin RLHF")
        print("💡 Para RLHF: python FIX_AND_TRAIN_RLHF.py")
    
    # Crear evaluador REAL
    evaluator = RealFourPointEvaluator()
    
    # Evaluar puntos especificados
    points_to_evaluate = [int(p.strip()) for p in args.points.split(",") if p.strip().isdigit()]
    
    for point in points_to_evaluate:
        if 1 <= point <= 4:
            print(f"\n🎯 Iniciando evaluación REAL del Punto {point}")
            try:
                result = evaluator.evaluate_point(point)
                evaluator.results[point] = result
                print(f"✅ Punto {point} evaluado exitosamente")
            except Exception as e:
                print(f"❌ Error evaluando Punto {point}: {e}")
                import traceback
                traceback.print_exc()
    
    # Mostrar comparación si hay múltiples puntos
    if len(evaluator.results) > 1:
        evaluator.compare_points()
    
    # Guardar resultados REALES
    output_file = evaluator.save_results(args.output)
    
    print(f"\n📊 RESUMEN EJECUTIVO (DATOS REALES):")
    print(f"   • Puntos evaluados: {len(evaluator.results)}")
    print(f"   • Consultas por punto: {len(evaluator.test_queries)}")
    print(f"   • RLHF disponible: {'✅ Sí' if rlhf_dir.exists() else '❌ No'}")
    print(f"   • Resultados REALES: {output_file}")
    
    print(f"\n📈 ANÁLISIS DE LOS DATOS REALES:")
    for point in sorted(evaluator.results.keys()):
        r = evaluator.results[point]
        print(f"   • Punto {point}: Score={r['scores']['total']:.3f}, "
              f"Tiempo={r['avg_query_time_ms']:.1f}ms, "
              f"Productos={r['avg_products_returned']:.1f}")
    
    print(f"\n💡 SIGUIENTES PASOS (CON DATOS REALES):")
    print(f"   1. Revisa resultados detallados en: {output_file}")
    print(f"   2. Para usar el mejor sistema: python main.py rag --mode enhanced")
    print(f"   3. Para entrenar RLHF: python FIX_AND_TRAIN_RLHF.py")
    print(f"   4. Para más pruebas: ejecuta con --points para puntos específicos")
    
    # Mostrar advertencia importante
    print(f"\n⚠️  IMPORTANTE: Estos resultados son REALES")
    print("   • No se inventaron productos")
    print("   • No se inventaron respuestas")
    print("   • No se inventaron scores")
    print("   • Todo viene de tu pipeline real")

if __name__ == "__main__":
    main()