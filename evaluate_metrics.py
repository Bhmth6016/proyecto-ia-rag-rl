#!/usr/bin/env python3
"""
evaluate_metrics.py (CORREGIDO)

Evalúa las 4 configuraciones:
1. Solo RAG
2. RAG + RL
3. RAG + Usuarios
4. RAG + RL + Usuarios

Mejoras:
- Crea un solo usuario de prueba por configuración cuando corresponde.
- No crea usuarios por cada query.
- Manejo robusto de respuestas faltantes.
- No imprime el reporte múltiples veces.
"""

import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import statistics

from src.core.rag.advanced.WorkingRAGAgent import WorkingAdvancedRAGAgent, RAGConfig
from src.core.data.user_manager import UserManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemEvaluator:
    def __init__(self):
        self.test_queries = [
            "juegos de nintendo switch para niños",
            "mejores juegos de acción para playstation",
            "rpg con buena historia",
            "juegos de deportes baratos",
            "fps multijugador online",
            "aventura en mundo abierto",
            "juegos de estrategia para pc",
            "simulador de conducción realista",
            "juegos de puzzle difíciles",
            "novedades gaming 2024"
        ]
        self.metrics_results = {}

    def evaluate_configuration(self, config_name: str, config: RAGConfig, use_users: bool = False):
        """Evalúa una configuración específica del sistema"""
        logger.info(f"🧪 Evaluando configuración: {config_name} (use_users={use_users})")

        metrics = {
            "response_times": [],
            "quality_scores": [],
            "products_returned": [],
            "successful_queries": 0,
            "total_queries": 0
        }

        # Inicializar UserManager sólo si lo necesitamos
        user_manager: Optional[UserManager] = UserManager() if use_users else None

        # Crear 1 usuario de prueba cuando use_users=True (evita creación por cada query)
        test_user = None
        if user_manager:
            # Intentar crear o reutilizar un usuario de prueba único
            try:
                test_user = user_manager.create_user_profile(age=25, gender="male", country="Spain")
                logger.info(f"👤 Usuario de prueba creado: {test_user.user_id}")
            except Exception as e:
                logger.warning(f"No se pudo crear user de prueba: {e}. Continuando con user_id='default'.")
                test_user = None

        try:
            agent = WorkingAdvancedRAGAgent(config=config)

            for query in self.test_queries:
                start_time = time.time()

                try:
                    # Usar el usuario de prueba creado (si existe) o 'default'
                    if test_user:
                        user_id = test_user.user_id
                    else:
                        user_id = "default"

                    response = agent.process_query(query, user_id)
                    end_time = time.time()

                    # Manejo robusto si response es None o faltan atributos
                    if response is None:
                        logger.error(f"  ❌ La respuesta para '{query}' es None")
                        metrics["total_queries"] += 1
                        continue

                    response_time = end_time - start_time
                    quality_score = getattr(response, "quality_score", 0.0) or 0.0
                    products = getattr(response, "products", []) or []
                    products_count = len(products)

                    metrics["response_times"].append(response_time)
                    metrics["quality_scores"].append(quality_score)
                    metrics["products_returned"].append(products_count)
                    metrics["total_queries"] += 1

                    if quality_score >= 0.7:
                        metrics["successful_queries"] += 1

                    logger.info(f"  ✅ '{query[:30]}...' -> {quality_score:.2f} | {products_count} productos | {response_time:.2f}s")

                except Exception as e:
                    logger.error(f"  ❌ Error en query '{query}': {e}")
                    metrics["total_queries"] += 1

            # Calcular métricas agregadas (proteger contra listas vacías)
            if metrics["response_times"]:
                metrics["avg_response_time"] = statistics.mean(metrics["response_times"])
            else:
                metrics["avg_response_time"] = 0.0

            if metrics["quality_scores"]:
                metrics["avg_quality_score"] = statistics.mean(metrics["quality_scores"])
            else:
                metrics["avg_quality_score"] = 0.0

            if metrics["products_returned"]:
                metrics["avg_products_returned"] = statistics.mean(metrics["products_returned"])
            else:
                metrics["avg_products_returned"] = 0.0

            metrics["success_rate"] = (metrics["successful_queries"] / metrics["total_queries"]) if metrics["total_queries"] > 0 else 0.0

            self.metrics_results[config_name] = metrics

            logger.info(f"📊 Resultados {config_name}:")
            logger.info(f"   Calidad promedio: {metrics['avg_quality_score']:.3f}")
            logger.info(f"   Tiempo respuesta: {metrics['avg_response_time']:.3f}s")
            logger.info(f"   Productos promedio: {metrics['avg_products_returned']:.1f}")
            logger.info(f"   Tasa de éxito: {metrics['success_rate']:.1%}")

        except Exception as e:
            logger.error(f"❌ Error evaluando {config_name}: {e}")
            self.metrics_results[config_name] = {"error": str(e)}

    def evaluate_all_configurations(self):
        """Evalúa las 4 configuraciones del sistema"""
        logger.info("🎯 INICIANDO EVALUACIÓN COMPARATIVA DEL SISTEMA")
        logger.info("=" * 70)

        # Configuración 1: Solo RAG (sin características avanzadas ni usuarios)
        config_rag_only = RAGConfig(
            enable_reranking=False,
            enable_rlhf=False,
            max_retrieved=10,
            max_final=3,
            use_advanced_features=False
        )
        self.evaluate_configuration("Solo RAG", config_rag_only, use_users=False)

        # Configuración 2: RAG + RL
        config_rag_rl = RAGConfig(
            enable_reranking=True,
            enable_rlhf=True,
            max_retrieved=15,
            max_final=5,
            use_advanced_features=True
        )
        self.evaluate_configuration("RAG + RL", config_rag_rl, use_users=False)

        # Configuración 3: RAG + Usuarios
        config_rag_users = RAGConfig(
            enable_reranking=True,
            enable_rlhf=False,
            max_retrieved=15,
            max_final=5,
            use_advanced_features=True
        )
        self.evaluate_configuration("RAG + Usuarios", config_rag_users, use_users=True)

        # Configuración 4: RAG + RL + Usuarios (Completo)
        config_full = RAGConfig(
            enable_reranking=True,
            enable_rlhf=True,
            max_retrieved=20,
            max_final=5,
            use_advanced_features=True
        )
        self.evaluate_configuration("RAG + RL + Usuarios", config_full, use_users=True)

        self._generate_comparison_report()

    def _generate_comparison_report(self):
        """Genera reporte comparativo de todas las configuraciones"""
        logger.info("\n" + "=" * 70)
        logger.info("📈 REPORTE COMPARATIVO FINAL")
        logger.info("=" * 70)

        # Encabezado de la tabla
        print(f"\n{'CONFIGURACIÓN':<25} {'CALIDAD':<8} {'TIEMPO(s)':<10} {'PRODUCTOS':<10} {'ÉXITO':<8}")
        print("-" * 70)

        for config_name, metrics in self.metrics_results.items():
            if "error" in metrics:
                print(f"{config_name:<25} {'ERROR':<8} {'-':<10} {'-':<10} {'-':<8}")
            else:
                print(f"{config_name:<25} {metrics['avg_quality_score']:.3f}   {metrics['avg_response_time']:.3f}     {metrics['avg_products_returned']:<8.1f}   {metrics['success_rate']:.1%}")

        # Análisis de mejoras
        if all("error" not in m for m in self.metrics_results.values()):
            self._calculate_improvements()

    def _calculate_improvements(self):
        """Calcula mejoras entre configuraciones"""
        base = self.metrics_results.get("Solo RAG", {})
        base_quality = base.get("avg_quality_score", 0.0)
        base_success = base.get("success_rate", 0.0)

        improvements = {}
        for config in ["RAG + RL", "RAG + Usuarios", "RAG + RL + Usuarios"]:
            m = self.metrics_results.get(config, {})
            improvements[config] = {
                "quality_improvement": m.get("avg_quality_score", 0.0) - base_quality,
                "success_improvement": m.get("success_rate", 0.0) - base_success
            }

        print(f"\n📊 MEJORAS RELATIVAS vs SOLO RAG:")
        print("-" * 50)
        for config, imp in improvements.items():
            print(f"{config:<25} +{imp['quality_improvement']:.3f} calidad   +{imp['success_improvement']:.1%} éxito")

        # Guardar resultados en JSON
        results_file = Path("evaluation_results.json")
        output_data = {
            "evaluation_date": datetime.now().isoformat(),
            "test_queries_count": len(self.test_queries),
            "configurations": self.metrics_results,
            "improvements": improvements
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 Resultados guardados en: {results_file}")

    def run_evaluation(self):
        """Ejecuta la evaluación completa"""
        try:
            self.evaluate_all_configurations()
        except Exception as e:
            logger.error(f"Error en evaluación: {e}")
            raise

if __name__ == "__main__":
    print("🎯 EVALUADOR DEL SISTEMA HÍBRIDO RAG + RL")
    print("📊 Comparando 4 configuraciones:")
    print("   1. Solo RAG")
    print("   2. RAG + RL")
    print("   3. RAG + Usuarios")
    print("   4. RAG + RL + Usuarios")
    print("=" * 60)

    evaluator = SystemEvaluator()
    evaluator.run_evaluation()

