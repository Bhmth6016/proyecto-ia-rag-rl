#!/usr/bin/env python3
"""
Script para generar datos de EVALUACIÓN POST-ENTRENAMIENTO
MEJORADO con: modularización, CLI, type hints completos y mejor manejo de errores
"""

import json
import random
import logging
import csv
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ==================== CONFIGURACIÓN DE RUTAS ====================
DATA_DIR = Path("data")
RAW_PRODUCTS = DATA_DIR / "raw" / "products.jsonl"
EVALUATION_DIR = DATA_DIR / "evaluation"
TEST_QUERIES = EVALUATION_DIR / "test_queries_dataset.jsonl"
MODEL_COMPARISONS = EVALUATION_DIR / "comparisons" / "model_responses_comparison.jsonl"
HUMAN_EVAL_DATASET = EVALUATION_DIR / "human_judgments" / "human_evaluation_dataset.jsonl"
TIMELINE_METRICS = EVALUATION_DIR / "metrics" / "timeline_metrics.jsonl"
METRICS_SUMMARY = EVALUATION_DIR / "metrics" / "summary.json"
DETAILED_METRICS_CSV = EVALUATION_DIR / "metrics" / "detailed_metrics.csv"
PROGRESS_CHART = EVALUATION_DIR / "metrics" / "progress_chart.png"
HUMAN_EVAL_FORM = EVALUATION_DIR / "human_judgments" / "evaluation_form.html"


# ==================== CONFIGURACIÓN DE EVALUACIÓN ====================
EVALUATION_QUERIES = {
    "relevance": [
        "ratón ergonómico para programar 8 horas al día",
        "teclado mecánico silencioso para oficina",
        "consola para niños de 10 años",
        "auriculares gaming con cancelación de ruido",
        "monitor 4K para trabajo y gaming",
        "silla ergonómica para streamer"
    ],
    "comparison": [
        "¿Qué es mejor, PS5 o Xbox Series X?",
        "Logitech MX Master 3 vs Razer DeathAdder V2",
        "Nintendo Switch OLED vs Switch Lite",
        "teclado mecánico vs membrana para gaming",
        "ratón alámbrico vs inalámbrico para FPS"
    ],
    "specific_requirements": [
        "ratón gaming con más de 10 botones para MMOs",
        "teclado compacto 60% para viajar",
        "auriculares con micrófono desmontable",
        "monitor con G-Sync y 240Hz",
        "silla gaming para personas altas (+1.90m)"
    ],
    "budget_constrained": [
        "mejor ratón gaming por menos de 50€",
        "teclado mecánico económico pero bueno",
        "auriculares gaming baratos con buena calidad",
        "accesorios gaming calidad-precio",
        "ratón inalámbrico económico para trabajo"
    ]
}

IDEAL_RESPONSES = {
    "ratón ergonómico para programar 8 horas al día": {
        "product": "B08N5WRWNW",  # Logitech MX Master 3
        "key_features": ["ergonómico", "batería larga duración", "scroll horizontal"],
        "justification": "Ideal para largas sesiones por su diseño ergonómico y batería de 70 días"
    },
    "consola para niños de 10 años": {
        "product": "B0D12C7Y5N",  # Nintendo Switch
        "key_features": ["portátil", "juegos familiares", "resistente"],
        "justification": "Perfecta para niños por su portabilidad y catálogo de juegos infantiles"
    },
    "mejor ratón gaming por menos de 50€": {
        "product": "B07S92QBCJ",  # Razer DeathAdder V2 (asumiendo oferta)
        "key_features": ["20K DPI", "diseño ergonómico", "buen precio"],
        "justification": "Ofrece características premium a un precio asequible"
    }
}

METRICS = [
    "relevance", "completeness", "accuracy", 
    "helpfulness", "personalization", "conciseness"
]


# ==================== FUNCIONES UTILITARIAS ====================

def write_jsonl(path: Path, data: List[Dict], append: bool = False) -> None:
    """Escribe datos en formato JSONL"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        
        with path.open(mode, encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        logger.info(f"✅ {path} - {len(data)} registros escritos")
    except Exception as e:
        logger.error(f"❌ Error al escribir {path}: {e}")
        raise

def read_jsonl(path: Path) -> List[Dict]:
    """Lee datos desde un archivo JSONL"""
    try:
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        logger.error(f"Error al leer {path}: {e}")
        return []


class TestQueryGenerator:
    """Clase para generar consultas de prueba"""
    
    def __init__(self, gaming_products: List[Dict[str, Any]]):
        self.gaming_products = gaming_products
    
    def _find_ideal_product_safe(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Encuentra el producto ideal para una consulta con fallback seguro
        
        Args:
            query: Consulta del usuario
        
        Returns:
            Producto ideal o None si no se encuentra
        """
        query_lower = query.lower()
        
        try:
            # Reglas para matching específico
            if "ratón" in query_lower and "programar" in query_lower:
                return next((p for p in self.gaming_products if "MX Master" in p["title"]), None)
            elif "consola" in query_lower and ("niño" in query_lower or "niña" in query_lower):
                nintendo_products = [p for p in self.gaming_products if "Nintendo" in p["title"]]
                return nintendo_products[0] if nintendo_products else None
            elif "menos de 50" in query_lower or "económico" in query_lower:
                affordable = [p for p in self.gaming_products if p["price"] < 80]
                return random.choice(affordable) if affordable else None
            elif "switch" in query_lower:
                return next((p for p in self.gaming_products if "Nintendo" in p["title"]), None)
            
            # Default: producto con mejor rating
            return max(self.gaming_products, key=lambda x: x["average_rating"]) if self.gaming_products else None
            
        except (StopIteration, ValueError):
            logger.warning(f"No se encontró producto ideal para consulta: {query}")
            return random.choice(self.gaming_products) if self.gaming_products else None
    
    def _create_distraction_responses_improved(self, query: str, ideal_product: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Crea respuestas de distracción mejoradas con variaciones realistas
        
        Args:
            query: Consulta original
            ideal_product: Producto ideal
        
        Returns:
            Lista de respuestas de distracción
        """
        distractions = []
        
        # Tipos de distracción con diferentes niveles de severidad
        distraction_types = [
            {
                "name": "producto_incorrecto",
                "severity": "high",
                "generator": self._generate_wrong_product_response
            },
            {
                "name": "respuesta_incompleta",
                "severity": "medium",
                "generator": self._generate_incomplete_response
            },
            {
                "name": "informacion_incorrecta",
                "severity": "high",
                "generator": self._generate_incorrect_info_response
            },
            {
                "name": "respuesta_ambigua",
                "severity": "low",
                "generator": self._generate_ambiguous_response
            }
        ]
        
        # Generar 2-3 distracciones por consulta
        for dist_type in random.sample(distraction_types, k=random.randint(2, 3)):
            distraction = dist_type["generator"](query, ideal_product)
            distraction.update({
                "issue": dist_type["name"],
                "severity": dist_type["severity"]
            })
            distractions.append(distraction)
        
        return distractions
    
    def _generate_wrong_product_response(self, query: str, ideal_product: Dict[str, Any]) -> Dict[str, Any]:
        """Genera respuesta con producto incorrecto pero relacionado"""
        wrong_products = [p for p in self.gaming_products if p["id"] != ideal_product["id"]]
        wrong_product = random.choice(wrong_products) if wrong_products else ideal_product
        
        return {
            "text": f"Para '{query}', te recomiendo el {wrong_product['title']}. {wrong_product['description'][:150]}...",
            "product_id": wrong_product["id"]
        }
    
    def _generate_incomplete_response(self, query: str, ideal_product: Dict[str, Any]) -> Dict[str, Any]:
        """Genera respuesta incompleta"""
        incomplete_templates = [
            f"El {ideal_product['title']} es una buena opción.",
            f"Te recomiendo el {ideal_product['title']}.",
            f"Podrías considerar el {ideal_product['title']} para tu caso."
        ]
        
        return {
            "text": random.choice(incomplete_templates),
            "product_id": ideal_product["id"]
        }
    
    def _generate_incorrect_info_response(self, query: str, ideal_product: Dict[str, Any]) -> Dict[str, Any]:
        """Genera respuesta con información incorrecta"""
        incorrect_info = [
            f"Tiene una batería de 100 días (incorrecto)",
            f"Cuesta $50 menos del precio real",
            f"No es compatible con tu sistema",
            f"Tiene una calificación de 3 estrellas (incorrecto)"
        ]
        
        return {
            "text": f"Para '{query}', te recomiendo el {ideal_product['title']}. {random.choice(incorrect_info)}.",
            "product_id": ideal_product["id"]
        }
    
    def _generate_ambiguous_response(self, query: str, ideal_product: Dict[str, Any]) -> Dict[str, Any]:
        """Genera respuesta ambigua o vaga"""
        ambiguous_templates = [
            f"Hay varias opciones disponibles para '{query}'.",
            f"Depende de tus preferencias personales.",
            f"Te sugiero que investigues más sobre el tema."
        ]
        
        return {
            "text": random.choice(ambiguous_templates),
            "product_id": ideal_product["id"]
        }
    
    def _get_evaluation_criteria(self, query_type: str) -> List[str]:
        """Obtiene criterios de evaluación según tipo de consulta"""
        criteria_map = {
            "relevance": ["relevance", "accuracy", "completeness"],
            "comparison": ["completeness", "objectivity", "helpfulness"],
            "specific_requirements": ["specificity", "accuracy", "completeness"],
            "budget_constrained": ["price_awareness", "value_proposition", "relevance"]
        }
        return criteria_map.get(query_type, METRICS)
    
    def _assign_difficulty_level(self, query: str, query_type: str) -> str:
        """Asigna nivel de dificultad a la consulta"""
        difficulty_rules = [
            (lambda q: "vs" in q or "compar" in q.lower(), "hard"),
            (lambda q: "menos de" in q or "económico" in q.lower(), "medium"),
            (lambda q: "mejor" in q.lower() and "para" in q.lower(), "medium"),
            (lambda q: True, "easy")  # Default
        ]
        
        for rule, level in difficulty_rules:
            if rule(query):
                return level
        
        return "easy"


class ModelResponseGenerator:
    """Clase para generar respuestas de diferentes versiones del modelo"""
    
    def __init__(self, gaming_products: List[Dict[str, Any]]):
        self.gaming_products = gaming_products
    
    def generate_baseline_response(self, test_query: Dict[str, Any]) -> Dict[str, Any]:
        """Genera respuesta del modelo base (pre-entrenamiento)"""
        generic_phrases = [
            "Hay varios productos disponibles.",
            "Te puedo recomendar algunas opciones.",
            "Basado en tu consulta, aquí hay una sugerencia."
        ]
        
        product = random.choice(self.gaming_products) if self.gaming_products else {}
        
        return {
            "text": f"{random.choice(generic_phrases)} Considera el {product.get('title', 'producto')}.",
            "product_id": product.get("id", ""),
            "confidence": random.uniform(0.6, 0.8),
            "model_version": "baseline_v1.0"
        }
    
    def generate_fine_tuned_response(self, test_query: Dict[str, Any]) -> Dict[str, Any]:
        """Genera respuesta del modelo fine-tuned"""
        query = test_query["query"].lower()
        ideal_product_id = test_query.get("ideal_product_id")
        
        # Intentar encontrar producto ideal
        if ideal_product_id:
            product = next((p for p in self.gaming_products if p["id"] == ideal_product_id), None)
        
        if not product:
            # Buscar por palabras clave
            relevant_products = []
            for product in self.gaming_products:
                title_words = product["title"].lower().split()[:3]
                if any(keyword in query for keyword in title_words):
                    relevant_products.append(product)
            
            product = max(relevant_products, key=lambda x: x["average_rating"]) if relevant_products else random.choice(self.gaming_products)
        
        return {
            "text": f"Para '{test_query['query']}', el {product['title']} es una buena opción. {product['description'][:100]}...",
            "product_id": product["id"],
            "confidence": random.uniform(0.7, 0.9),
            "model_version": "fine_tuned_v2.1"
        }
    
    def generate_rlhf_response(self, test_query: Dict[str, Any]) -> Dict[str, Any]:
        """Genera respuesta del modelo después de RLHF (mejor calidad)"""
        ideal_product_id = test_query.get("ideal_product_id")
        
        if ideal_product_id:
            product = next((p for p in self.gaming_products if p["id"] == ideal_product_id), 
                          random.choice(self.gaming_products) if self.gaming_products else {})
        else:
            # Buscar producto más relevante
            product = self._find_most_relevant_product(test_query["query"])
        
        # Respuesta detallada y personalizada
        response_text = f"""Basado en tu consulta sobre '{test_query['query']}', te recomiendo el {product.get('title', 'producto')}.

📊 **Características principales:**
• {product.get('description', 'Descripción no disponible')}
• Precio: ${product.get('price', 'N/A')}
• Calificación: {product.get('average_rating', 'N/A')}/5

🎯 **Por qué es adecuado:**
Este producto se adapta bien a tus necesidades específicas mencionadas en la consulta."""

        return {
            "text": response_text,
            "product_id": product.get("id", ""),
            "confidence": random.uniform(0.85, 0.98),
            "model_version": "rlhf_v3.5",
            "key_features": product.get("features", [])[:3] if "features" in product else []
        }
    
    def _find_most_relevant_product(self, query: str) -> Dict[str, Any]:
        """Encuentra el producto más relevante para la consulta"""
        if not self.gaming_products:
            return {}
        
        query_lower = query.lower()
        
        # Puntuar productos según relevancia
        scored_products = []
        for product in self.gaming_products:
            score = 0
            
            # Puntos por coincidencia en título
            title_lower = product["title"].lower()
            for word in query_lower.split():
                if word in title_lower and len(word) > 3:
                    score += 2
            
            # Puntos por coincidencia en descripción
            desc_lower = product.get("description", "").lower()
            for word in query_lower.split():
                if word in desc_lower and len(word) > 3:
                    score += 1
            
            # Bonus por rating alto
            score += product.get("average_rating", 0) / 10
            
            scored_products.append((score, product))
        
        # Devolver producto con mayor puntuación
        scored_products.sort(reverse=True, key=lambda x: x[0])
        return scored_products[0][1] if scored_products else self.gaming_products[0]


class MetricsCalculator:
    """Clase para calcular métricas de evaluación"""
    
    @staticmethod
    def calculate_response_metrics(responses: Dict[str, Dict], ideal_response: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Calcula métricas para cada respuesta con distribución más realista
        
        Args:
            responses: Diccionario de respuestas por modelo
            ideal_response: Respuesta ideal
        
        Returns:
            Diccionario de métricas por modelo
        """
        metrics = {}
        ideal_product_id = ideal_response.get("product_id")
        
        for model_name, response in responses.items():
            if model_name == "ideal_response":
                continue
            
            # Distribución normal para métricas más realistas
            base_relevance = max(0.1, min(1.0, random.normalvariate(0.7, 0.15)))
            base_accuracy = max(0.1, min(1.0, random.normalvariate(0.75, 0.12)))
            base_completeness = max(0.1, min(1.0, random.normalvariate(0.65, 0.18)))
            
            # Ajustar según modelo (RLHF mejor, baseline peor)
            if model_name == "rlhf_model":
                multiplier = 1.2
            elif model_name == "fine_tuned_model":
                multiplier = 1.1
            else:  # baseline
                multiplier = 1.0
            
            product_match = 1.0 if response.get("product_id") == ideal_product_id else 0.0
            
            metrics[model_name] = {
                "relevance": min(1.0, base_relevance * multiplier),
                "accuracy": min(1.0, base_accuracy * multiplier),
                "completeness": min(1.0, base_completeness * multiplier),
                "helpfulness": min(1.0, (base_relevance * 0.7 + base_accuracy * 0.3) * multiplier),
                "conciseness": random.uniform(0.7, 0.95),
                "product_match": product_match
            }
        
        return metrics
    
    @staticmethod
    def generate_summary(comparison_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Genera resumen de métricas para análisis rápido"""
        import numpy as np
        
        summary = {
            "total_queries": len(comparison_data),
            "models": ["baseline_model", "fine_tuned_model", "rlhf_model"],
            "metrics_summary": {},
            "performance_by_difficulty": {},
            "performance_by_query_type": {}
        }
        
        # Calcular promedios por modelo
        for model in summary["models"]:
            model_metrics = []
            for entry in comparison_data:
                if model in entry["evaluation_metrics"]:
                    model_metrics.append(entry["evaluation_metrics"][model])
            
            if model_metrics:
                avg_metrics = {}
                for metric in METRICS + ["product_match"]:
                    values = [m.get(metric, 0) for m in model_metrics]
                    if values:
                        avg_metrics[metric] = {
                            "mean": np.mean(values),
                            "std": np.std(values),
                            "min": np.min(values),
                            "max": np.max(values)
                        }
                
                summary["metrics_summary"][model] = avg_metrics
        
        return summary


class PostTrainingEvaluator:
    def __init__(self):
        # Configurar directorios de evaluación
        self.setup_evaluation_directories()
        
        # Cargar o crear productos base
        self.gaming_products = self.load_or_create_products()
        
        # Inicializar componentes
        self.test_generator = TestQueryGenerator(self.gaming_products)
        self.model_generator = ModelResponseGenerator(self.gaming_products)
        self.metrics_calculator = MetricsCalculator()

    def setup_evaluation_directories(self) -> None:
        """Crea estructura de directorios para evaluación"""
        directories = [
            "data/evaluation",
            "data/evaluation/pre_training",
            "data/evaluation/post_training",
            "data/evaluation/human_judgments",
            "data/evaluation/metrics",
            "data/evaluation/comparisons"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def load_or_create_products(self) -> List[Dict[str, Any]]:
        """Carga productos existentes o crea dataset de evaluación"""
        if RAW_PRODUCTS.exists():
            products = read_jsonl(RAW_PRODUCTS)
            logger.info(f"✅ Cargados {len(products)} productos existentes")
            return products
        else:
            # Productos básicos para evaluación
            return [
                {
                    "id": "B08N5WRWNW",
                    "title": "Logitech MX Master 3 - Ratón Inalámbrico",
                    "description": "Ratón ergonómico con scroll horizontal, 4000 DPI, batería 70 días",
                    "main_category": "Periféricos Gaming",
                    "price": 99.99,
                    "average_rating": 4.7
                },
                {
                    "id": "B07S92QBCJ",
                    "title": "Razer DeathAdder V2 - Ratón Gaming",
                    "description": "Ratón gaming con sensor óptico 20K DPI, 8 botones programables",
                    "main_category": "Periféricos Gaming",
                    "price": 69.99,
                    "average_rating": 4.6
                },
                {
                    "id": "B0D12C7Y5N",
                    "title": "Nintendo Switch OLED - Mario Red Edition",
                    "description": "Consola híbrida con pantalla OLED 7 pulgadas, modo portátil y TV",
                    "main_category": "Consolas",
                    "price": 349.99,
                    "average_rating": 4.8
                },
                {
                    "id": "B08FC5L3RG",
                    "title": "PlayStation 5 - Consola Standard",
                    "description": "Consola de videojuegos de última generación con SSD ultra rápido",
                    "main_category": "Consolas",
                    "price": 499.99,
                    "average_rating": 4.9
                },
                {
                    "id": "B09V3JN27K",
                    "title": "SteelSeries Apex Pro - Teclado Mecánico",
                    "description": "Teclado mecánico gaming con switches ajustables y pantalla OLED",
                    "main_category": "Periféricos Gaming",
                    "price": 199.99,
                    "average_rating": 4.8
                }
            ]

    def generate_test_queries_dataset(self) -> List[Dict[str, Any]]:
        """Genera conjunto de pruebas con consultas y respuestas ideales"""
        logger.info("🧪 Generando dataset de consultas de prueba...")
        
        test_queries = []
        
        for query_type, queries in EVALUATION_QUERIES.items():
            for query in queries:
                # Construir consulta de prueba
                test_query = self._build_test_query(query, query_type)
                test_queries.append(test_query)
        
        # Guardar dataset
        write_jsonl(TEST_QUERIES, test_queries)
        return test_queries
    
    def _build_test_query(self, query: str, query_type: str) -> Dict[str, Any]:
        """Construye una consulta de prueba individual"""
        # Determinar producto ideal
        ideal_product = self.test_generator._find_ideal_product_safe(query)
        
        # Crear respuesta ideal
        ideal_response = self._create_ideal_response(query, ideal_product)
        
        # Crear respuestas de distracción
        distraction_responses = (
            self.test_generator._create_distraction_responses_improved(query, ideal_product) 
            if ideal_product else []
        )
        
        return {
            "query_id": f"test_{len(open(TEST_QUERIES, 'r').readlines()) if TEST_QUERIES.exists() else 0:04d}",
            "query": query,
            "query_type": query_type,
            "ideal_product_id": ideal_product["id"] if ideal_product else None,
            "ideal_response": ideal_response,
            "distraction_responses": distraction_responses,
            "evaluation_criteria": self.test_generator._get_evaluation_criteria(query_type),
            "difficulty_level": self.test_generator._assign_difficulty_level(query, query_type),
            "domain": "gaming",
            "timestamp": datetime.now().isoformat()
        }
    
    def _create_ideal_response(self, query: str, product: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Crea una respuesta ideal para la consulta"""
        if not product:
            return {
                "text": f"Para '{query}', no se encontró un producto específico que cumpla todos los requisitos.",
                "product_id": None,
                "key_points": ["No se encontró producto ideal"],
                "justification": "Consulta muy específica o productos no disponibles",
                "confidence_score": 0.5
            }
        
        return {
            "text": f"Para '{query}', te recomiendo el {product['title']}. {product['description']} Su precio es ${product['price']} y tiene una calificación de {product['average_rating']}/5.",
            "product_id": product["id"],
            "key_points": [
                f"Relevante para: {query}",
                f"Precio: ${product['price']}",
                f"Calificación: {product['average_rating']}/5",
                f"Categoría: {product['main_category']}"
            ],
            "justification": "Este producto cumple con los requisitos específicos de tu consulta",
            "confidence_score": 0.95
        }

    def generate_model_responses_comparison(self, test_queries: List[Dict[str, Any]]) -> None:
        """Genera respuestas de diferentes versiones del modelo para comparación"""
        logger.info("🤖 Generando comparación de respuestas del modelo...")
        
        comparison_data = self._build_comparison_data(test_queries)
        
        # Guardar comparaciones
        write_jsonl(MODEL_COMPARISONS, comparison_data)
        
        # Generar resumen de métricas
        self._generate_metrics_summary(comparison_data)
    
    def _build_comparison_data(self, test_queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Construye datos de comparación para todas las consultas"""
        comparison_data = []
        
        for test_query in test_queries:
            # Generar respuestas de diferentes "versiones" del modelo
            responses = {
                "baseline_model": self.model_generator.generate_baseline_response(test_query),
                "fine_tuned_model": self.model_generator.generate_fine_tuned_response(test_query),
                "rlhf_model": self.model_generator.generate_rlhf_response(test_query),
                "ideal_response": test_query["ideal_response"]
            }
            
            # Calcular métricas
            evaluation_metrics = self.metrics_calculator.calculate_response_metrics(
                responses, test_query["ideal_response"]
            )
            
            comparison_entry = {
                "query_id": test_query["query_id"],
                "query": test_query["query"],
                "query_type": test_query["query_type"],
                "responses": responses,
                "difficulty": test_query["difficulty_level"],
                "evaluation_metrics": evaluation_metrics
            }
            
            comparison_data.append(comparison_entry)
        
        return comparison_data
    
    def _generate_metrics_summary(self, comparison_data: List[Dict[str, Any]]) -> None:
        """Genera resumen de métricas para análisis rápido"""
        logger.info("📊 Generando resumen de métricas...")
        
        summary = self.metrics_calculator.generate_summary(comparison_data)
        
        # Guardar resumen JSON
        with METRICS_SUMMARY.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Generar también CSV para análisis en hojas de cálculo
        self._generate_csv_metrics(comparison_data)
        
        # Generar Excel si pandas está disponible
        self._try_generate_excel(comparison_data)
    
    def _generate_csv_metrics(self, comparison_data: List[Dict[str, Any]]) -> None:
        """Genera métricas en formato CSV"""
        try:
            with DETAILED_METRICS_CSV.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                
                # Encabezados
                headers = ["query_id", "query_type", "difficulty", "model", "relevance", 
                          "accuracy", "completeness", "helpfulness", "product_match"]
                writer.writerow(headers)
                
                # Datos
                for entry in comparison_data:
                    for model in ["baseline_model", "fine_tuned_model", "rlhf_model"]:
                        if model in entry["evaluation_metrics"]:
                            metrics = entry["evaluation_metrics"][model]
                            row = [
                                entry["query_id"],
                                entry["query_type"],
                                entry.get("difficulty", "medium"),
                                model,
                                round(metrics.get("relevance", 0), 3),
                                round(metrics.get("accuracy", 0), 3),
                                round(metrics.get("completeness", 0), 3),
                                round(metrics.get("helpfulness", 0), 3),
                                round(metrics.get("product_match", 0), 3)
                            ]
                            writer.writerow(row)
            
            logger.info(f"✅ {DETAILED_METRICS_CSV} - Métricas detalladas en CSV")
        except Exception as e:
            logger.error(f"❌ Error al generar CSV: {e}")
    
    def _try_generate_excel(self, comparison_data: List[Dict[str, Any]]) -> None:
        """Intenta generar archivo Excel con pandas (opcional)"""
        try:
            import pandas as pd
            
            # Preparar datos para Excel
            excel_data = []
            for entry in comparison_data:
                for model in ["baseline_model", "fine_tuned_model", "rlhf_model"]:
                    if model in entry["evaluation_metrics"]:
                        excel_data.append({
                            "query_id": entry["query_id"],
                            "query": entry["query"][:50],  # Truncar para Excel
                            "query_type": entry["query_type"],
                            "difficulty": entry.get("difficulty", "medium"),
                            "model": model,
                            **entry["evaluation_metrics"][model]
                        })
            
            if excel_data:
                df = pd.DataFrame(excel_data)
                excel_file = EVALUATION_DIR / "metrics" / "detailed_metrics.xlsx"
                df.to_excel(excel_file, index=False)
                logger.info(f"✅ {excel_file} - Métricas en Excel generadas")
                
        except ImportError:
            logger.info("ℹ️ Pandas no disponible, omitiendo generación de Excel")
        except Exception as e:
            logger.error(f"❌ Error al generar Excel: {e}")

    def generate_human_evaluation_dataset(self) -> None:
        """Genera dataset para evaluación humana"""
        logger.info("👥 Generando dataset para evaluación humana...")
        
        # Cargar consultas de prueba
        test_queries = read_jsonl(TEST_QUERIES)
        if not test_queries:
            logger.warning("No hay consultas de prueba, generándolas primero...")
            test_queries = self.generate_test_queries_dataset()
        
        # Seleccionar subset para evaluación humana
        selected_queries = random.sample(test_queries, min(20, len(test_queries)))
        
        human_eval_data = self._build_human_evaluation_data(selected_queries)
        
        # Guardar dataset de evaluación humana
        write_jsonl(HUMAN_EVAL_DATASET, human_eval_data)
        
        # Generar formulario HTML para evaluadores
        self._generate_human_eval_form(human_eval_data)
    
    def _build_human_evaluation_data(self, selected_queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Construye datos para evaluación humana"""
        human_eval_data = []
        
        for query_data in selected_queries:
            # Generar respuestas para evaluación comparativa
            responses = [
                self.model_generator.generate_baseline_response(query_data),
                self.model_generator.generate_fine_tuned_response(query_data),
                self.model_generator.generate_rlhf_response(query_data)
            ]
            
            # Crear pares A/B
            response_pairs = []
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    response_pairs.append({
                        "pair_id": f"{query_data['query_id']}_pair_{len(response_pairs)}",
                        "response_a": responses[i],
                        "response_b": responses[j],
                        "query": query_data["query"]
                    })
            
            human_eval_entry = {
                "eval_id": f"human_eval_{len(human_eval_data):04d}",
                "query_id": query_data["query_id"],
                "query": query_data["query"],
                "query_type": query_data["query_type"],
                "difficulty": query_data["difficulty_level"],
                "response_pairs": response_pairs[:3],  # Limitar a 3 pares máximo
                "evaluation_instructions": {
                    "criteria": query_data["evaluation_criteria"],
                    "scale": "1-5 (1=totalmente inadecuado, 5=excelente)",
                    "time_estimate": "2-3 minutos por par"
                }
            }
            
            human_eval_data.append(human_eval_entry)
        
        return human_eval_data
    
    def _generate_human_eval_form(self, human_eval_data: List[Dict[str, Any]]) -> None:
        """Genera formulario HTML simple para evaluación humana"""
        try:
            html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Evaluación de Respuestas del Modelo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .query { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 8px; }
        .response { background: white; padding: 15px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; }
        .rating { margin: 15px 0; }
        .submit-btn { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>Evaluación de Respuestas del Sistema de Recomendación</h1>
    <p>Por favor, evalúa las siguientes respuestas del modelo según su calidad y utilidad.</p>
"""
            
            for entry in human_eval_data[:5]:  # Mostrar solo 5 para ejemplo
                html_content += f"""
    <div class="query">
        <h3>Consulta #{entry['eval_id']}: {entry['query']}</h3>
        <p><strong>Tipo:</strong> {entry['query_type']} | <strong>Dificultad:</strong> {entry['difficulty']}</p>
"""
                
                for i, pair in enumerate(entry['response_pairs'][:2]):  # Mostrar solo 2 pares
                    html_content += f"""
        <div class="pair">
            <h4>Par de respuestas #{i+1}:</h4>
            <div class="response">
                <h5>Respuesta A:</h5>
                <p>{pair['response_a']['text'][:200]}...</p>
            </div>
            <div class="response">
                <h5>Respuesta B:</h5>
                <p>{pair['response_b']['text'][:200]}...</p>
            </div>
            <div class="rating">
                <label>¿Cuál respuesta es mejor?</label><br>
                <input type="radio" name="pair_{entry['eval_id']}_{i}" value="A"> Respuesta A<br>
                <input type="radio" name="pair_{entry['eval_id']}_{i}" value="B"> Respuesta B<br>
                <input type="radio" name="pair_{entry['eval_id']}_{i}" value="equal"> Ambas iguales<br>
            </div>
            <div class="rating">
                <label>Calidad de la mejor respuesta (1-5):</label>
                <select name="quality_{entry['eval_id']}_{i}">
                    <option value="1">1 - Muy mala</option>
                    <option value="2">2 - Mala</option>
                    <option value="3">3 - Aceptable</option>
                    <option value="4">4 - Buena</option>
                    <option value="5">5 - Excelente</option>
                </select>
            </div>
        </div>
"""
                
                html_content += """
    </div>
    <hr>
"""
            
            html_content += """
    <button class="submit-btn">Enviar Evaluación</button>
    <script>
        document.querySelector('.submit-btn').addEventListener('click', function() {
            alert('¡Gracias por tu evaluación! (Esta es una versión de demostración)');
        });
    </script>
</body>
</html>"""
            
            with HUMAN_EVAL_FORM.open("w", encoding="utf-8") as f:
                f.write(html_content)
            
            logger.info(f"✅ {HUMAN_EVAL_FORM} - Formulario de evaluación HTML generado")
        except Exception as e:
            logger.error(f"❌ Error al generar formulario HTML: {e}")

    def generate_pre_post_comparison(self) -> None:
        """Genera comparación pre vs post entrenamiento"""
        logger.info("📈 Generando comparación pre vs post entrenamiento...")
        
        timeline_data = self._generate_timeline_data()
        
        # Guardar timeline
        write_jsonl(TIMELINE_METRICS, timeline_data)
        
        # Generar gráfico de progreso
        self._generate_progress_chart(timeline_data)
    
    def _generate_timeline_data(self) -> List[Dict[str, Any]]:
        """Genera datos de línea de tiempo de métricas"""
        timeline_data = []
        start_date = datetime.now() - timedelta(days=60)
        
        for day in range(60):
            current_date = start_date + timedelta(days=day)
            
            # Determinar fase
            if day < 20:
                phase = "baseline"
                accuracy_base = 0.65
            elif day < 40:
                phase = "fine_tuning"
                accuracy_base = 0.75
            else:
                phase = "rlhf"
                accuracy_base = 0.85
            
            # Métricas con variación realista
            accuracy = max(0.1, min(1.0, random.normalvariate(accuracy_base, 0.05)))
            
            timeline_entry = {
                "date": current_date.strftime("%Y-%m-%d"),
                "phase": phase,
                "metrics": {
                    "accuracy": round(accuracy, 3),
                    "relevance": round(accuracy * random.uniform(0.9, 1.0), 3),
                    "user_satisfaction": round(accuracy * 100, 1),
                    "response_time_ms": random.randint(800, 1200)
                },
                "training_examples": random.randint(1000, 5000) if day % 7 == 0 else None
            }
            
            timeline_data.append(timeline_entry)
        
        return timeline_data
    
    def _generate_progress_chart(self, timeline_data: List[Dict[str, Any]]) -> None:
        """Genera gráfico de progreso en formato simple"""
        try:
            import matplotlib.pyplot as plt
            
            # Extraer datos para el gráfico
            dates = [entry["date"] for entry in timeline_data]
            accuracy = [entry["metrics"]["accuracy"] for entry in timeline_data]
            phases = [entry["phase"] for entry in timeline_data]
            
            # Crear gráfico
            plt.figure(figsize=(12, 6))
            
            # Colores por fase
            colors = []
            for phase in phases:
                if phase == "baseline":
                    colors.append("red")
                elif phase == "fine_tuning":
                    colors.append("orange")
                else:
                    colors.append("green")
            
            plt.scatter(range(len(dates)), accuracy, c=colors, alpha=0.6, s=50)
            
            # Línea de tendencia
            plt.plot(range(len(dates)), accuracy, 'b-', alpha=0.3, linewidth=2)
            
            # Etiquetas
            plt.xlabel('Días desde inicio')
            plt.ylabel('Precisión')
            plt.title('Progreso del Modelo: Precisión vs Tiempo')
            plt.grid(True, alpha=0.3)
            
            # Leyenda
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.6, label='Baseline'),
                Patch(facecolor='orange', alpha=0.6, label='Fine-tuning'),
                Patch(facecolor='green', alpha=0.6, label='RLHF')
            ]
            plt.legend(handles=legend_elements, loc='upper left')
            
            # Guardar gráfico
            plt.tight_layout()
            plt.savefig(PROGRESS_CHART, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ {PROGRESS_CHART} - Gráfico de progreso generado")
        except ImportError:
            logger.warning("Matplotlib no disponible, omitiendo generación de gráfico")
        except Exception as e:
            logger.error(f"❌ Error al generar gráfico: {e}")

    def run_full_evaluation(self) -> None:
        """Ejecuta la generación completa de datos de evaluación"""
        logger.info("🔬 INICIANDO GENERACIÓN DE DATOS DE EVALUACIÓN POST-ENTRENAMIENTO")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        
        # 1. Generar dataset de consultas de prueba
        test_queries = self.generate_test_queries_dataset()
        
        # 2. Generar comparación de respuestas del modelo
        self.generate_model_responses_comparison(test_queries)
        
        # 3. Generar dataset para evaluación humana
        self.generate_human_evaluation_dataset()
        
        # 4. Generar comparación pre vs post entrenamiento
        self.generate_pre_post_comparison()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("=" * 70)
        logger.info(f"🎯 EVALUACIÓN POST-ENTRENAMIENTO COMPLETADA en {duration:.2f} segundos!")
        self._print_evaluation_summary()
    
    def _print_evaluation_summary(self) -> None:
        """Muestra resumen estadístico de los datos generados"""
        logger.info("📋 RESUMEN ESTADÍSTICO:")
        
        try:
            summary = {
                "test_queries": len(read_jsonl(TEST_QUERIES)) if TEST_QUERIES.exists() else 0,
                "model_comparisons": len(read_jsonl(MODEL_COMPARISONS)) if MODEL_COMPARISONS.exists() else 0,
                "human_evaluation_pairs": 0,
                "timeline_points": len(read_jsonl(TIMELINE_METRICS)) if TIMELINE_METRICS.exists() else 0
            }
            
            # Contar pares de evaluación humana
            if HUMAN_EVAL_DATASET.exists():
                human_data = read_jsonl(HUMAN_EVAL_DATASET)
                summary["human_evaluation_pairs"] = sum(len(entry.get("response_pairs", [])) for entry in human_data)
            
            logger.info(f"   • Consultas de prueba: {summary['test_queries']}")
            logger.info(f"   • Comparaciones de modelo: {summary['model_comparisons']}")
            logger.info(f"   • Pares para evaluación humana: {summary['human_evaluation_pairs']}")
            logger.info(f"   • Puntos en timeline: {summary['timeline_points']}")
            
        except Exception as e:
            logger.error(f"Error al generar resumen: {e}")


def main():
    """Función principal con interfaz de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Generador de datos de evaluación post-entrenamiento",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s --full                          # Ejecuta evaluación completa
  %(prog)s --test-queries                  # Genera solo consultas de prueba
  %(prog)s --model-comparison              # Genera comparación de modelos
  %(prog)s --human-eval --queries 10       # Evaluación humana con 10 consultas
        """
    )
    
    parser.add_argument("--full", action="store_true", 
                       help="Ejecuta evaluación completa")
    parser.add_argument("--test-queries", action="store_true", 
                       help="Genera dataset de consultas de prueba")
    parser.add_argument("--model-comparison", action="store_true", 
                       help="Genera comparación de respuestas del modelo")
    parser.add_argument("--human-eval", action="store_true", 
                       help="Genera dataset para evaluación humana")
    parser.add_argument("--timeline", action="store_true", 
                       help="Genera comparación pre vs post entrenamiento")
    parser.add_argument("--queries", type=int, default=20,
                       help="Número de consultas para evaluación humana (default: 20)")
    parser.add_argument("--products-file", type=Path, default=RAW_PRODUCTS,
                       help="Ruta al archivo de productos (default: data/raw/products.jsonl)")
    
    args = parser.parse_args()
    
    # Si no se especifica ninguna opción, ejecutar evaluación completa
    if not any([args.full, args.test_queries, args.model_comparison, 
                args.human_eval, args.timeline]):
        args.full = True
    
    # Sobrescribir ruta de productos si se especifica
    global RAW_PRODUCTS
    if args.products_file:
        RAW_PRODUCTS = args.products_file
    
    evaluator = PostTrainingEvaluator()
    
    try:
        if args.full:
            evaluator.run_full_evaluation()
        else:
            if args.test_queries:
                evaluator.generate_test_queries_dataset()
            
            if args.model_comparison:
                test_queries = read_jsonl(TEST_QUERIES)
                if not test_queries:
                    logger.warning("Generando consultas de prueba primero...")
                    test_queries = evaluator.generate_test_queries_dataset()
                evaluator.generate_model_responses_comparison(test_queries)
            
            if args.human_eval:
                evaluator.generate_human_evaluation_dataset()
            
            if args.timeline:
                evaluator.generate_pre_post_comparison()
                
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())