#!/usr/bin/env python3
"""
Script COMPLETO Y MEJORADO para generar datos de entrenamiento para el sistema RAG + RL.
Corregido para evitar errores de sampleo cuando las listas son vacías o demasiado pequeñas.
"""

import json
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta
import hashlib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CompleteTrainingDataGenerator:
    def __init__(self, num_users=12, searches_per_user=50):
        self.num_users = num_users
        self.searches_per_user = searches_per_user

        # Estructura completa de directorios REQUERIDA por el sistema
        self.directories = [
            "data/users",
            "data/feedback",
            "data/feedback/rlhf_metrics",
            "data/processed/historial",
            "data/raw",
            "data/processed",
            "data/models/rl_models",
            "logs"
        ]

        # Productos realistas de Amazon - EXPANDIDOS para mejor testing colaborativo
        self.amazon_products = [
            # Consolas (mismo grupo de usuarios)
            {"id": "B0D12C7Y5N", "title": "Nintendo Switch OLED - Mario Red Edition", "main_category": "consoles", "price": 349.99, "average_rating": 4.8},
            {"id": "B0CGRVH2N4", "title": "PS5 Slim Standard Edition", "main_category": "consoles", "price": 499.99, "average_rating": 4.9},
            {"id": "B08FC6MR62", "title": "Xbox Series X 1TB", "main_category": "consoles", "price": 479.99, "average_rating": 4.7},

            # Gaming (mismo grupo)
            {"id": "B0C5N4VYF2", "title": "Logitech G PRO X Superlight 2", "main_category": "gaming", "price": 159.99, "average_rating": 4.6},
            {"id": "B0BDJHN2GS", "title": "HyperX Cloud III Wireless", "main_category": "gaming", "price": 129.99, "average_rating": 4.5},
            {"id": "B0BQKPHZWS", "title": "Razer Huntsman V3 Pro Keyboard", "main_category": "gaming", "price": 199.99, "average_rating": 4.4},

            # Juegos populares (alto rating para filtro colaborativo)
            {"id": "B09V3HN1KC", "title": "Horizon Forbidden West", "main_category": "games", "price": 69.99, "average_rating": 4.8},
            {"id": "B09B8RBFDD", "title": "Elden Ring", "main_category": "games", "price": 59.99, "average_rating": 4.9},
            {"id": "B0BN1ZKJ7D", "title": "The Legend of Zelda: Tears of the Kingdom", "main_category": "games", "price": 69.99, "average_rating": 4.9},
            {"id": "B0B72K7H9N", "title": "Call of Duty: Modern Warfare III", "main_category": "games", "price": 69.99, "average_rating": 4.3},

            # Productos con feedback NEGATIVO para testing
            {"id": "B0BRC9XTQ1", "title": "Generic Gaming Chair Basic", "main_category": "chairs", "price": 89.99, "average_rating": 2.8},
            {"id": "B08K4S6WJ1", "title": "Low Quality Gaming Mouse", "main_category": "gaming", "price": 19.99, "average_rating": 2.5},
        ]

        # Consultas organizadas por grupos de usuarios similares
        self.query_groups = {
            "hardcore_gamers": [
                "mejores juegos ps5 2024", "ratón fps competitivo", "monitor 240hz gaming",
                "teclado mecánico switches speed", "auriculares gaming 7.1"
            ],
            "casual_gamers": [
                "juegos nintendo switch para niños", "juegos familiares multiplayer",
                "nintendo switch oled ofertas", "juegos de aventura mundo abierto"
            ],
            "tech_enthusiasts": [
                "ssd nvme 1tb gaming", "fuente alimentación 750w gold",
                "monitor 4k 144hz", "componentes pc gamer"
            ]
        }

        # Grupos de usuarios más definidos para mejor testing colaborativo
        # FIX: Reemplacé ids inexistentes por ids de ejemplo disponibles o dejé los ids originales,
        # pero el código ahora maneja correctamente fav_products no encontrados.
        self.user_profiles = [
            {
                "type": "hardcore_gamer",
                "preferences": ["games", "consoles", "gaming", "competitive"],
                "age_range": (18, 35),
                "queries": self.query_groups["hardcore_gamers"],
                "fav_products": ["B0CGRVH2N4", "B0C5N4VYF2", "B09B8RBFDD"]  # PS5, Logitech, Elden Ring
            },
            {
                "type": "casual_gamer",
                "preferences": ["family", "kids", "fun", "nintendo"],
                "age_range": (25, 45),
                "queries": self.query_groups["casual_gamers"],
                "fav_products": ["B0D12C7Y5N", "B0BN1ZKJ7D"]  # Nintendo Switch, Zelda
            },
            {
                "type": "tech_enthusiast",
                "preferences": ["pc", "monitors", "hardware", "components"],
                "age_range": (20, 40),
                "queries": self.query_groups["tech_enthusiasts"],
                # Estos IDs no existen en amazon_products en el ejemplo original.
                # Ahora el código manejará correctamente el caso de lists vacías.
                "fav_products": ["B09V3HN1KC", "B0C5N4VYF2"]  # reemplazados por ids válidos
            }
        ]

        # Lista de marcas
        self.available_brands = ["Sony", "Microsoft", "Nintendo", "Logitech", "Razer"]

    # ---------------------------------------------------------------------
    # HELPERS seguros de sampleo (evitan sample sobre poblaciones vacías)
    # ---------------------------------------------------------------------
    def _get_products_by_ids(self, ids_list):
        """Devuelve lista de productos que coinciden con ids_list. No falla si no hay coincidencias."""
        if not ids_list:
            return []
        found = [p for p in self.amazon_products if p["id"] in ids_list]
        return found

    def _safe_sample(self, population, k_min=2, k_max=4):
        """
        Muestra entre k_min y k_max items de population de forma robusta:
        - Si population tiene >= k, usa random.sample.
        - Si population no tiene suficientes elementos pero no está vacía, usa random.choices (con reemplazo).
        - Si population está vacía, usa amazon_products como fallback.
        """
        if population is None:
            population = []
        k = random.randint(k_min, k_max)
        if not population:
            population = self.amazon_products
        if len(population) >= k:
            return random.sample(population, k)
        else:
            # fallback: permitir repeticiones para cumplir k
            return random.choices(population, k=k)

    # ---------------------------------------------------------------------
    # Funciones principales (usando los helpers seguros)
    # ---------------------------------------------------------------------
    def setup_directories(self):
        """Crea TODOS los directorios requeridos por el sistema"""
        logger.info("📁 Creando estructura de directorios...")
        for dir_path in self.directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✅ {dir_path}")

    def generate_user_profile(self, idx):
        """Genera perfil de usuario COMPATIBLE con el nuevo calculate_similarity"""
        profile_type = random.choice(self.user_profiles)
        age = random.randint(*profile_type["age_range"])

        user_id = f"user_{idx:03d}"
        session_id = f"{user_id}_{int(datetime.now().timestamp())}"

        # Precio range basado en tipo de usuario
        if profile_type["type"] == "bargain_hunter":
            price_range = {"min": 0, "max": random.randint(100, 300)}
        elif profile_type["type"] == "hardcore_gamer":
            price_range = {"min": 0, "max": random.randint(500, 1500)}
        else:
            price_range = {"min": 0, "max": random.randint(300, 800)}

        # Protejo la selección de marcas por si la lista es pequeña
        n_brands = min(3, len(self.available_brands))
        preferred_brands = random.sample(self.available_brands, n_brands)

        user_data = {
            "user_id": user_id,
            "session_id": session_id,
            "age": age,
            "gender": random.choice(["male", "female"]),
            "country": random.choice(["Spain", "Mexico", "Argentina", "Colombia"]),
            "language": "es",
            "preferred_categories": profile_type["preferences"],
            "preferred_brands": preferred_brands,
            "avoided_categories": [],
            "price_sensitivity": random.choice(["low", "medium", "high"]),
            "preferred_price_range": price_range,
            "search_history": [],
            "feedback_history": [],
            "purchase_history": [],
            "created_at": (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
            "last_active": datetime.now().isoformat(),
            "total_sessions": random.randint(5, 20)
        }

        # 🔥 GENERAR HISTORIAL COLABORATIVO COHERENTE
        for _ in range(random.randint(15, 40)):
            query = random.choice(profile_type["queries"])

            # Productos preferidos del grupo tienen mayor probabilidad
            if random.random() < 0.7:  # 70% de feedback a productos del grupo
                fav_products = self._get_products_by_ids(profile_type.get("fav_products", []))
                products_shown = self._safe_sample(fav_products, 2, 4)
            else:
                products_shown = self._safe_sample(self.amazon_products, 2, 4)

            selected_product = random.choice(products_shown)

            # 🔥 FEEDBACK POSITIVO CONSISTENTE para productos del grupo
            if selected_product["id"] in profile_type.get("fav_products", []):
                rating = random.choices([4, 5], weights=[0.3, 0.7])[0]  # Mayor probabilidad de 5
            else:
                rating = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.15, 0.25, 0.3, 0.2])[0]

            user_data["feedback_history"].append({
                "query": query,
                "response": f"Recomendación para {query}",
                "rating": rating,
                "timestamp": (datetime.now() - timedelta(days=random.randint(1, 60))).isoformat(),
                "products_shown": [p["id"] for p in products_shown],
                "selected_product": selected_product["id"]
            })

        return user_data

    def generate_success_queries_log(self):
        """Genera success_queries.log con FEEDBACK POSITIVO para filtro colaborativo"""
        logger.info("📝 Generando success_queries.log (solo feedback 4-5)...")

        success_file = Path("data/feedback/success_queries.log")
        existing_ids = set()

        with open(success_file, "w", encoding="utf-8") as f:
            for i in range(80):  # Más consultas exitosas
                profile_type = random.choice(self.user_profiles)
                query = random.choice(profile_type["queries"])

                # Productos del grupo tienen mayor probabilidad
                if random.random() < 0.6:
                    fav_products = self._get_products_by_ids(profile_type.get("fav_products", []))
                    product = random.choice(fav_products) if fav_products else random.choice(self.amazon_products)
                else:
                    product = random.choice(self.amazon_products)

                query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
                entry_id = f"session_{i:03d}-{query_hash}"

                if entry_id not in existing_ids:
                    existing_ids.add(entry_id)

                    record = {
                        "timestamp": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                        "session_id": f"session_{i:03d}",
                        "query": query,
                        "response": f"Te recomiendo {product['title']}. Calificación {product['average_rating']}/5 - ${product['price']}",
                        "feedback": random.choices([4, 5], weights=[0.3, 0.7])[0],  # Solo positivo
                        "selected_product_id": product["id"],
                        "source_file": "conversation_001.json",
                        "processed": False
                    }

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"  ✅ {success_file} - {len(existing_ids)} registros POSITIVOS")

    def generate_failed_queries_log(self):
        """Genera failed_queries.log con FEEDBACK NEGATIVO para soft negative filtering"""
        logger.info("📝 Generando failed_queries.log (feedback 1-3)...")

        failed_file = Path("data/feedback/failed_queries.log")
        existing_ids = set()

        failure_reasons = ["product_not_found", "incomplete_data", "low_quality_response"]

        with open(failed_file, "w", encoding="utf-8") as f:
            for i in range(30):  # Menos fallos (más realista)
                query = random.choice([q for group in self.query_groups.values() for q in group])

                query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
                entry_id = f"session_fail_{i:03d}-{query_hash}"

                if entry_id not in existing_ids:
                    existing_ids.add(entry_id)

                    # 🔥 PRODUCTOS CON BAJO RATING para testing de negative filtering
                    low_rated_products = [p for p in self.amazon_products if p["average_rating"] < 3.5]
                    mentioned_product = random.choice(low_rated_products) if low_rated_products else random.choice(self.amazon_products)

                    record = {
                        "timestamp": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                        "session_id": f"session_fail_{i:03d}",
                        "query": query,
                        "response": f"No encontré buenas opciones para '{query}'. {mentioned_product['title']} tiene baja calificación.",
                        "feedback": random.choices([1, 2, 3], weights=[0.3, 0.4, 0.3])[0],  # Solo negativo
                        "failure_reason": random.choice(failure_reasons),
                        "selected_product_id": mentioned_product["id"],  # Para negative filtering
                        "source_file": "conversation_001.json",
                        "processed": False
                    }

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"  ✅ {failed_file} - {len(existing_ids)} registros NEGATIVOS")

    def generate_feedback_weights(self):
        """Genera pesos de feedback iniciales para testing del decay temporal"""
        logger.info("⚖️ Generando pesos de feedback iniciales...")

        weights_file = Path("data/feedback/feedback_weights.json")
        weights = {}

        # Productos populares tienen pesos altos
        for product in self.amazon_products:
            if product["average_rating"] >= 4.5:
                weights[product["id"]] = random.randint(3, 8)
            elif product["average_rating"] >= 4.0:
                weights[product["id"]] = random.randint(1, 4)
            else:
                weights[product["id"]] = random.randint(0, 1)

        with open(weights_file, "w", encoding="utf-8") as f:
            json.dump(weights, f, ensure_ascii=False, indent=2)

        logger.info(f"  ✅ {weights_file} - {len(weights)} productos con pesos")

    def generate_realtime_feedback(self):
        """Genera feedback en tiempo real con INFERENCIA MEJORADA"""
        logger.info("📝 Generando feedback en tiempo real...")

        for day in range(7):
            date_str = (datetime.now() - timedelta(days=day)).strftime("%Y-%m-%d")
            feedback_file = Path(f"data/feedback/feedback_{date_str}.jsonl")

            with open(feedback_file, "w", encoding="utf-8") as f:
                for i in range(random.randint(8, 20)):
                    profile_type = random.choice(self.user_profiles)
                    query = random.choice(profile_type["queries"])

                    # Productos mostrados basados en grupo de usuario
                    if random.random() < 0.7:
                        fav_products = self._get_products_by_ids(profile_type.get("fav_products", []))
                        products = self._safe_sample(fav_products, 2, 4)
                    else:
                        products = self._safe_sample(self.amazon_products, 2, 4)

                    # Inferencia mejorada: producto más relevante al query
                    selected_product = self._infer_selected_product(query, products)
                    rating = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.1, 0.2, 0.35, 0.3])[0]

                    record = {
                        "timestamp": (datetime.now() - timedelta(days=day, hours=random.randint(1, 23))).isoformat(),
                        "query": query,
                        "response": f"Recomendación: {selected_product['title']} - ${selected_product['price']}",
                        "feedback": rating,
                        "products_shown": [p["id"] for p in products],
                        "selected_product_id": selected_product["id"],  # ✅ SIEMPRE presente
                        "user_age": random.randint(18, 45),
                        "user_gender": random.choice(["male", "female"]),
                        "user_country": random.choice(["Spain", "Mexico"]),
                        "inference_method": "multi_strategy",
                        "processed": False
                    }

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            logger.info(f"  ✅ {feedback_file}")

    def _infer_selected_product(self, query: str, products: list) -> dict:
        """Simula la inferencia mejorada del WorkingRAGAgent"""
        query_lower = query.lower()

        # Estrategia 1: Coincidencia de términos en título
        for product in products:
            title_lower = product["title"].lower()
            if any(term in title_lower for term in query_lower.split()):
                return product

        # Estrategia 2: Producto con mejor rating
        best_rated = max(products, key=lambda x: x["average_rating"])
        return best_rated

    def generate_rlhf_metrics(self):
        """Genera métricas RLHF que muestran MEJORA PROGRESIVA"""
        logger.info("📊 Generando métricas RLHF...")

        metrics_file = Path("data/feedback/rlhf_metrics/training_metrics.jsonl")
        base_accuracy = 0.60

        with open(metrics_file, "w", encoding="utf-8") as f:
            for i in range(20):
                # Mejora progresiva con algunos altibajos
                if i < 5:
                    improvement = random.uniform(0.02, 0.08)  # Mejora rápida inicial
                elif i < 15:
                    improvement = random.uniform(-0.03, 0.05)  # Estabilización
                else:
                    improvement = random.uniform(0.01, 0.03)  # Mejora lenta final

                new_accuracy = max(0.5, min(0.92, base_accuracy + improvement))

                record = {
                    "timestamp": (datetime.now() - timedelta(days=20 - i)).isoformat(),
                    "examples_used": random.randint(80, 250),
                    "previous_accuracy": round(base_accuracy, 3),
                    "new_accuracy": round(new_accuracy, 3),
                    "improvement": round(improvement, 3),
                    "training_time_seconds": random.randint(400, 2200),
                    "success": improvement > 0
                }

                base_accuracy = new_accuracy
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"  ✅ {metrics_file} - Progreso de entrenamiento simulado")

    def run(self):
        """Ejecuta la generación completa ACTUALIZADA"""
        logger.info("🚀 INICIANDO GENERACIÓN DE DATOS ACTUALIZADA")
        logger.info("=" * 70)
        logger.info("💡 INCLUYE TODAS LAS MEJORAS IMPLEMENTADAS:")
        logger.info("   ✅ Cache MD5 para filtro colaborativo")
        logger.info("   ✅ Soft negative filtering")
        logger.info("   ✅ Decay temporal automático")
        logger.info("   ✅ Pesos dinámicos normalizados")
        logger.info("   ✅ Inferencia mejorada de productos")
        logger.info("=" * 70)

        self.setup_directories()

        # 1. Perfiles de usuarios con comportamiento colaborativo coherente
        logger.info("👥 Generando perfiles de usuarios COLABORATIVOS...")
        for i in range(1, self.num_users + 1):
            user_profile = self.generate_user_profile(i)
            user_file = Path(f"data/users/{user_profile['user_id']}.json")
            with open(user_file, "w", encoding="utf-8") as f:
                json.dump(user_profile, f, ensure_ascii=False, indent=2)
            logger.info(f"  ✅ {user_file} - {user_profile['preferred_categories'][0]}")

        # 2. Sistema de feedback mejorado
        self.generate_success_queries_log()
        self.generate_failed_queries_log()
        self.generate_feedback_weights()
        self.generate_realtime_feedback()

        # 3. Métricas RLHF realistas
        self.generate_rlhf_metrics()

        logger.info("=" * 70)
        logger.info("🎉 GENERACIÓN COMPLETA FINALIZADA!")
        logger.info("📊 DATOS OPTIMIZADOS PARA:")
        logger.info("   🔍 Filtro colaborativo con usuarios similares")
        logger.info("   ⚖️  Soft negative filtering automático")
        logger.info("   📈 Reentrenamiento RLHF progresivo")
        logger.info("   🎯 Inferencia inteligente de productos")
        logger.info("=" * 70)


if __name__ == "__main__":
    generator = CompleteTrainingDataGenerator(num_users=15, searches_per_user=50)
    generator.run()
