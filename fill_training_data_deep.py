#!/usr/bin/env python3
"""
Script COMPLETAMENTE MODIFICADO y CORREGIDO para generar TODOS los datos específicos que requiere el sistema de evaluación Deepeval.
CORREGIDO: Manejo de claves faltantes en productos.
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
    def __init__(self, num_users=50, searches_per_user=50):
        self.num_users = num_users
        self.searches_per_user = searches_per_user

        # Estructura EXACTA de directorios que requiere el script deepeval
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

        # Productos REALES de Amazon Gaming - CORREGIDO: Todas las claves presentes
        self.amazon_products = [
            # Consolas y hardware
            {"id": "B0CGRVH2N4", "title": "PlayStation 5 Standard Edition", "main_category": "consoles", "price": 499.99, "average_rating": 4.9},
            {"id": "B08FC6MR62", "title": "Xbox Series X 1TB Console", "main_category": "consoles", "price": 479.99, "average_rating": 4.7},
            {"id": "B0D12C7Y5N", "title": "Nintendo Switch OLED Model", "main_category": "consoles", "price": 349.99, "average_rating": 4.8},
            
            # Periféricos gaming
            {"id": "B0C5N4VYF2", "title": "Logitech G PRO X Superlight 2 Gaming Mouse", "main_category": "gaming", "price": 159.99, "average_rating": 4.6},
            {"id": "B0BDJHN2GS", "title": "HyperX Cloud III Wireless Gaming Headset", "main_category": "gaming", "price": 129.99, "average_rating": 4.5},
            {"id": "B0BQKPHZWS", "title": "Razer Huntsman V3 Pro Gaming Keyboard", "main_category": "gaming", "price": 199.99, "average_rating": 4.4},
            {"id": "B08N5WRWNW", "title": "SteelSeries Arctis Pro Wireless Headset", "main_category": "gaming", "price": 329.99, "average_rating": 4.7},
            
            # Juegos populares
            {"id": "B09V3HN1KC", "title": "God of War Ragnarök PS5", "main_category": "games", "price": 69.99, "average_rating": 4.9},
            {"id": "B09B8RBFDD", "title": "Elden Ring Standard Edition", "main_category": "games", "price": 59.99, "average_rating": 4.8},
            {"id": "B0BN1ZKJ7D", "title": "The Legend of Zelda: Tears of the Kingdom", "main_category": "games", "price": 69.99, "average_rating": 4.9},
            {"id": "B0B72K7H9N", "title": "Call of Duty: Modern Warfare III", "main_category": "games", "price": 69.99, "average_rating": 4.3},
            {"id": "B0CJ8CM6VJ", "title": "Spider-Man 2 PS5", "main_category": "games", "price": 69.99, "average_rating": 4.8},
            {"id": "B0C4Y3W3VV", "title": "Starfield Xbox Series X", "main_category": "games", "price": 69.99, "average_rating": 4.2},
            
            # Monitores y hardware
            {"id": "B0B3X66FVQ", "title": "Samsung Odyssey G7 27\" Gaming Monitor", "main_category": "monitors", "price": 599.99, "average_rating": 4.6},
            {"id": "B09VCTV1PX", "title": "ASUS ROG Swift 27\" 1440P Monitor", "main_category": "monitors", "price": 699.99, "average_rating": 4.7},
            
            # Sillas gaming
            {"id": "B08N5K5F6S", "title": "Secretlab Titan Evo 2022 Gaming Chair", "main_category": "chairs", "price": 489.99, "average_rating": 4.8},
            {"id": "B09Y6KVBQ8", "title": "Razer Enki Pro Gaming Chair", "main_category": "chairs", "price": 499.99, "average_rating": 4.5},
            
            # Productos con bajo rating - CORREGIDO: Clave average_rating presente
            {"id": "B0BRC9XTQ1", "title": "Generic Gaming Chair Basic Model", "main_category": "chairs", "price": 89.99, "average_rating": 2.8},
            {"id": "B08K4S6WJ1", "title": "Low Quality Gaming Mouse Generic", "main_category": "gaming", "price": 19.99, "average_rating": 2.5},
        ]

        # Consultas ESPECÍFICAS que el script deepeval usa para evaluación
        self.evaluation_queries = [
            "juegos de acción para ps5",
            "auriculares gaming inalámbricos", 
            "teclado mecánico gamer",
            "monitor gaming 144hz", 
            "silla gamer ergonómica",
            "nintendo switch oled",
            "xbox series x",
            "juegos multiplayer pc", 
            "ratón gaming inalámbrico",
            "ssd nvme 1tb",
            "playstation 5 slim",
            "god of war ragnarok",
            "elden ring xbox",
            "monitor 4k gaming",
            "headset wireless gaming"
        ]

        # Grupos de usuarios para filtro colaborativo
        self.user_profiles = [
            {
                "type": "hardcore_gamer",
                "preferences": ["games", "consoles", "gaming", "competitive"],
                "age_range": (18, 35),
                "fav_products": ["B0CGRVH2N4", "B0C5N4VYF2", "B09B8RBFDD", "B09V3HN1KC"]
            },
            {
                "type": "casual_gamer", 
                "preferences": ["family", "kids", "fun", "nintendo"],
                "age_range": (25, 45),
                "fav_products": ["B0D12C7Y5N", "B0BN1ZKJ7D", "B0CJ8CM6VJ"]
            },
            {
                "type": "tech_enthusiast",
                "preferences": ["pc", "monitors", "hardware", "components"],
                "age_range": (20, 40),
                "fav_products": ["B0B3X66FVQ", "B09VCTV1PX", "B08N5K5F6S"]
            }
        ]

        self.available_brands = ["Sony", "Microsoft", "Nintendo", "Logitech", "Razer", "SteelSeries", "Samsung", "ASUS", "Secretlab"]

    def setup_directories(self):
        """Crea TODOS los directorios que el script deepeval requiere"""
        logger.info("📁 Creando estructura de directorios EXACTA para deepeval...")
        for dir_path in self.directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✅ {dir_path}")

    def get_product_rating(self, product):
        """Método seguro para obtener el rating de un producto"""
        return product.get('average_rating', 4.0)  # Valor por defecto si no existe

    def generate_user_profile(self, idx):
        """Genera perfil de usuario COMPATIBLE con el sistema de evaluación"""
        profile_type = random.choice(self.user_profiles)
        age = random.randint(*profile_type["age_range"])

        user_id = f"user_{idx:03d}"
        
        user_data = {
            "id": user_id,  # ✅ CLAVE: El script deepeval espera campo "id"
            "user_id": user_id,
            "session_id": f"{user_id}_{int(datetime.now().timestamp())}",
            "age": age,
            "gender": random.choice(["male", "female"]),
            "country": random.choice(["Spain", "Mexico", "Argentina", "Colombia"]),
            "language": "es",
            "categories": profile_type["preferences"],  # ✅ CLAVE: Para collaborative filtering
            "preferred_categories": profile_type["preferences"],
            "preferred_brands": random.sample(self.available_brands, min(3, len(self.available_brands))),
            "price_sensitivity": random.choice(["low", "medium", "high"]),
            "preferred_price_range": {"min": 0, "max": random.randint(200, 1000)},
            "search_history": [],
            "feedback_history": [],
            "purchase_history": [],
            "created_at": (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
            "last_active": datetime.now().isoformat(),
            "total_sessions": random.randint(5, 50)
        }

        # Generar historial de búsquedas REALISTAS para evaluación
        for _ in range(random.randint(20, 60)):
            query = random.choice(self.evaluation_queries)
            products_shown = random.sample(self.amazon_products, random.randint(3, 8))
            selected_product = random.choice(products_shown)

            user_data["search_history"].append({
                "query": query,
                "timestamp": (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat(),
                "products_shown": [p["id"] for p in products_shown],
                "selected_product": selected_product["id"]
            })

        return user_data

    def generate_products_file(self):
        """Genera el archivo products.json que el script deepeval necesita"""
        logger.info("📦 Generando products.json para evaluación...")
        
        products_file = Path("data/processed/products.json")
        
        with open(products_file, "w", encoding="utf-8") as f:
            json.dump(self.amazon_products, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  ✅ {products_file} - {len(self.amazon_products)} productos")

    def generate_raw_data_files(self):
        """Genera archivos raw en data/raw/ que el script deepeval puede cargar como fallback"""
        logger.info("📄 Generando archivos raw de respaldo...")
        
        # Archivo de conversaciones
        conversations_file = Path("data/raw/conversations.jsonl")
        with open(conversations_file, "w", encoding="utf-8") as f:
            for i in range(100):
                query = random.choice(self.evaluation_queries)
                product = random.choice(self.amazon_products)
                
                # CORREGIDO: Usar método seguro para obtener rating
                rating = self.get_product_rating(product)
                
                record = {
                    "id": f"conv_{i:03d}",
                    "query": query,
                    "response": f"Basado en tu búsqueda '{query}', te recomiendo {product['title']} con calificación {rating}/5",
                    "product_id": product["id"],
                    "timestamp": (datetime.now() - timedelta(days=random.randint(1, 60))).isoformat(),
                    "user_id": f"user_{random.randint(1, 50):03d}",
                    "rating": random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.1, 0.15, 0.3, 0.4])[0]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.info(f"  ✅ {conversations_file}")

    def generate_success_queries_log(self):
        """Genera success_queries.log con datos REALES para evaluación"""
        logger.info("📝 Generando success_queries.log para evaluación...")

        success_file = Path("data/feedback/success_queries.log")
        
        with open(success_file, "w", encoding="utf-8") as f:
            for i in range(150):
                query = random.choice(self.evaluation_queries)
                product = random.choice(self.amazon_products)
                
                # Crear ground truth relationships específicas para evaluación
                if "ps5" in query.lower():
                    ps5_products = [p for p in self.amazon_products if "ps5" in p["title"].lower() or "playstation" in p["title"].lower()]
                    if ps5_products:
                        product = random.choice(ps5_products)
                elif "nintendo" in query.lower() or "switch" in query.lower():
                    nintendo_products = [p for p in self.amazon_products if "nintendo" in p["title"].lower() or "switch" in p["title"].lower()]
                    if nintendo_products:
                        product = random.choice(nintendo_products)
                elif "xbox" in query.lower():
                    xbox_products = [p for p in self.amazon_products if "xbox" in p["title"].lower()]
                    if xbox_products:
                        product = random.choice(xbox_products)

                # CORREGIDO: Usar método seguro para rating
                rating = self.get_product_rating(product)

                record = {
                    "timestamp": (datetime.now() - timedelta(days=random.randint(1, 45))).isoformat(),
                    "session_id": f"session_{i:03d}",
                    "query": query,
                    "response": f"Recomendación: {product['title']} - ${product['price']} - Rating: {rating}/5",
                    "feedback": random.choices([4, 5], weights=[0.3, 0.7])[0],
                    "selected_product_id": product["id"],  # ✅ CLAVE: Para ground truth
                    "source_file": "conversations.jsonl",
                    "processed": random.choice([True, False])
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"  ✅ {success_file} - 150 registros de éxito")

    def generate_failed_queries_log(self):
        """Genera failed_queries.log para negative sampling"""
        logger.info("📝 Generando failed_queries.log...")

        failed_file = Path("data/feedback/failed_queries.log")
        
        with open(failed_file, "w", encoding="utf-8") as f:
            for i in range(40):
                query = random.choice(self.evaluation_queries)
                # CORREGIDO: Usar método seguro para filtrar productos con bajo rating
                low_rated_products = [p for p in self.amazon_products if self.get_product_rating(p) < 3.5]
                product = random.choice(low_rated_products) if low_rated_products else random.choice(self.amazon_products)

                record = {
                    "timestamp": (datetime.now() - timedelta(days=random.randint(1, 45))).isoformat(),
                    "session_id": f"session_fail_{i:03d}",
                    "query": query,
                    "response": f"No se encontraron buenos resultados para '{query}'. Producto alternativo: {product['title']}",
                    "feedback": random.choices([1, 2, 3], weights=[0.3, 0.4, 0.3])[0],
                    "failure_reason": random.choice(["product_not_found", "low_quality_match", "insufficient_data"]),
                    "selected_product_id": product["id"],  # ✅ CLAVE: Para negative filtering
                    "source_file": "conversations.jsonl", 
                    "processed": False
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"  ✅ {failed_file} - 40 registros de fallos")

    def generate_feedback_weights(self):
        """Genera feedback_weights.json para testing de temporal decay"""
        logger.info("⚖️ Generando feedback_weights.json...")

        weights_file = Path("data/feedback/feedback_weights.json")
        weights = {}

        for product in self.amazon_products:
            # CORREGIDO: Usar método seguro para rating
            rating = self.get_product_rating(product)
            if rating >= 4.5:
                weights[product["id"]] = random.randint(5, 10)
            elif rating >= 4.0:
                weights[product["id"]] = random.randint(2, 6)
            else:
                weights[product["id"]] = random.randint(0, 2)

        with open(weights_file, "w", encoding="utf-8") as f:
            json.dump(weights, f, ensure_ascii=False, indent=2)

        logger.info(f"  ✅ {weights_file} - {len(weights)} productos con pesos")

    def generate_realtime_feedback(self):
        """Genera archivos de feedback en tiempo real para RLHF"""
        logger.info("🔄 Generando feedback en tiempo real...")

        for day in range(7, 0, -1):
            date_str = (datetime.now() - timedelta(days=day)).strftime("%Y-%m-%d")
            feedback_file = Path(f"data/feedback/feedback_{date_str}.jsonl")

            with open(feedback_file, "w", encoding="utf-8") as f:
                for i in range(random.randint(10, 25)):
                    query = random.choice(self.evaluation_queries)
                    products = random.sample(self.amazon_products, random.randint(3, 6))
                    selected_product = random.choice(products)

                    record = {
                        "timestamp": (datetime.now() - timedelta(days=day, hours=random.randint(1, 23))).isoformat(),
                        "query": query,
                        "response": f"Recomendación: {selected_product['title']} - ${selected_product['price']}",
                        "feedback": random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.1, 0.15, 0.3, 0.4])[0],
                        "products_shown": [p["id"] for p in products],
                        "selected_product_id": selected_product["id"],
                        "user_id": f"user_{random.randint(1, 50):03d}",
                        "processed": False
                    }

                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            logger.info(f"  ✅ {feedback_file}")

    def generate_rlhf_metrics(self):
        """Genera training_metrics.jsonl para evaluación RLHF"""
        logger.info("📊 Generando training_metrics.jsonl...")

        metrics_file = Path("data/feedback/rlhf_metrics/training_metrics.jsonl")
        base_accuracy = 0.65

        with open(metrics_file, "w", encoding="utf-8") as f:
            for i in range(25):
                if i < 8:
                    improvement = random.uniform(0.03, 0.10)
                elif i < 18:
                    improvement = random.uniform(-0.02, 0.06)
                else:
                    improvement = random.uniform(0.01, 0.04)

                new_accuracy = max(0.6, min(0.95, base_accuracy + improvement))

                record = {
                    "timestamp": (datetime.now() - timedelta(days=25 - i)).isoformat(),
                    "examples_used": random.randint(100, 300),
                    "previous_accuracy": round(base_accuracy, 4),
                    "new_accuracy": round(new_accuracy, 4),
                    "improvement": round(improvement, 4),
                    "training_time_seconds": random.randint(300, 1800),
                    "success": improvement > 0,
                    "loss": random.uniform(0.1, 0.8)
                }

                base_accuracy = new_accuracy
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"  ✅ {metrics_file} - 25 métricas de entrenamiento")

    def create_evaluation_ground_truth(self):
        """Crea datos específicos para las funciones de evaluación del primer script"""
        logger.info("🎯 Creando datos específicos para evaluación...")
        
        log_file = Path("logs/amazon_recommendations.log")
        log_file.parent.mkdir(exist_ok=True)
        log_file.write_text("")  # Log vacío pero existente
        
        logger.info(f"  ✅ {log_file} - Listo para evaluación")

    def run(self):
        """Ejecuta la generación COMPLETA de datos para el script deepeval"""
        logger.info("🚀 INICIANDO GENERACIÓN DE DATOS PARA DEEPEVAL")
        logger.info("=" * 70)
        logger.info("🎯 GENERANDO DATOS ESPECÍFICOS PARA:")
        logger.info("   ✅ Evaluación básica RAG (MRR, NDCG, BLEU, ROUGE)")
        logger.info("   ✅ Filtro colaborativo con usuarios similares") 
        logger.info("   ✅ Evaluación RLHF con métricas progresivas")
        logger.info("   ✅ Sistema híbrido completo")
        logger.info("=" * 70)

        self.setup_directories()

        # 1. Usuarios para evaluación colaborativa
        logger.info("👥 Generando 50 usuarios para evaluación...")
        for i in range(1, self.num_users + 1):
            user_profile = self.generate_user_profile(i)
            user_file = Path(f"data/users/{user_profile['user_id']}.json")
            with open(user_file, "w", encoding="utf-8") as f:
                json.dump(user_profile, f, ensure_ascii=False, indent=2)

        # 2. Archivos de productos esenciales
        self.generate_products_file()
        self.generate_raw_data_files()

        # 3. Sistema de feedback completo
        self.generate_success_queries_log()
        self.generate_failed_queries_log() 
        self.generate_feedback_weights()
        self.generate_realtime_feedback()

        # 4. Métricas y datos de evaluación
        self.generate_rlhf_metrics()
        self.create_evaluation_ground_truth()

        logger.info("=" * 70)
        logger.info("🎉 GENERACIÓN COMPLETADA EXITOSAMENTE!")
        logger.info("📁 ESTRUCTURA CREADA:")
        logger.info(f"   👥 {self.num_users} usuarios en data/users/")
        logger.info(f"   📦 {len(self.amazon_products)} productos en data/processed/products.json")
        logger.info(f"   📝 150+ registros en data/feedback/success_queries.log") 
        logger.info(f"   📊 25 métricas RLHF en data/feedback/rlhf_metrics/")
        logger.info("=" * 70)
        logger.info("✅ EL SCRIPT DEEPEVAL AHORA TIENE TODOS LOS DATOS QUE NECESITA")


if __name__ == "__main__":
    generator = CompleteTrainingDataGenerator(num_users=50, searches_per_user=50)
    generator.run()