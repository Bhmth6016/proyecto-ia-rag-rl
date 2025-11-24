#!/usr/bin/env python3
"""
Script COMPLETO Y MEJORADO para generar datos de entrenamiento para el sistema RAG + RL.
INCLUYE TODOS LOS ARCHIVOS Y ESTRUCTURAS REQUERIDAS:

✅ data/users/*.json - Perfiles de usuarios con datos demográficos
✅ data/feedback/success_queries.log - Consultas exitosas (formato exacto)
✅ data/feedback/failed_queries.log - Consultas fallidas (formato exacto)  
✅ data/feedback/feedback_*.jsonl - Feedback en tiempo real
✅ data/processed/historial/conversation_*.json - Historial de conversaciones
✅ data/feedback/rlhf_metrics/training_metrics.jsonl - Métricas RLHF
✅ data/raw/*.jsonl - Datos de productos simulados
✅ data/processed/products.json - Productos procesados
✅ data/models/rl_models/ - Directorio para modelos RLHF
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

        # Productos realistas de Amazon simulados - MÁS COMPLETOS
        self.amazon_products = [
            # Consolas
            {"id": "B0D12C7Y5N", "title": "Nintendo Switch OLED - Mario Red Edition", "main_category": "consoles", "price": 349.99, "average_rating": 4.8, "description": "Consola Nintendo Switch OLED edición especial Mario Rojo"},
            {"id": "B0CGRVH2N4", "title": "PS5 Slim Standard Edition", "main_category": "consoles", "price": 499.99, "average_rating": 4.9, "description": "PlayStation 5 Slim con 1TB SSD"},
            {"id": "B08FC6MR62", "title": "Xbox Series X 1TB", "main_category": "consoles", "price": 479.99, "average_rating": 4.7, "description": "Xbox Series X con 1TB de almacenamiento"},
            
            # Gaming
            {"id": "B0C5N4VYF2", "title": "Logitech G PRO X Superlight 2", "main_category": "gaming", "price": 159.99, "average_rating": 4.6, "description": "Ratón gaming inalámbrico ultra ligero"},
            {"id": "B0BDJHN2GS", "title": "HyperX Cloud III Wireless", "main_category": "gaming", "price": 129.99, "average_rating": 4.5, "description": "Auriculares gaming inalámbricos 7.1"},
            {"id": "B0BQKPHZWS", "title": "Razer Huntsman V3 Pro Keyboard", "main_category": "gaming", "price": 199.99, "average_rating": 4.4, "description": "Teclado mecánico gaming óptico"},
            
            # Monitores
            {"id": "B0CGLX7JYF", "title": "LG Ultragear 27'' 144Hz Gaming Monitor", "main_category": "monitors", "price": 299.99, "average_rating": 4.7, "description": "Monitor gaming 27 pulgadas 144Hz IPS"},
            {"id": "B08N6K26RQ", "title": "Samsung Odyssey G5 32''", "main_category": "monitors", "price": 349.99, "average_rating": 4.6, "description": "Monitor curvo gaming 32 pulgadas 144Hz"},
            
            # Juegos
            {"id": "B09V3HN1KC", "title": "Horizon Forbidden West", "main_category": "games", "price": 69.99, "average_rating": 4.8, "description": "Juego de aventura y acción para PS5"},
            {"id": "B09B8RBFDD", "title": "Elden Ring", "main_category": "games", "price": 59.99, "average_rating": 4.9, "description": "Juego de rol y acción mundo abierto"},
            {"id": "B0BN1ZKJ7D", "title": "The Legend of Zelda: Tears of the Kingdom", "main_category": "games", "price": 69.99, "average_rating": 4.9, "description": "Aventura de Zelda para Nintendo Switch"},
            {"id": "B0B72K7H9N", "title": "Call of Duty: Modern Warfare III", "main_category": "games", "price": 69.99, "average_rating": 4.3, "description": "Shooter en primera persona"},
            
            # Sillas Gaming
            {"id": "B0BRC9XTQS", "title": "Silla Gamer Corsair T3 Rush", "main_category": "chairs", "price": 249.99, "average_rating": 4.4, "description": "Silla gaming ergonómica con soporte lumbar"},
            {"id": "B09TZJ4W8C", "title": "Noblechairs Hero", "main_category": "chairs", "price": 399.99, "average_rating": 4.7, "description": "Silla gaming premium cuero negro"},
            
            # Accesorios PC
            {"id": "B08K4S6WJ9", "title": "Corsair RM750x Power Supply", "main_category": "components", "price": 119.99, "average_rating": 4.8, "description": "Fuente alimentación 750W 80 Plus Gold"},
            {"id": "B09V1P3X2Z", "title": "WD Black SN850X NVMe SSD 1TB", "main_category": "components", "price": 129.99, "average_rating": 4.9, "description": "SSD NVMe PCIe 4.0 de alto rendimiento"},
        ]

        self.query_list = [
            "juegos nintendo switch para niños",
            "mejores juegos ps5 2024",
            "consolas baratas gaming",
            "auriculares gaming inalámbricos con micrófono",
            "teclado mecánico gamer rgb",
            "silla gamer económica ergonómica",
            "monitor 144Hz barato 27 pulgadas",
            "ratón inalámbrico gaming ligero",
            "juegos de aventura mundo abierto",
            "accesorios para pc gamer rgb",
            "nintendo switch oled ofertas",
            "playstation 5 juegos exclusivos",
            "xbox series x vs ps5 comparativa",
            "mejor silla gaming para espalda",
            "monitor 4k gaming 144hz",
            "teclados mecánicos switches red",
            "auriculares ps5 3d audio",
            "juegos nintendo switch multiplayer",
            "gaming chair under 200 euros",
            "mejor ratón fps competitivo"
        ]

        # Tipos de usuarios más detallados
        self.user_profiles = [
            {"type": "hardcore_gamer", "preferences": ["games", "consoles", "gaming", "competitive"], "age_range": (18, 35)},
            {"type": "casual_gamer", "preferences": ["family", "kids", "fun", "nintendo"], "age_range": (25, 45)},
            {"type": "tech_enthusiast", "preferences": ["pc", "monitors", "hardware", "components"], "age_range": (20, 40)},
            {"type": "bargain_hunter", "preferences": ["cheap", "budget", "ofertas", "descuento"], "age_range": (18, 60)},
            {"type": "collector", "preferences": ["exclusive", "limited", "special edition"], "age_range": (25, 50)},
            {"type": "streamer", "preferences": ["audio", "webcam", "streaming", "microphone"], "age_range": (18, 35)},
        ]

    def setup_directories(self):
        """Crea TODOS los directorios requeridos por el sistema"""
        logger.info("📁 Creando estructura de directorios...")
        for dir_path in self.directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✅ {dir_path}")

    # ------------------------------------------------------------------------------------
    # GENERACIÓN DE USUARIOS (formato compatible con UserManager)
    # ------------------------------------------------------------------------------------
    def generate_user_profile(self, idx):
        """Genera perfil de usuario compatible con UserManager y UserProfile"""
        profile = random.choice(self.user_profiles)
        age = random.randint(*profile["age_range"])
        
        user_id = f"user_{idx:03d}"
        session_id = f"{user_id}_{int(datetime.now().timestamp())}"
        
        user_data = {
            "user_id": user_id,
            "session_id": session_id,
            "age": age,
            "gender": random.choice(["male", "female"]),
            "country": random.choice(["Spain", "Mexico", "Argentina", "Colombia", "Chile"]),
            "language": "es",
            "preferred_categories": profile["preferences"],
            "preferred_brands": random.sample(["Sony", "Microsoft", "Nintendo", "Logitech", "Razer", "Corsair"], 3),
            "avoided_categories": [],
            "price_sensitivity": random.choice(["low", "medium", "high"]),
            "preferred_price_range": {"min": 0, "max": random.randint(300, 1000)},
            "search_history": [],
            "feedback_history": [],
            "purchase_history": [],
            "created_at": (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
            "last_active": datetime.now().isoformat(),
            "total_sessions": random.randint(3, 15)
        }

        # Generar historial de búsquedas
        for _ in range(random.randint(20, 60)):
            query = random.choice(self.query_list)
            user_data["search_history"].append({
                "query": query,
                "timestamp": (datetime.now() - timedelta(days=random.randint(1, 60))).isoformat(),
                "results_count": random.randint(3, 10),
                "clicked_products": [p["id"] for p in random.sample(self.amazon_products, random.randint(1, 3))],
                "session_duration": random.uniform(30, 300)
            })

        # Generar historial de feedback
        for _ in range(random.randint(10, 30)):
            query = random.choice(self.query_list)
            products_shown = random.sample(self.amazon_products, random.randint(3, 5))
            selected_product = random.choice(products_shown)
            
            user_data["feedback_history"].append({
                "query": query,
                "response": f"Recomendación para {query}",
                "rating": random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.15, 0.25, 0.3, 0.2])[0],
                "timestamp": (datetime.now() - timedelta(days=random.randint(1, 60))).isoformat(),
                "products_shown": [p["id"] for p in products_shown],
                "selected_product": selected_product["id"]
            })

        return user_data

    # ------------------------------------------------------------------------------------
    # SUCCESS_QUERIES.LOG (formato EXACTO requerido por FeedbackProcessor)
    # ------------------------------------------------------------------------------------
    def generate_success_queries_log(self):
        """Genera success_queries.log con formato EXACTO para FeedbackProcessor"""
        logger.info("📝 Generando success_queries.log...")
        
        success_file = Path("data/feedback/success_queries.log")
        existing_ids = set()
        
        with open(success_file, "w", encoding="utf-8") as f:
            for i in range(60):  # 60 consultas exitosas
                query = random.choice(self.query_list)
                product = random.choice(self.amazon_products)
                
                # Generar ID único como lo hace FeedbackProcessor
                query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
                entry_id = f"session_{i:03d}-{query_hash}"
                
                if entry_id not in existing_ids:
                    existing_ids.add(entry_id)
                    
                    record = {
                        "timestamp": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                        "session_id": f"session_{i:03d}",
                        "query": query,
                        "response": f"Te recomiendo {product['title']}. Es un excelente producto con calificación {product['average_rating']}/5 y precio ${product['price']}.",
                        "feedback": random.choice([4, 5]),
                        "selected_product_id": product["id"],
                        "source_file": "conversation_001.json",
                        "processed": False
                    }
                    
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.info(f"  ✅ {success_file} - {len(existing_ids)} registros")

    # ------------------------------------------------------------------------------------
    # FAILED_QUERIES.LOG (formato EXACTO requerido por FeedbackProcessor)
    # ------------------------------------------------------------------------------------
    def generate_failed_queries_log(self):
        """Genera failed_queries.log con formato EXACTO para FeedbackProcessor"""
        logger.info("📝 Generando failed_queries.log...")
        
        failed_file = Path("data/feedback/failed_queries.log")
        existing_ids = set()
        
        failure_reasons = ["product_not_found", "incomplete_data", "low_quality_response", "system_error"]
        
        with open(failed_file, "w", encoding="utf-8") as f:
            for i in range(40):  # 40 consultas fallidas
                query = random.choice(self.query_list)
                
                # Generar ID único
                query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
                entry_id = f"session_fail_{i:03d}-{query_hash}"
                
                if entry_id not in existing_ids:
                    existing_ids.add(entry_id)
                    
                    record = {
                        "timestamp": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                        "session_id": f"session_fail_{i:03d}",
                        "query": query,
                        "response": f"No pude encontrar productos específicos para '{query}'. Intenta con términos más generales.",
                        "feedback": random.choice([1, 2, 3]),
                        "failure_reason": random.choice(failure_reasons),
                        "source_file": "conversation_001.json",
                        "processed": False
                    }
                    
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.info(f"  ✅ {failed_file} - {len(existing_ids)} registros")

    # ------------------------------------------------------------------------------------
    # FEEDBACK EN TIEMPO REAL (formato para feedback_*.jsonl)
    # ------------------------------------------------------------------------------------
    def generate_realtime_feedback(self):
        """Genera archivos de feedback en tiempo real"""
        logger.info("📝 Generando feedback en tiempo real...")
        
        for day in range(7):  # Últimos 7 días
            date_str = (datetime.now() - timedelta(days=day)).strftime("%Y-%m-%d")
            feedback_file = Path(f"data/feedback/feedback_{date_str}.jsonl")
            
            with open(feedback_file, "w", encoding="utf-8") as f:
                for i in range(random.randint(5, 15)):
                    query = random.choice(self.query_list)
                    product = random.choice(self.amazon_products)
                    rating = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.15, 0.25, 0.3, 0.2])[0]
                    
                    record = {
                        "timestamp": (datetime.now() - timedelta(days=day, hours=random.randint(1, 23))).isoformat(),
                        "query": query,
                        "response": f"Recomendación: {product['title']} - ${product['price']}",
                        "feedback": rating,
                        "processed": False
                    }
                    
                    if rating >= 4:
                        record["selected_product_id"] = product["id"]
                    else:
                        record["failure_reason"] = random.choice(["product_not_found", "low_quality_response"])
                    
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            logger.info(f"  ✅ {feedback_file}")

    # ------------------------------------------------------------------------------------
    # CONVERSATION HISTORY (formato para conversation_*.json)
    # ------------------------------------------------------------------------------------
    def generate_conversation_history(self):
        """Genera historial de conversaciones en formato conversation_*.json"""
        logger.info("📝 Generando historial de conversaciones...")
        
        for file_num in range(1, 4):  # 3 archivos de conversación
            conversation_file = Path(f"data/processed/historial/conversation_{file_num:03d}.json")
            
            conversations = []
            for i in range(20):  # 20 conversaciones por archivo
                query = random.choice(self.query_list)
                products = random.sample(self.amazon_products, random.randint(2, 4))
                
                conversation = {
                    "timestamp": (datetime.now() - timedelta(days=random.randint(1, 60))).isoformat(),
                    "session_id": f"conv_{file_num:03d}_{i:03d}",
                    "query": query,
                    "response": f"Encontré {len(products)} productos para '{query}': " + 
                               ", ".join([p["title"] for p in products]),
                    "products_recommended": products,
                    "feedback": random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.15, 0.25, 0.3, 0.2])[0]
                }
                
                conversations.append(conversation)
            
            with open(conversation_file, "w", encoding="utf-8") as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)
            
            logger.info(f"  ✅ {conversation_file} - {len(conversations)} conversaciones")

    # ------------------------------------------------------------------------------------
    # MÉTRICAS RLHF (formato training_metrics.jsonl)
    # ------------------------------------------------------------------------------------
    def generate_rlhf_metrics(self):
        """Genera métricas de entrenamiento RLHF"""
        logger.info("📊 Generando métricas RLHF...")
        
        metrics_file = Path("data/feedback/rlhf_metrics/training_metrics.jsonl")
        base_accuracy = 0.65
        
        with open(metrics_file, "w", encoding="utf-8") as f:
            for i in range(25):  # 25 sesiones de entrenamiento
                improvement = random.uniform(-0.08, 0.12)
                new_accuracy = max(0.5, min(0.95, base_accuracy + improvement))
                
                record = {
                    "timestamp": (datetime.now() - timedelta(days=25-i)).isoformat(),
                    "examples_used": random.randint(50, 200),
                    "previous_accuracy": round(base_accuracy, 3),
                    "new_accuracy": round(new_accuracy, 3),
                    "improvement": round(improvement, 3),
                    "training_time_seconds": random.randint(300, 1800),
                    "success": improvement > 0
                }
                
                base_accuracy = new_accuracy
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        logger.info(f"  ✅ {metrics_file} - 25 sesiones de entrenamiento")

    # ------------------------------------------------------------------------------------
    # DATOS DE PRODUCTOS (raw y processed)
    # ------------------------------------------------------------------------------------
    def generate_product_data(self):
        """Genera datos de productos en formato raw y processed"""
        logger.info("📦 Generando datos de productos...")
        
        # Raw data (JSONL)
        raw_file = Path("data/raw/amazon_products.jsonl")
        with open(raw_file, "w", encoding="utf-8") as f:
            for product in self.amazon_products:
                # Añadir más campos para hacerlo más realista
                enhanced_product = {
                    **product,
                    "categories": [product["main_category"], "electronics", "gaming"],
                    "rating_count": random.randint(50, 1000),
                    "tags": ["gaming", "electronics", "amazon_choice"],
                    "details": {
                        "features": [f"Feature {i+1}" for i in range(random.randint(2, 5))],
                        "specifications": {
                            "brand": random.choice(["Sony", "Microsoft", "Nintendo", "Logitech", "Razer"]),
                            "color": random.choice(["black", "white", "red", "blue"]),
                            "weight": f"{random.randint(1, 5)} kg"
                        }
                    },
                    "compatible_devices": random.choice([["PC", "PS5", "Xbox"], ["Nintendo Switch"], ["PC"], ["PS5"]])
                }
                f.write(json.dumps(enhanced_product, ensure_ascii=False) + "\n")
        
        logger.info(f"  ✅ {raw_file} - {len(self.amazon_products)} productos")
        
        # Processed data (formato para DataLoader)
        processed_file = Path("data/processed/products.json")
        with open(processed_file, "w", encoding="utf-8") as f:
            json.dump(self.amazon_products, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  ✅ {processed_file}")

    # ------------------------------------------------------------------------------------
    # EJECUCIÓN COMPLETA
    # ------------------------------------------------------------------------------------
    def run(self):
        """Ejecuta la generación completa de datos"""
        logger.info("🚀 INICIANDO GENERACIÓN COMPLETA DE DATOS")
        logger.info("=" * 70)
        
        # 1. Estructura de directorios
        self.setup_directories()
        
        # 2. Datos de productos
        self.generate_product_data()
        
        # 3. Perfiles de usuarios
        logger.info("👥 Generando perfiles de usuarios...")
        for i in range(1, self.num_users + 1):
            user_profile = self.generate_user_profile(i)
            user_file = Path(f"data/users/{user_profile['user_id']}.json")
            with open(user_file, "w", encoding="utf-8") as f:
                json.dump(user_profile, f, ensure_ascii=False, indent=2)
            logger.info(f"  ✅ {user_file}")
        
        # 4. Sistema de feedback
        self.generate_success_queries_log()
        self.generate_failed_queries_log()
        self.generate_realtime_feedback()
        
        # 5. Historial y métricas
        self.generate_conversation_history()
        self.generate_rlhf_metrics()
        
        # 6. Resumen final
        logger.info("=" * 70)
        logger.info("🎉 GENERACIÓN COMPLETA FINALIZADA!")
        logger.info("📊 RESUMEN DE DATOS CREADOS:")
        logger.info(f"   👥 {self.num_users} perfiles de usuario")
        logger.info(f"   📦 {len(self.amazon_products)} productos Amazon")
        logger.info(f"   ✅ success_queries.log - 60 consultas exitosas") 
        logger.info(f"   ❌ failed_queries.log - 40 consultas fallidas")
        logger.info(f"   💬 3 archivos de conversación histórica")
        logger.info(f"   📊 25 sesiones de métricas RLHF")
        logger.info(f"   🔄 7 días de feedback en tiempo real")
        logger.info("=" * 70)
        logger.info("💡 El sistema RAG + RL está listo para usar!")


if __name__ == "__main__":
    generator = CompleteTrainingDataGenerator(num_users=12, searches_per_user=50)
    generator.run()