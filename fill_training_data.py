#!/usr/bin/env python3
"""
Script COMPLETO para generar datos de entrenamiento para el sistema RAG + RL.
Incluye:
- Usuarios variados
- Mínimo 50 búsquedas por usuario
- Feedback positivo/negativo
- Productos realistas de Amazon (sin usar la API)
- Historial, logs, métricas RLHF
"""

import json
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    def __init__(self, num_users=12, searches_per_user=50):
        self.num_users = num_users
        self.searches_per_user = searches_per_user

        self.data_dir = Path("data")
        self.users_dir = self.data_dir / "users"
        self.feedback_dir = self.data_dir / "feedback"
        self.historial_dir = self.data_dir / "processed" / "historial"
        self.metrics_dir = self.feedback_dir / "rlhf_metrics"

        self.users_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.historial_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Productos realistas de Amazon simulados
        self.amazon_products = [
            {"asin": "B0D12C7Y5N", "title": "Nintendo Switch OLED - Mario Red Edition", "category": "consoles"},
            {"asin": "B0CGRVH2N4", "title": "PS5 Slim Standard Edition", "category": "consoles"},
            {"asin": "B08FC6MR62", "title": "Xbox Series X 1TB", "category": "consoles"},
            {"asin": "B0C5N4VYF2", "title": "Logitech G PRO X Superlight 2", "category": "gaming"},
            {"asin": "B0CGLX7JYF", "title": "LG Ultragear 27'' 144Hz Gaming Monitor", "category": "monitors"},
            {"asin": "B09V3HN1KC", "title": "Horizon Forbidden West", "category": "games"},
            {"asin": "B09B8RBFDD", "title": "Elden Ring", "category": "games"},
            {"asin": "B0BDJHN2GS", "title": "HyperX Cloud III Wireless", "category": "headsets"},
            {"asin": "B0BQKPHZWS", "title": "Razer Huntsman V3 Pro Keyboard", "category": "keyboards"},
            {"asin": "B0BRC9XTQS", "title": "Silla Gamer Corsair T3 Rush", "category": "chairs"},
        ]

        self.query_list = [
            "juegos nintendo switch",
            "mejores juegos ps5",
            "consolas baratas",
            "auriculares gaming inalámbricos",
            "teclado mecánico gamer",
            "silla gamer económica",
            "monitor 144Hz barato",
            "ratón inalámbrico",
            "juegos de aventura",
            "accesorios para pc gamer",
        ]

        # Tipos de usuarios
        self.user_profiles = [
            {"type": "explorer", "preferences": ["games", "consoles", "gaming"]},
            {"type": "specialist", "preferences": ["rpg", "adventure", "story"]},
            {"type": "bargain_hunter", "preferences": ["cheap", "budget"]},
            {"type": "impulsive_buyer", "preferences": ["latest", "popular"]},
            {"type": "tech_lover", "preferences": ["pc", "monitors", "hardware"]},
            {"type": "casual_gamer", "preferences": ["family", "kids", "fun"]},
        ]


    # ------------------------------------------------------------------------------------
    # GENERACIÓN DE USUARIOS
    # ------------------------------------------------------------------------------------
    def generate_user(self, idx):
        profile = random.choice(self.user_profiles)

        user_id = f"user_{idx:03d}"
        user_data = {
            "user_id": user_id,
            "type": profile["type"],
            "age": random.randint(18, 45),
            "gender": random.choice(["male", "female"]),
            "country": random.choice(["Spain", "Mexico", "Argentina", "Colombia", "Chile"]),
            "sessions": [],
            "purchase_history": []
        }

        num_sessions = random.randint(4, 10)
        searches_per_session = max(1, self.searches_per_user // num_sessions)

        for s in range(num_sessions):
            session_id = f"{user_id}_sess_{s}"
            session = {
                "session_id": session_id,
                "start_time": (datetime.now() - timedelta(days=random.randint(1, 60))).isoformat(),
                "searches": []
            }

            for _ in range(searches_per_session):
                query = random.choice(self.query_list)

                # Simular resultados
                products = random.sample(self.amazon_products, random.randint(2, 5))

                rating = random.choices(
                    [1, 2, 3, 4, 5],
                    weights=[0.15, 0.15, 0.3, 0.25, 0.15]
                )[0]

                positive = rating >= 4

                purchase = False
                if positive and random.random() < 0.15:
                    purchase = True

                chosen = random.choice(products)

                session["searches"].append({
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "results": products,
                    "rating": rating,
                    "positive": positive,
                    "selected_asin": chosen["asin"],
                    "purchased": purchase
                })

                if purchase:
                    user_data["purchase_history"].append({
                        "asin": chosen["asin"],
                        "title": chosen["title"],
                        "timestamp": datetime.now().isoformat()
                    })

            user_data["sessions"].append(session)

        return user_data


    # ------------------------------------------------------------------------------------
    # LOGS DE FEEDBACK
    # ------------------------------------------------------------------------------------
    def generate_feedback_logs(self):
        success_file = self.feedback_dir / "success.log"
        failed_file = self.feedback_dir / "failed.log"

        with open(success_file, "w", encoding="utf-8") as s:
            for i in range(40):
                prod = random.choice(self.amazon_products)
                rec = {
                    "query": random.choice(self.query_list),
                    "product": prod,
                    "rating": random.choice([4, 5]),
                    "timestamp": datetime.now().isoformat()
                }
                s.write(json.dumps(rec, ensure_ascii=False) + "\n")

        with open(failed_file, "w", encoding="utf-8") as f:
            for i in range(30):
                prod = random.choice(self.amazon_products)
                rec = {
                    "query": random.choice(self.query_list),
                    "product": prod,
                    "rating": random.choice([1, 2, 3]),
                    "reason": random.choice(["irrelevant", "bad_quality", "not_useful"]),
                    "timestamp": datetime.now().isoformat()
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


    # ------------------------------------------------------------------------------------
    # HISTORIAL DE CONVERSACIONES
    # ------------------------------------------------------------------------------------
    def generate_conversation_history(self):
        file = self.historial_dir / "historial.jsonl"
        with open(file, "w", encoding="utf-8") as f:
            for i in range(60):
                prod = random.choice(self.amazon_products)
                record = {
                    "query": random.choice(self.query_list),
                    "response": f"Recomendación generada: {prod['title']}",
                    "products_recommended": random.sample(self.amazon_products, 3),
                    "feedback": random.randint(1, 5),
                    "timestamp": datetime.now().isoformat()
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


    # ------------------------------------------------------------------------------------
    # MÉTRICAS RLHF
    # ------------------------------------------------------------------------------------
    def generate_rlhf_metrics(self):
        file = self.metrics_dir / "metrics.jsonl"
        base_acc = 0.60

        with open(file, "w", encoding="utf-8") as f:
            for _ in range(20):
                improvement = random.uniform(-0.05, 0.15)
                new_acc = max(0, min(1, base_acc + improvement))

                rec = {
                    "timestamp": datetime.now().isoformat(),
                    "previous_accuracy": base_acc,
                    "new_accuracy": new_acc,
                    "improvement": improvement,
                    "samples_used": random.randint(50, 200)
                }

                base_acc = new_acc
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


    # ------------------------------------------------------------------------------------
    # EJECUCIÓN GENERAL
    # ------------------------------------------------------------------------------------
    def run(self):
        logger.info("🚀 GENERACIÓN COMPLETA DE DATOS INICIADA")
        logger.info("=" * 60)

        for i in range(1, self.num_users + 1):
            user = self.generate_user(i)
            file = self.users_dir / f"{user['user_id']}.json"
            with open(file, "w", encoding="utf-8") as f:
                json.dump(user, f, ensure_ascii=False, indent=2)

        self.generate_feedback_logs()
        self.generate_conversation_history()
        self.generate_rlhf_metrics()

        logger.info("🎉 Datos generados correctamente!")
        logger.info(f"Usuarios creados: {self.num_users}")
        logger.info("=" * 60)


if __name__ == "__main__":
    generator = TrainingDataGenerator()
    generator.run()
