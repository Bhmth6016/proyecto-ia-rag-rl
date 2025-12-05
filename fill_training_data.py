#!/usr/bin/env python3
"""
Script ADAPTADO para generar datos específicos de videojuegos en los formatos requeridos
MEJORADO con: modularización, type hints, validaciones y mejoras sugeridas
"""

import json
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ==================== CONFIGURACIÓN EXTERNA ====================
# Estos datos podrían cargarse desde JSON/YAML externo

PRODUCTS_CONFIG = [
    {
        "id": "B08N5WRWNW",
        "title": "Logitech MX Master 3 - Ratón Inalámbrico",
        "description": "Ratón ergonómico con scroll horizontal, 4000 DPI, batería 70 días, Bluetooth/Unifying",
        "main_category": "Periféricos Gaming",
        "categories": ["Ratones", "Accesorios PC", "Oficina", "Gaming"],
        "price": 99.99,
        "average_rating": 4.7,
        "rating_count": 2345,
        "features": ["Ergonómico", "Batería larga duración", "Scroll horizontal", "Multi-dispositivo"],
        "brand": "Logitech",
        "attributes": {"color": "Negro", "conectividad": "Bluetooth/2.4GHz", "DPI": "4000", "bateria_dias": "70"}
    },
    {
        "id": "B07S92QBCJ",
        "title": "Razer DeathAdder V2 - Ratón Gaming",
        "description": "Ratón gaming con sensor óptico 20K DPI, 8 botones programables",
        "main_category": "Periféricos Gaming",
        "categories": ["Ratones Gaming", "Accesorios PC", "Gaming"],
        "price": 69.99,
        "average_rating": 4.6,
        "rating_count": 1890,
        "features": ["20K DPI", "8 botones", "Iluminación RGB", "Cable Speedflex"],
        "brand": "Razer",
        "attributes": {"color": "Negro", "DPI": "20000", "botones": "8", "peso": "82g"}
    },
    {
        "id": "B09V3JN27K",
        "title": "SteelSeries Apex Pro - Teclado Mecánico",
        "description": "Teclado mecánico gaming con switches ajustables y pantalla OLED",
        "main_category": "Periféricos Gaming",
        "categories": ["Teclados", "Gaming", "Accesorios PC"],
        "price": 199.99,
        "average_rating": 4.8,
        "rating_count": 1567,
        "features": ["Switches ajustables", "Pantalla OLED", "Iluminación RGB", "Reposamuñecas magnético"],
        "brand": "SteelSeries",
        "attributes": {"switch": "OmniPoint", "layout": "US", "conexion": "USB", "teclas": "104"}
    },
    {
        "id": "B08FC5L3RG",
        "title": "PlayStation 5 - Consola Standard",
        "description": "Consola de videojuegos de última generación con SSD ultra rápido",
        "main_category": "Consolas",
        "categories": ["Consolas", "Gaming", "Electrónica"],
        "price": 499.99,
        "average_rating": 4.9,
        "rating_count": 4500,
        "features": ["SSD 825GB", "Ray Tracing", "4K 120Hz", "Compatibilidad con PS4"],
        "brand": "Sony",
        "attributes": {"almacenamiento": "825GB", "resolucion": "8K", "hdr": "Sí", "puertos_usb": "3"}
    },
    {
        "id": "B08H93ZRK9",
        "title": "Xbox Series X - Consola 1TB",
        "description": "Consola más potente de Xbox con Quick Resume y Game Pass",
        "main_category": "Consolas",
        "categories": ["Consolas", "Gaming", "Electrónica"],
        "price": 479.99,
        "average_rating": 4.8,
        "rating_count": 3800,
        "features": ["1TB SSD", "Quick Resume", "4K 120fps", "Game Pass incluido"],
        "brand": "Microsoft",
        "attributes": {"almacenamiento": "1TB", "resolucion": "4K", "fps": "120", "backward": "Sí"}
    },
    {
        "id": "B0BN1ZKJ7D",
        "title": "The Legend of Zelda: Tears of the Kingdom",
        "description": "Juego de aventura y acción para Nintendo Switch",
        "main_category": "Videojuegos",
        "categories": ["Juegos Switch", "Aventura", "Acción"],
        "price": 69.99,
        "average_rating": 4.9,
        "rating_count": 5200,
        "features": ["Mundo abierto", "Multijugador", "Aventura épica"],
        "brand": "Nintendo",
        "attributes": {"plataforma": "Nintendo Switch", "genero": "Aventura", "online": "Sí", "idiomas": "ES"}
    },
    {
        "id": "B09B8RBFDD",
        "title": "Elden Ring - Edición Estándar",
        "description": "Juego de rol y acción de mundo abierto por FromSoftware",
        "main_category": "Videojuegos",
        "categories": ["Juegos PS5", "RPG", "Acción"],
        "price": 59.99,
        "average_rating": 4.7,
        "rating_count": 8900,
        "features": ["Mundo abierto", "Combate souls-like", "Multijugador online"],
        "brand": "Bandai Namco",
        "attributes": {"plataforma": "PS5/Xbox/PC", "genero": "RPG", "online": "Sí", "modo": "Un jugador/Multijugador"}
    },
    {
        "id": "B0C5N4VYF2",
        "title": "Logitech G PRO X Superlight 2",
        "description": "Ratón gaming ultra ligero para competición profesional",
        "main_category": "Periféricos Gaming",
        "categories": ["Ratones Gaming", "eSports", "Accesorios PC"],
        "price": 159.99,
        "average_rating": 4.6,
        "rating_count": 1200,
        "features": ["60g peso", "Sensor HERO 25K", "Lightspeed wireless", "Powerplay compatible"],
        "brand": "Logitech",
        "attributes": {"peso": "60g", "DPI": "25000", "conexion": "Inalámbrico", "bateria": "70h"}
    }
]

USER_PROFILES_CONFIG = [
    {
        "type": "hardcore_gamer",
        "preferred_categories": ["Periféricos Gaming", "Consolas", "Videojuegos"],
        "preferred_brands": ["Logitech", "Razer", "SteelSeries", "Sony"],
        "price_range": {"min": 50, "max": 500},
        "age_range": (18, 35),
        "professions": ["programador", "estudiante", "diseñador", "streamer"]
    },
    {
        "type": "casual_gamer",
        "preferred_categories": ["Videojuegos", "Consolas"],
        "preferred_brands": ["Nintendo", "Sony", "Microsoft"],
        "price_range": {"min": 30, "max": 300},
        "age_range": (25, 50),
        "professions": ["profesor", "administrativo", "médico", "ingeniero"]
    },
    {
        "type": "competitive_gamer",
        "preferred_categories": ["Periféricos Gaming", "eSports"],
        "preferred_brands": ["Logitech", "Razer", "SteelSeries", "Corsair"],
        "price_range": {"min": 100, "max": 1000},
        "age_range": (16, 30),
        "professions": ["estudiante", "profesional esports", "streamer", "coach"]
    }
]

QUERIES_CONFIG = [
    "ratón ergonómico para programar",
    "mejor ratón para juegos FPS",
    "teclado mecánico gaming barato",
    "consola ps5 vs xbox series x",
    "juegos nintendo switch para niños",
    "auriculares gaming con buen micrófono",
    "silla gaming ergonómica",
    "monitor 240hz gaming",
    "mejor juego mundo abierto 2024",
    "ratón inalámbrico para trabajo"
]

# ==================== FUNCIONES UTILITARIAS ====================

def write_jsonl(path: Path, data: List[Dict], append: bool = False) -> None:
    """
    Escribe datos en formato JSONL con manejo de errores
    
    Args:
        path: Ruta del archivo
        data: Lista de diccionarios a escribir
        append: Si True, añade al archivo existente
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        
        with path.open(mode, encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        
        logger.info(f"  ✅ {path} - {len(data)} registros escritos")
    except Exception as e:
        logger.error(f"  ❌ Error al escribir {path}: {e}")
        raise

def read_jsonl(path: Path) -> List[Dict]:
    """
    Lee datos desde un archivo JSONL
    
    Args:
        path: Ruta del archivo
    
    Returns:
        Lista de diccionarios con los datos
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except Exception as e:
        logger.error(f"Error al leer {path}: {e}")
        return []


class GamingDataGenerator:
    def __init__(self, num_users: int = 10, interactions_per_user: int = 20):
        self.num_users = num_users
        self.interactions_per_user = interactions_per_user
        
        # Configuración desde constantes externas
        self.gaming_products = PRODUCTS_CONFIG
        self.user_profiles = USER_PROFILES_CONFIG
        self.gaming_queries = QUERIES_CONFIG
        
        # Estructura de directorios
        self.setup_directories()

    def setup_directories(self) -> None:
        """Crea la estructura de directorios requerida"""
        directories = [
            "data/raw/interactions",
            "data/raw/conversations",
            "data/raw/feedback",
            "data/raw/product_similarity",
            "data/raw/user_preferences",
            "data/processed"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✅ Directorio creado: {dir_path}")

    def ts(self, days_range: int = 30) -> str:
        """
        Genera un timestamp aleatorio dentro del rango especificado
        
        Args:
            days_range: Número máximo de días hacia atrás
        
        Returns:
            Timestamp en formato ISO con Z
        """
        return (datetime.now() - timedelta(days=random.randint(0, days_range))).isoformat() + "Z"

    def random_user_profile(self) -> Dict[str, Any]:
        """Selecciona aleatoriamente un perfil de usuario"""
        return random.choice(self.user_profiles)

    def sample_products(self, k_range: Tuple[int, int] = (3, 5)) -> List[Dict[str, Any]]:
        """
        Muestrea productos aleatoriamente
        
        Args:
            k_range: Rango (mín, máx) de productos a muestrear
        
        Returns:
            Lista de productos seleccionados
        """
        k = random.randint(*k_range)
        return random.sample(self.gaming_products, k=min(k, len(self.gaming_products)))

    def product_similarity(self, product_a: Dict[str, Any], product_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula la similitud entre dos productos
        
        Args:
            product_a: Primer producto
            product_b: Segundo producto
        
        Returns:
            Diccionario con score de similitud y características compartidas
        """
        similarity_score = 0.0
        similarity_type = ""
        shared_features = []
        
        # Misma categoría principal
        if product_a["main_category"] == product_b["main_category"]:
            similarity_score += 0.3
            similarity_type = "category"
            shared_features.append(product_a["main_category"])
        
        # Misma marca
        if product_a["brand"] == product_b["brand"]:
            similarity_score += 0.4
            similarity_type = "brand"
            shared_features.append(product_a["brand"])
        
        # Categorías similares
        common_cats = set(product_a["categories"]) & set(product_b["categories"])
        if common_cats:
            similarity_score += len(common_cats) * 0.1
            if not similarity_type:
                similarity_type = "categories"
            shared_features.extend(list(common_cats)[:2])
        
        # Precio similar (±20%)
        if product_a["price"] > 0 and product_b["price"] > 0:
            price_ratio = min(product_a["price"], product_b["price"]) / max(product_a["price"], product_b["price"])
            if price_ratio > 0.8:
                similarity_score += 0.2
                if not similarity_type:
                    similarity_type = "price_range"
                shared_features.append("precio similar")
        
        # Asegurar score entre 0 y 1
        similarity_score = min(1.0, similarity_score)
        
        return {
            "similarity_score": round(similarity_score, 2),
            "similarity_type": similarity_type,
            "shared_features": list(set(shared_features))[:3]
        }

    def generate_user_interactions(self) -> List[Dict[str, Any]]:
        """Genera datos de interacción usuario-producto"""
        logger.info("🔄 Generando interacciones usuario-producto...")
        
        interactions = []
        total_interactions = 0
        
        for user_num in range(self.num_users):
            profile = self.random_user_profile()
            user_id = f"user_{user_num:03d}"
            
            user_interactions = random.randint(5, self.interactions_per_user)
            total_interactions += user_interactions
            
            for interaction_num in range(user_interactions):
                session_id = f"sess_{datetime.now().strftime('%Y%m%d')}_{user_num:03d}_{interaction_num:03d}"
                query = random.choice(self.gaming_queries)
                
                # Seleccionar productos vistos
                viewed_products = self.sample_products((3, 5))
                viewed_ids = [p["id"] for p in viewed_products]
                
                # Producto clickeado
                clicked_product = random.choice(viewed_products)
                
                # Determinar si compró
                dwell_time = random.randint(10, 180)
                purchased = random.random() < (dwell_time / 200)
                
                # Generar rating si compró
                rating = None
                feedback_text = None
                if purchased:
                    rating = random.choices([4, 5], weights=[0.3, 0.7])[0]
                    feedback_options = [
                        "Muy cómodo para largas jornadas",
                        "Excelente para gaming",
                        "Buen relación calidad-precio",
                        "Superó mis expectativas",
                        "Lo recomendaría a otros gamers"
                    ]
                    feedback_text = random.choice(feedback_options)
                
                interaction = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "query": query,
                    "timestamp": self.ts(30),
                    "viewed_products": viewed_ids,
                    "clicked_product": clicked_product["id"],
                    "dwell_time_seconds": dwell_time,
                    "purchased": purchased,
                    "rating": rating,
                    "feedback_text": feedback_text
                }
                
                interactions.append(interaction)
        
        # Guardar a archivo
        interactions_file = Path("data/raw/interactions/user_interactions.jsonl")
        write_jsonl(interactions_file, interactions)
        
        logger.info(f"  📊 Total interacciones generadas: {total_interactions}")
        return interactions

    def generate_conversations(self) -> List[Dict[str, Any]]:
        """Genera datos de conversaciones para RAG/RLHF"""
        logger.info("💬 Generando conversaciones para RAG/RLHF...")
        
        conversation_templates = [
            {
                "user_intent": "recomendación producto gaming",
                "user_messages": [
                    "Necesito un ratón cómodo para trabajar todo el día",
                    "¿Funciona con Mac?",
                    "¿Tiene garantía?"
                ],
                "assistant_responses": [
                    "Te recomiendo el Logitech MX Master 3. Es ergonómico, tiene batería de 70 días y scroll horizontal para productividad.",
                    "Sí, es compatible con Mac, Windows y Linux.",
                    "Tiene garantía de 2 años del fabricante."
                ]
            },
            {
                "user_intent": "comparación consolas",
                "user_messages": [
                    "¿Qué consola me recomiendas, PS5 o Xbox Series X?",
                    "¿Cuál tiene mejores exclusivos?",
                    "¿Y para juego en familia?"
                ],
                "assistant_responses": [
                    "Depende de tus preferencias. PS5 tiene exclusivos como Spider-Man 2, Xbox incluye Game Pass.",
                    "PS5 tiene más exclusivos de acción/aventura, Xbox tiene mejor servicio por suscripción.",
                    "Para familia, Nintendo Switch es mejor opción con juegos como Mario Kart."
                ]
            },
            {
                "user_intent": "selección periféricos",
                "user_messages": [
                    "Busco un teclado mecánico para gaming",
                    "¿Qué switches son mejores?",
                    "¿Recomiendas algún modelo específico?"
                ],
                "assistant_responses": [
                    "Te recomiendo el SteelSeries Apex Pro con switches ajustables.",
                    "Los switches lineales (rojos) son buenos para gaming, los táctiles (marrón) para mixto.",
                    "El SteelSeries Apex Pro o el Razer Huntsman V2 son excelentes opciones."
                ]
            }
        ]
        
        conversations = []
        
        for conv_num in range(50):
            template = random.choice(conversation_templates)
            profile = self.random_user_profile()
            
            # Crear conversación con múltiples turnos
            turns = []
            turn_count = random.randint(2, 4)
            
            for i in range(turn_count):
                if i == 0:
                    # Primer turno: usuario
                    turns.append({
                        "role": "user",
                        "content": template["user_messages"][0],
                        "timestamp": self.ts(30)
                    })
                    
                    # Respuesta del asistente
                    recommended = random.sample(self.gaming_products, k=random.randint(1, 3))
                    turns.append({
                        "role": "assistant",
                        "content": template["assistant_responses"][0],
                        "recommended_products": [p["id"] for p in recommended],
                        "timestamp": self.ts(30)
                    })
                elif i < len(template["user_messages"]):
                    # Turnos adicionales
                    turns.append({
                        "role": "user",
                        "content": template["user_messages"][i],
                        "timestamp": self.ts(30)
                    })
            
            conversation = {
                "conversation_id": f"conv_{conv_num:03d}",
                "turns": turns,
                "successful": random.random() > 0.2,
                "final_rating": random.choices([3, 4, 5], weights=[0.1, 0.3, 0.6])[0] if random.random() > 0.2 else random.randint(1, 2),
                "user_profile": {
                    "age": random.randint(*profile["age_range"]),
                    "profession": random.choice(profile["professions"]),
                    "use_case": random.choice(["trabajo", "gaming", "estudio", "mixto"])
                }
            }
            
            conversations.append(conversation)
        
        # Guardar a archivo
        conversations_file = Path("data/raw/conversations/rag_training_data.jsonl")
        write_jsonl(conversations_file, conversations)
        
        logger.info(f"  📊 Total conversaciones generadas: {len(conversations)}")
        return conversations

    def generate_feedback_rlhf(self) -> List[Dict[str, Any]]:
        """Genera datos de feedback para RLHF"""
        logger.info("📝 Generando feedback para RLHF...")
        
        queries_feedback = [
            {
                "query": "mejor ratón para juegos",
                "model_response": "El Razer DeathAdder V2 es excelente para gaming con 20K DPI y diseño ergonómico.",
                "human_response": "Sí, pero también considera el Logitech G Pro X Superlight 2 para juegos competitivos.",
                "rating": 3,
                "preferred_response": "human",
                "reasons": ["más específico", "menciona alternativa", "contexto competitivo"]
            },
            {
                "query": "consola para niños",
                "model_response": "La PlayStation 5 tiene muchos juegos familiares.",
                "human_response": "Para niños, la Nintendo Switch es mejor por sus controles intuitivos y juegos como Mario.",
                "rating": 2,
                "preferred_response": "human",
                "reasons": ["más adecuado para niños", "ejemplos concretos", "considera usabilidad"]
            },
            {
                "query": "teclado gaming barato",
                "model_response": "Cualquier teclado mecánico con switches red.",
                "human_response": "El Redragon K552 ofrece buena calidad-precio con switches mecánicos y construcción duradera.",
                "rating": 4,
                "preferred_response": "human",
                "reasons": ["recomendación específica", "menciona marca concreta", "justifica la elección"]
            }
        ]
        
        feedback_list = []
        
        for i in range(40):
            base_feedback = random.choice(queries_feedback)
            
            # Variar ligeramente cada ejemplo
            feedback = {
                "query": base_feedback["query"],
                "model_response": base_feedback["model_response"],
                "human_response": base_feedback["human_response"],
                "rating": max(1, min(5, base_feedback["rating"] + random.randint(-1, 1))),
                "preferred_response": base_feedback["preferred_response"],
                "reasons": base_feedback["reasons"],
                "domain": "gaming",
                "expertise_level": random.choice(["principiante", "intermedio", "avanzado"]),
                "timestamp": self.ts(60)
            }
            
            feedback_list.append(feedback)
        
        # Guardar a archivo
        feedback_file = Path("data/raw/feedback/rlhf_training.jsonl")
        write_jsonl(feedback_file, feedback_list)
        
        logger.info(f"  📊 Total ejemplos de feedback generados: {len(feedback_list)}")
        return feedback_list

    def generate_product_similarity(self) -> Dict[str, Any]:
        """Genera datos de similitud entre productos"""
        logger.info("🔗 Generando similitud de productos...")
        
        product_pairs = []
        
        for i in range(len(self.gaming_products)):
            for j in range(i + 1, len(self.gaming_products)):
                product_a = self.gaming_products[i]
                product_b = self.gaming_products[j]
                
                # Calcular similitud
                similarity_data = self.product_similarity(product_a, product_b)
                
                if similarity_data["similarity_score"] > 0.3:
                    product_pairs.append({
                        "product_a": product_a["id"],
                        "product_b": product_b["id"],
                        **similarity_data
                    })
        
        # Limitar a 20 pares más relevantes
        product_pairs.sort(key=lambda x: x["similarity_score"], reverse=True)
        selected_pairs = product_pairs[:20]
        
        # Guardar a archivo
        similarity_data = {"product_pairs": selected_pairs}
        similarity_file = Path("data/raw/product_similarity/product_similarities.json")
        
        try:
            similarity_file.parent.mkdir(parents=True, exist_ok=True)
            with similarity_file.open("w", encoding="utf-8") as f:
                json.dump(similarity_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"  ✅ {similarity_file} - {len(selected_pairs)} pares de similitud generados")
        except Exception as e:
            logger.error(f"  ❌ Error al escribir {similarity_file}: {e}")
        
        return similarity_data

    def generate_user_preferences(self) -> List[Dict[str, Any]]:
        """Genera datos de preferencias de usuario"""
        logger.info("👤 Generando preferencias de usuario...")
        
        preferences_list = []
        
        for user_num in range(self.num_users):
            profile = self.random_user_profile()
            user_id = f"user_{user_num:03d}"
            
            # Generar interacciones históricas
            historical_interactions = []
            for _ in range(random.randint(3, 10)):
                product = random.choice(self.gaming_products)
                action = random.choices(["view", "click", "purchase"], weights=[0.5, 0.3, 0.2])[0]
                
                interaction = {
                    "product_id": product["id"],
                    "action": action,
                    "timestamp": (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat()
                }
                
                if action == "purchase":
                    interaction["rating"] = random.choices([4, 5], weights=[0.3, 0.7])[0]
                elif action == "view":
                    interaction["dwell_time"] = random.randint(5, 120)
                
                historical_interactions.append(interaction)
            
            user_preferences = {
                "user_id": user_id,
                "preferred_categories": profile["preferred_categories"],
                "preferred_brands": profile["preferred_brands"],
                "price_range": profile["price_range"],
                "avoided_categories": random.sample(["Ropa", "Alimentación", "Muebles", "Libros"], k=random.randint(0, 2)),
                "historical_interactions": historical_interactions,
                "demographics": {
                    "age": random.randint(*profile["age_range"]),
                    "gender": random.choice(["male", "female", "other"]),
                    "country": random.choice(["Spain", "Mexico", "Argentina", "Colombia", "Chile"]),
                    "language": "es"
                },
                "gaming_profile": {
                    "experience_years": random.randint(1, 20),
                    "preferred_genres": random.sample(["FPS", "RPG", "Estrategia", "Deportes", "Aventura"], k=random.randint(2, 4)),
                    "play_time_weekly": random.choice(["5-10h", "10-20h", "20-40h", "40+h"]),
                    "gaming_platforms": random.sample(["PC", "PlayStation", "Xbox", "Nintendo Switch", "Mobile"], k=random.randint(1, 3))
                }
            }
            
            preferences_list.append(user_preferences)
        
        # Guardar a archivo
        preferences_file = Path("data/raw/user_preferences/user_preferences.jsonl")
        write_jsonl(preferences_file, preferences_list)
        
        logger.info(f"  📊 Total preferencias generadas: {len(preferences_list)}")
        return preferences_list

    def save_products(self) -> None:
        """Guarda los productos en formato JSONL"""
        products_file = Path("data/raw/products.jsonl")
        write_jsonl(products_file, self.gaming_products)
        logger.info(f"  📊 Total productos guardados: {len(self.gaming_products)}")

    def run_parallel(self) -> None:
        """Ejecuta la generación en paralelo"""
        logger.info("⚡ Ejecutando generación paralela...")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.generate_user_interactions): "interacciones",
                executor.submit(self.generate_conversations): "conversaciones",
                executor.submit(self.generate_feedback_rlhf): "feedback",
                executor.submit(self.generate_product_similarity): "similitud",
                executor.submit(self.generate_user_preferences): "preferencias"
            }
            
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    future.result()
                    logger.info(f"  ✅ {task_name} completado")
                except Exception as e:
                    logger.error(f"  ❌ Error en {task_name}: {e}")
        
        # Guardar productos (pequeño, no necesita paralelismo)
        self.save_products()

    def run_sequential(self) -> None:
        """Ejecuta la generación secuencial"""
        self.generate_user_interactions()
        self.generate_conversations()
        self.generate_feedback_rlhf()
        self.generate_product_similarity()
        self.generate_user_preferences()
        self.save_products()

    def run(self, parallel: bool = False) -> None:
        """Ejecuta la generación completa de datos"""
        logger.info("🚀 INICIANDO GENERACIÓN DE DATOS DE VIDEOJUEGOS")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        
        if parallel:
            self.run_parallel()
        else:
            self.run_sequential()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("=" * 70)
        logger.info(f"🎉 GENERACIÓN COMPLETA FINALIZADA en {duration:.2f} segundos!")
        logger.info("📊 DATOS GENERADOS EN:")
        logger.info("   📁 data/raw/interactions/user_interactions.jsonl")
        logger.info("   📁 data/raw/conversations/rag_training_data.jsonl")
        logger.info("   📁 data/raw/feedback/rlhf_training.jsonl")
        logger.info("   📁 data/raw/product_similarity/product_similarities.json")
        logger.info("   📁 data/raw/user_preferences/user_preferences.jsonl")
        logger.info("   📁 data/raw/products.jsonl")
        logger.info("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generador de datos de videojuegos")
    parser.add_argument("--users", type=int, default=15, help="Número de usuarios")
    parser.add_argument("--interactions", type=int, default=25, help="Interacciones por usuario")
    parser.add_argument("--parallel", action="store_true", help="Ejecutar en paralelo")
    
    args = parser.parse_args()
    
    generator = GamingDataGenerator(num_users=args.users, interactions_per_user=args.interactions)
    generator.run(parallel=args.parallel)