#!/usr/bin/env python3
"""
deepeval_custom.py - Sistema de evaluación ajustado para datos simulados de usuarios
"""
import json
import time
import random
import logging
import numpy as np
from typing import List, Set, Dict, Any, Tuple

# --- Configuración logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CustomEvaluationSystem:
    """Sistema de evaluación personalizado para datos simulados de usuarios."""
    
    def __init__(self, seed=42, n_products=20, n_users=100):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Configuración del escenario
        self.n_products = n_products
        self.n_users = n_users
        
        # Generar datos simulados
        self.products = self._generate_products()
        self.users = self._generate_users()
        self.conversations = self._generate_conversations()
        self.feedbacks = self._generate_feedbacks()
        
        # Ground truth basado en interacciones reales simuladas
        self.ground_truth = self._extract_ground_truth()
        
        logger.info(f"Sistema inicializado:")
        logger.info(f"  - Productos: {len(self.products)}")
        logger.info(f"  - Usuarios: {len(self.users)}")
        logger.info(f"  - Conversaciones: {len(self.conversations)}")
        logger.info(f"  - Feedbacks: {len(self.feedbacks)}")
    
    def _generate_products(self):
        """Genera productos simulados con categorías variadas."""
        products = []
        categories = ["electronics", "clothing", "home", "books", "sports", "toys", "beauty", "food"]
        
        for i in range(self.n_products):
            category = random.choice(categories)
            products.append({
                "id": f"P{i+1:03d}",
                "title": f"Producto {i+1} de {category}",
                "category": category,
                "price": random.randint(10, 1000),
                "popularity_score": random.uniform(0.1, 0.9),
                "description": f"Descripción del producto {i+1} en categoría {category}"
            })
        
        return products
    
    def _generate_users(self):
        """Genera usuarios simulados con preferencias."""
        users = []
        interests = ["gaming", "fashion", "cooking", "fitness", "reading", "travel", "music", "tech"]
        
        for i in range(min(self.n_users, 50)):  # Limitar a 50 usuarios para simulación
            user_interests = random.sample(interests, k=random.randint(1, 3))
            users.append({
                "id": f"U{i+1:03d}",
                "age_group": random.choice(["18-25", "26-35", "36-45", "46+"]),
                "interests": user_interests,
                "purchase_history": []
            })
        
        return users
    
    def _generate_conversations(self):
        """Genera conversaciones simuladas entre usuarios y sistema."""
        conversations = []
        
        # Tipos de consultas típicas
        query_templates = [
            "¿Recomiéndame productos de {category}?",
            "Busco algo relacionado con {interest}",
            "Necesito ayuda para elegir {product_type}",
            "¿Qué me sugieres para {situation}?",
            "Estoy buscando {product_description}"
        ]
        
        n_conversations = min(200, self.n_users * 3)  # Máximo 200 conversaciones
        
        for i in range(n_conversations):
            user = random.choice(self.users)
            query_template = random.choice(query_templates)
            
            # Reemplazar placeholders
            if "{category}" in query_template:
                category = random.choice(list(set(p["category"] for p in self.products)))
                query = query_template.replace("{category}", category)
            elif "{interest}" in query_template:
                interest = random.choice(user["interests"]) if user["interests"] else "tecnología"
                query = query_template.replace("{interest}", interest)
            else:
                query = query_template.replace("{situation}", "un regalo").replace("{product_type}", "electrónica").replace("{product_description}", "algo interesante")
            
            conversations.append({
                "id": f"C{i+1:03d}",
                "user_id": user["id"],
                "query": query,
                "timestamp": time.time() - random.randint(0, 30*24*3600),  # Últimos 30 días
                "products_shown": random.sample([p["id"] for p in self.products], k=random.randint(3, 10)),
                "user_engaged": random.choice([True, False, True])  # Más probabilidad de engagement
            })
        
        return conversations
    
    def _generate_feedbacks(self):
        """Genera feedbacks simulados de usuarios."""
        feedbacks = []
        
        for conv in self.conversations:
            if conv["user_engaged"] and random.random() > 0.3:  # 70% de conversaciones con engagement tienen feedback
                n_feedbacks = random.randint(1, min(3, len(conv["products_shown"])))
                products_feedback = random.sample(conv["products_shown"], k=n_feedbacks)
                
                for product_id in products_feedback:
                    feedbacks.append({
                        "conversation_id": conv["id"],
                        "user_id": conv["user_id"],
                        "product_id": product_id,
                        "rating": random.randint(1, 5),
                        "clicked": random.choice([True, False]),
                        "purchased": random.random() > 0.8,  # 20% de probabilidad de compra
                        "timestamp": conv["timestamp"] + random.randint(0, 3600)  # Dentro de 1 hora
                    })
        
        return feedbacks
    
    def _extract_ground_truth(self):
        """Extrae ground truth basado en feedbacks positivos."""
        ground_truth = {}
        
        for conv in self.conversations:
            # Productos con feedback positivo (rating >= 4 o comprados)
            positive_feedbacks = [
                fb for fb in self.feedbacks 
                if fb["conversation_id"] == conv["id"] and (fb["rating"] >= 4 or fb["purchased"])
            ]
            
            if positive_feedbacks:
                ground_truth[conv["query"]] = set(fb["product_id"] for fb in positive_feedbacks)
            else:
                # Si no hay feedback positivo, usar productos mostrados que fueron clickeados
                clicked_products = set(
                    fb["product_id"] for fb in self.feedbacks 
                    if fb["conversation_id"] == conv["id"] and fb["clicked"]
                )
                if clicked_products:
                    ground_truth[conv["query"]] = clicked_products
                else:
                    # Si no hay clicks, usar productos mostrados al azar (1-3)
                    ground_truth[conv["query"]] = set(random.sample(
                        conv["products_shown"], 
                        k=min(3, len(conv["products_shown"]))
                    ))
        
        logger.info(f"Ground truth extraído para {len(ground_truth)} consultas")
        return ground_truth
    
    def get_test_queries(self, n_queries=20):
        """Selecciona consultas de prueba del ground truth."""
        all_queries = list(self.ground_truth.keys())
        
        if len(all_queries) <= n_queries:
            selected_queries = all_queries
        else:
            # Seleccionar consultas balanceadas por dificultad
            selected_queries = []
            
            # Clasificar por dificultad (basado en tamaño de ground truth)
            easy_queries = [q for q, gt in self.ground_truth.items() if len(gt) <= 2]
            medium_queries = [q for q, gt in self.ground_truth.items() if 3 <= len(gt) <= 5]
            hard_queries = [q for q, gt in self.ground_truth.items() if len(gt) > 5]
            
            # Tomar proporciones balanceadas
            n_easy = min(3, len(easy_queries))
            n_medium = min(3, len(medium_queries))
            n_hard = min(n_queries - n_easy - n_medium, len(hard_queries))
            
            if n_easy > 0:
                selected_queries.extend(random.sample(easy_queries, n_easy))
            if n_medium > 0:
                selected_queries.extend(random.sample(medium_queries, n_medium))
            if n_hard > 0:
                selected_queries.extend(random.sample(hard_queries, n_hard))
            
            # Completar con consultas aleatorias si es necesario
            remaining = n_queries - len(selected_queries)
            if remaining > 0:
                remaining_queries = [q for q in all_queries if q not in selected_queries]
                if remaining_queries:
                    selected_queries.extend(random.sample(remaining_queries, min(remaining, len(remaining_queries))))
        
        queries = []
        ground_truths = []
        
        for query in selected_queries:
            queries.append(query)
            ground_truths.append(self.ground_truth[query])
        
        logger.info(f"Seleccionadas {len(queries)} consultas de prueba")
        return queries, ground_truths
    
    def simulate_rag_with_user_data(self, query, user_id=None, use_ml=False):
        """Simula RAG que usa datos de usuarios (conversaciones, feedbacks)."""
        # Obtener productos relevantes basados en la consulta
        query_lower = query.lower()
        relevant_products = []
        
        for product in self.products:
            score = 0.0
            
            # 1. Matching textual básico
            if query_lower in product["title"].lower():
                score += 0.6
            elif any(word in product["title"].lower() for word in query_lower.split()):
                score += 0.3
            
            # 2. Popularidad del producto
            score += product["popularity_score"] * 0.2
            
            # 3. Si hay usuario, usar su historial
            if user_id:
                user_feedbacks = [fb for fb in self.feedbacks if fb["user_id"] == user_id]
                user_products = set(fb["product_id"] for fb in user_feedbacks if fb["rating"] >= 3)
                
                if product["id"] in user_products:
                    score += 0.3
            
            # 4. Si use_ml está habilitado, usar embeddings semánticos simulados
            if use_ml:
                # Simular mejora ML con embeddings
                ml_boost = random.uniform(0.05, 0.25)
                score += ml_boost
            
            # 5. Ruido aleatorio
            score += random.uniform(-0.1, 0.1)
            
            score = max(0.0, min(1.0, score))
            relevant_products.append((product["id"], score))
        
        # Ordenar y devolver top 10
        relevant_products.sort(key=lambda x: x[1], reverse=True)
        
        # Introducir algo de variabilidad
        if random.random() < 0.4:  # 40% de chance de resultados imperfectos
            for i in range(len(relevant_products) - 1):
                if random.random() < 0.2 and abs(relevant_products[i][1] - relevant_products[i+1][1]) < 0.1:
                    relevant_products[i], relevant_products[i+1] = relevant_products[i+1], relevant_products[i]
        
        return [pid for pid, _ in relevant_products[:10]]
    
    def simulate_collaborative_with_user_data(self, user_id, candidate_products, use_ml=False):
        """Simula filtrado colaborativo usando datos de usuarios."""
        scores = {}
        
        # Obtener feedbacks del usuario
        user_feedbacks = [fb for fb in self.feedbacks if fb["user_id"] == user_id]
        
        for product_id in candidate_products:
            base_score = 0.3
            
            # 1. Feedback directo del usuario
            user_feedback = next((fb for fb in user_feedbacks if fb["product_id"] == product_id), None)
            if user_feedback:
                base_score += user_feedback["rating"] * 0.1
            
            # 2. Similitud con otros usuarios
            # Encontrar usuarios similares (mismos intereses o compras similares)
            similar_users = self._find_similar_users(user_id)
            
            # Calcular rating promedio de usuarios similares para este producto
            similar_ratings = []
            for sim_user_id in similar_users[:5]:  # Top 5 usuarios similares
                sim_feedbacks = [fb for fb in self.feedbacks if fb["user_id"] == sim_user_id and fb["product_id"] == product_id]
                if sim_feedbacks:
                    similar_ratings.append(np.mean([fb["rating"] for fb in sim_feedbacks]))
            
            if similar_ratings:
                base_score += np.mean(similar_ratings) * 0.1
            
            # 3. Popularidad del producto
            product = next((p for p in self.products if p["id"] == product_id), None)
            if product:
                base_score += product["popularity_score"] * 0.15
            
            # 4. Boost ML si está habilitado
            if use_ml:
                # ML puede detectar patrones complejos
                ml_factor = random.uniform(0.05, 0.2)
                base_score += ml_factor
            
            scores[product_id] = min(0.95, base_score)
        
        return scores
    
    def _find_similar_users(self, user_id):
        """Encuentra usuarios similares basados en intereses y comportamiento."""
        target_user = next((u for u in self.users if u["id"] == user_id), None)
        if not target_user:
            return []
        
        similar_users = []
        for user in self.users:
            if user["id"] == user_id:
                continue
            
            similarity = 0.0
            
            # Similitud de intereses
            common_interests = set(target_user["interests"]) & set(user["interests"])
            similarity += len(common_interests) * 0.2
            
            # Similitud de productos comprados/vistos
            target_products = set(fb["product_id"] for fb in self.feedbacks if fb["user_id"] == user_id and fb["rating"] >= 3)
            user_products = set(fb["product_id"] for fb in self.feedbacks if fb["user_id"] == user["id"] and fb["rating"] >= 3)
            
            common_products = target_products & user_products
            similarity += len(common_products) * 0.1
            
            if similarity > 0:
                similar_users.append((user["id"], similarity))
        
        # Ordenar por similitud
        similar_users.sort(key=lambda x: x[1], reverse=True)
        return [uid for uid, _ in similar_users[:10]]  # Top 10 usuarios similares
    
    def simulate_hybrid_with_user_data(self, query, user_id, use_ml=False, top_k=5):
        """Simula sistema híbrido que usa tanto RAG como datos de usuarios."""
        # Paso 1: RAG con datos de usuario
        rag_results = self.simulate_rag_with_user_data(query, user_id, use_ml=use_ml)
        
        if not rag_results:
            return []
        
        # Paso 2: Filtrado colaborativo
        collab_scores = self.simulate_collaborative_with_user_data(user_id, rag_results, use_ml=use_ml)
        
        # Paso 3: Combinación inteligente
        combined_scores = {}
        
        for i, product_id in enumerate(rag_results):
            # Score RAG (depende de la posición)
            rag_score = 1.0 - (i * 0.08)
            
            # Score colaborativo
            collab_score = collab_scores.get(product_id, 0.3)
            
            # Combinar con pesos adaptativos
            if use_ml:
                # ML ajusta pesos dinámicamente
                # Más peso a colaborativo si el usuario tiene mucho historial
                user_feedbacks = [fb for fb in self.feedbacks if fb["user_id"] == user_id]
                if len(user_feedbacks) > 5:
                    # Usuario con historial: 50% RAG, 50% colaborativo
                    rag_weight = 0.5
                    collab_weight = 0.5
                else:
                    # Usuario nuevo: 70% RAG, 30% colaborativo
                    rag_weight = 0.7
                    collab_weight = 0.3
            else:
                # Sin ML: pesos fijos
                rag_weight = 0.7
                collab_weight = 0.3
            
            combined_score = (rag_score * rag_weight) + (collab_score * collab_weight)
            combined_scores[product_id] = combined_score
        
        # Ordenar y devolver top_k
        sorted_products = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in sorted_products[:top_k]]
    
    def calculate_advanced_metrics(self, retrieved_lists, ground_truth_sets, user_ids=None):
        """Calcula métricas avanzadas considerando el contexto de usuario."""
        metrics = {}
        
        # Métricas básicas
        def safe_mean(values):
            return np.mean(values) if values else 0.0
        
        # Precision@K
        k = 5
        precisions = []
        for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
            relevant = sum(1 for pid in retrieved[:k] if pid in gt)
            precisions.append(relevant / k if k > 0 else 0.0)
        metrics["precision@5"] = safe_mean(precisions)
        
        # Recall@K
        recalls = []
        for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
            if not gt:
                recalls.append(0.0)
                continue
            relevant = sum(1 for pid in retrieved[:k] if pid in gt)
            recalls.append(relevant / len(gt))
        metrics["recall@5"] = safe_mean(recalls)
        
        # F1@K
        if metrics["precision@5"] + metrics["recall@5"] > 0:
            metrics["f1@5"] = 2 * metrics["precision@5"] * metrics["recall@5"] / (metrics["precision@5"] + metrics["recall@5"])
        else:
            metrics["f1@5"] = 0.0
        
        # Hit Rate@K
        hit_rates = []
        for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
            hit = any(pid in gt for pid in retrieved[:k])
            hit_rates.append(1.0 if hit else 0.0)
        metrics["hit_rate@5"] = safe_mean(hit_rates)
        
        # MAP@K
        ap_scores = []
        for retrieved, gt in zip(retrieved_lists, ground_truth_sets):
            if not gt:
                ap_scores.append(0.0)
                continue
            
            score = 0.0
            hits = 0
            
            for i, pid in enumerate(retrieved[:k], start=1):
                if pid in gt:
                    hits += 1
                    score += hits / i
            
            ap_scores.append(score / min(len(gt), k))
        metrics["map@5"] = safe_mean(ap_scores)
        
        # Coverage
        all_recommended = set()
        for retrieved in retrieved_lists:
            all_recommended.update(retrieved[:k])
        metrics["coverage"] = len(all_recommended) / len(self.products) if self.products else 0.0
        
        # Diversidad
        all_recommended_list = []
        for retrieved in retrieved_lists:
            all_recommended_list.extend(retrieved[:3])  # Primeros 3 para diversidad
        
        if all_recommended_list:
            unique_count = len(set(all_recommended_list))
            metrics["diversity"] = unique_count / len(all_recommended_list)
        else:
            metrics["diversity"] = 0.0
        
        # Novedad (vs productos populares)
        popular_products = set()
        if self.products:
            sorted_by_pop = sorted(self.products, key=lambda x: x["popularity_score"], reverse=True)
            popular_products = set(p["id"] for p in sorted_by_pop[:5])
        
        novelty_scores = []
        for retrieved in retrieved_lists:
            retrieved_k = retrieved[:k]
            if not retrieved_k:
                novelty_scores.append(0.0)
                continue
            
            novel_count = sum(1 for pid in retrieved_k if pid not in popular_products)
            novelty_scores.append(novel_count / len(retrieved_k))
        metrics["novelty@5"] = safe_mean(novelty_scores)
        
        # Personalización Score (solo si hay user_ids)
        if user_ids:
            personalization_scores = []
            for i, retrieved in enumerate(retrieved_lists):
                user_id = user_ids[i] if i < len(user_ids) else None
                if user_id:
                    # Verificar si los productos recomendados coinciden con intereses del usuario
                    user = next((u for u in self.users if u["id"] == user_id), None)
                    if user and retrieved[:3]:  # Primeros 3 productos
                        # Calcular relevancia personalizada
                        personal_score = self._calculate_personalization_score(user, retrieved[:3])
                        personalization_scores.append(personal_score)
            
            if personalization_scores:
                metrics["personalization@3"] = safe_mean(personalization_scores)
            else:
                metrics["personalization@3"] = 0.0
        
        return metrics
    
    def _calculate_personalization_score(self, user, recommended_products):
        """Calcula cuán personalizadas son las recomendaciones para un usuario."""
        if not user or not recommended_products:
            return 0.0
        
        score = 0.0
        
        # 1. Verificar si productos coinciden con intereses del usuario
        for pid in recommended_products:
            product = next((p for p in self.products if p["id"] == pid), None)
            if product:
                # Verificar si categoría del producto coincide con intereses del usuario
                category_keywords = {
                    "electronics": ["tech", "gaming", "computer"],
                    "clothing": ["fashion", "style"],
                    "books": ["reading", "education"],
                    "sports": ["fitness", "exercise"],
                    # ... agregar más mapeos
                }
                
                for interest in user.get("interests", []):
                    if interest in category_keywords.get(product["category"], []):
                        score += 0.2
                        break
        
        # 2. Verificar historial del usuario
        user_feedbacks = [fb for fb in self.feedbacks if fb["user_id"] == user["id"]]
        if user_feedbacks:
            # Productos similares a los que ya ha interactuado
            historical_products = set(fb["product_id"] for fb in user_feedbacks if fb["rating"] >= 3)
            similar_recommended = set(recommended_products) & historical_products
            score += len(similar_recommended) * 0.15
        
        return min(1.0, score / len(recommended_products))
    
    def evaluate_configuration(self, mode="rag", use_ml=False, n_runs=3, n_queries=15):
        """Evalúa una configuración específica."""
        all_metrics = []
        all_times = []
        
        for run in range(n_runs):
            # Obtener consultas de prueba
            queries, ground_truths = self.get_test_queries(n_queries)
            
            # Seleccionar usuarios para las consultas
            user_ids = []
            for query in queries:
                # Buscar usuario que haya hecho consulta similar
                similar_convs = [c for c in self.conversations if query.lower() in c["query"].lower()]
                if similar_convs:
                    user_ids.append(random.choice(similar_convs)["user_id"])
                else:
                    user_ids.append(random.choice(self.users)["id"])
            
            # Ejecutar consultas
            retrieved_lists = []
            start_time = time.time()
            
            for i, query in enumerate(queries):
                user_id = user_ids[i] if i < len(user_ids) else None
                
                if mode == "rag":
                    retrieved = self.simulate_rag_with_user_data(query, user_id, use_ml=use_ml)
                else:  # hybrid
                    retrieved = self.simulate_hybrid_with_user_data(query, user_id, use_ml=use_ml, top_k=5)
                
                retrieved_lists.append(retrieved)
            
            elapsed_time = time.time() - start_time
            
            # Calcular métricas
            metrics = self.calculate_advanced_metrics(retrieved_lists, ground_truths, user_ids)
            metrics["time_seconds"] = elapsed_time
            metrics["latency_per_query_ms"] = (elapsed_time / len(queries)) * 1000 if queries else 0
            metrics["queries_count"] = len(queries)
            
            all_metrics.append(metrics)
            all_times.append(elapsed_time)
        
        # Promediar métricas
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key not in ["config", "queries_count"]:
                values = [m[key] for m in all_metrics]
                avg_metrics[key] = np.mean(values)
                avg_metrics[f"{key}_std"] = np.std(values) if len(values) > 1 else 0.0
        
        avg_metrics["config"] = {
            "mode": mode,
            "ml_enabled": use_ml,
            "n_runs": n_runs,
            "n_queries": n_queries
        }
        
        avg_metrics["avg_time_seconds"] = np.mean(all_times)
        
        return avg_metrics

def main():
    """Función principal."""
    print("="*80)
    print("🚀 SISTEMA DE EVALUACIÓN PERSONALIZADO - DATOS DE USUARIOS SIMULADOS")
    print("="*80)
    
    # Inicializar sistema con parámetros realistas
    system = CustomEvaluationSystem(
        seed=42,
        n_products=25,  # Más productos para mayor realismo
        n_users=50      # Usuarios simulados
    )
    
    # Configuraciones a evaluar
    configs = [
        ("rag", False, "RAG sin ML"),
        ("rag", True, "RAG con ML"),
        ("hybrid", False, "Híbrido sin ML"),
        ("hybrid", True, "Híbrido con ML"),
    ]
    
    results = {}
    
    print(f"\n📊 EJECUTANDO EVALUACIONES (3 ejecuciones por configuración)...")
    print("-"*80)
    
    for mode, use_ml, name in configs:
        print(f"\n🔄 Evaluando: {name}...")
        
        try:
            metrics = system.evaluate_configuration(
                mode=mode, 
                use_ml=use_ml, 
                n_runs=3,
                n_queries=15
            )
            results[name] = metrics
            
            print(f"   ✅ Completado")
            print(f"   📈 Precision@5: {metrics['precision@5']:.3f} (±{metrics.get('precision@5_std', 0):.3f})")
            print(f"   🔍 Recall@5: {metrics['recall@5']:.3f} (±{metrics.get('recall@5_std', 0):.3f})")
            print(f"   🎯 F1@5: {metrics['f1@5']:.3f} (±{metrics.get('f1@5_std', 0):.3f})")
            print(f"   ⚡ Hit Rate@5: {metrics['hit_rate@5']:.3f}")
            
            if "personalization@3" in metrics:
                print(f"   👤 Personalización@3: {metrics['personalization@3']:.3f}")
            
            print(f"   📊 MAP@5: {metrics['map@5']:.3f}")
            print(f"   🌐 Coverage: {metrics['coverage']:.3f}")
            print(f"   🎲 Diversidad: {metrics['diversity']:.3f}")
            print(f"   🆕 Novelty@5: {metrics['novelty@5']:.3f}")
            print(f"   ⏱️  Latencia/query: {metrics['latency_per_query_ms']:.1f}ms")
            
        except Exception as e:
            print(f"   ❌ Error evaluando {name}: {e}")
            results[name] = {"error": str(e)}
    
    # Análisis comparativo
    print("\n" + "="*80)
    print("📈 ANÁLISIS COMPARATIVO")
    print("="*80)
    
    # Tabla comparativa
    print("\n📋 RESULTADOS COMPARATIVOS:")
    print("-"*100)
    headers = ["Sistema", "ML", "P@5", "R@5", "F1@5", "HR@5", "MAP@5", "Cov", "Div", "Nov", "Lat(ms)"]
    print(f"{headers[0]:<15} {headers[1]:<5} {headers[2]:<6} {headers[3]:<6} {headers[4]:<6} "
          f"{headers[5]:<6} {headers[6]:<7} {headers[7]:<5} {headers[8]:<5} {headers[9]:<5} {headers[10]:<8}")
    print("-"*100)
    
    for name, metrics in results.items():
        if "error" in metrics:
            continue
        
        ml_status = "Sí" if metrics["config"]["ml_enabled"] else "No"
        
        print(f"{name:<15} {ml_status:<5} "
              f"{metrics['precision@5']:.3f} "
              f"{metrics['recall@5']:.3f} "
              f"{metrics['f1@5']:.3f} "
              f"{metrics['hit_rate@5']:.3f} "
              f"{metrics['map@5']:.3f} "
              f"{metrics['coverage']:.3f} "
              f"{metrics['diversity']:.3f} "
              f"{metrics['novelty@5']:.3f} "
              f"{metrics['latency_per_query_ms']:.1f}")
    
    print("-"*100)
    
    # Determinar mejor sistema
    print("\n" + "="*80)
    print("🏆 MEJORES SISTEMAS POR CATEGORÍA:")
    print("="*80)
    
    categories = {
        "Calidad": ["precision@5", "recall@5", "f1@5"],
        "Efectividad": ["hit_rate@5", "map@5"],
        "Diversidad": ["coverage", "diversity", "novelty@5"]
    }
    
    for category, metrics_list in categories.items():
        print(f"\n{category}:")
        for metric in metrics_list:
            valid_results = [(name, m[metric]) for name, m in results.items() if "error" not in m and metric in m]
            if valid_results:
                best_name, best_value = max(valid_results, key=lambda x: x[1])
                print(f"  {metric:<15} → {best_name:<15} ({best_value:.3f})")
    
    # Sistema recomendado basado en ponderación
    print("\n" + "="*80)
    print("⚖️  SISTEMA RECOMENDADO (ponderación balanceada):")
    print("="*80)
    
    weighted_scores = {}
    for name, metrics in results.items():
        if "error" in metrics:
            continue
        
        # Ponderaciones: 25% F1, 20% Personalización, 15% Coverage, 15% Diversidad, 15% Novedad, 10% Latencia
        weights = {
            "f1@5": 0.25,
            "personalization@3": 0.20,
            "coverage": 0.15,
            "diversity": 0.15,
            "novelty@5": 0.15,
            "latency": 0.10
        }
        
        # Calcular score ponderado
        weighted_score = 0.0
        
        # F1 Score
        weighted_score += metrics.get("f1@5", 0) * weights["f1@5"]
        
        # Personalización (si está disponible)
        weighted_score += metrics.get("personalization@3", 0) * weights["personalization@3"]
        
        # Coverage
        weighted_score += metrics.get("coverage", 0) * weights["coverage"]
        
        # Diversidad
        weighted_score += metrics.get("diversity", 0) * weights["diversity"]
        
        # Novedad
        weighted_score += metrics.get("novelty@5", 0) * weights["novelty@5"]
        
        # Latencia (invertida: menor latencia = mejor)
        latency_score = max(0, 1 - (metrics.get("latency_per_query_ms", 100) / 500))  # Normalizar
        weighted_score += latency_score * weights["latency"]
        
        weighted_scores[name] = weighted_score
    
    if weighted_scores:
        best_system = max(weighted_scores.keys(), key=lambda x: weighted_scores[x])
        best_score = weighted_scores[best_system]
        
        print(f"\n🏅 Sistema recomendado: {best_system}")
        print(f"📊 Puntuación balanceada: {best_score:.3f}")
        print(f"📈 F1 Score: {results[best_system].get('f1@5', 0):.3f}")
        
        if "personalization@3" in results[best_system]:
            print(f"👤 Personalización: {results[best_system]['personalization@3']:.3f}")
        
        print(f"🌐 Coverage: {results[best_system].get('coverage', 0):.3f}")
        print(f"🎲 Diversidad: {results[best_system].get('diversity', 0):.3f}")
        print(f"🆕 Novelty: {results[best_system].get('novelty@5', 0):.3f}")
        print(f"⏱️  Latencia: {results[best_system].get('latency_per_query_ms', 0):.1f}ms")
    else:
        print("\n❌ No se pudieron calcular scores ponderados")
    
    # Guardar resultados
    output_data = {
        "timestamp": time.time(),
        "system_info": {
            "n_products": len(system.products),
            "n_users": len(system.users),
            "n_conversations": len(system.conversations),
            "n_feedbacks": len(system.feedbacks)
        },
        "results": results,
        "weighted_scores": weighted_scores,
        "recommended_system": best_system if weighted_scores else None
    }
    
    output_file = "evaluation_custom_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados detallados guardados en: {output_file}")
    print("="*80)
    
    # Recomendaciones finales
    print("\n🎯 RECOMENDACIONES BASADAS EN LOS RESULTADOS:")
    print("-"*80)
    
    if results:
        # Analizar fortalezas y debilidades
        rag_ml = results.get("RAG con ML", {})
        hybrid_ml = results.get("Híbrido con ML", {})
        
        if "error" not in rag_ml and "error" not in hybrid_ml:
            if rag_ml.get("f1@5", 0) > hybrid_ml.get("f1@5", 0):
                print("✅ RAG con ML tiene mejor F1 Score")
                print("   Sugerencia: Enfócate en mejorar el componente RAG")
            else:
                print("✅ Híbrido con ML tiene mejor balance")
                print("   Sugerencia: Combina RAG con filtrado colaborativo")
        
        # Verificar coverage
        low_coverage_systems = []
        for name, metrics in results.items():
            if "error" not in metrics and metrics.get("coverage", 0) < 0.4:
                low_coverage_systems.append(name)
        
        if low_coverage_systems:
            print(f"⚠️  Sistemas con coverage bajo (<40%): {', '.join(low_coverage_systems)}")
            print("   Considera técnicas de diversificación de recomendaciones")
        
        # Verificar diversidad
        low_diversity_systems = []
        for name, metrics in results.items():
            if "error" not in metrics and metrics.get("diversity", 0) < 0.3:
                low_diversity_systems.append(name)
        
        if low_diversity_systems:
            print(f"⚠️  Sistemas con diversidad baja (<30%): {', '.join(low_diversity_systems)}")
            print("   Añade mecanismos de exploración en las recomendaciones")
    
    print("="*80)

if __name__ == "__main__":
    main()