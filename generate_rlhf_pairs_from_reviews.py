# generate_rlhf_pairs_from_reviews.py
#!/usr/bin/env python3
"""
Generador de Pares de Preferencia RLHF desde Reviews
====================================================

Convierte las reviews de productos en pares (chosen, rejected) para RLHF.

Estrategia:
1. Agregar reviews por producto → reward score
2. Generar pseudo-queries desde categorías/títulos
3. Crear pares de preferencia: productos con mejor reward vs peor reward
4. Formato compatible con el sistema RLHF existente

Basado en: https://arxiv.org/abs/2203.02155 (InstructGPT)
"""

import json
import gzip
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProductRewardStats:
    """Estadísticas de reward para un producto"""
    parent_asin: str
    title: str
    category: str
    
    # Métricas de reviews
    avg_rating: float
    num_reviews: int
    num_helpful_votes: int
    verified_purchase_ratio: float
    
    # Reward final
    reward_score: float
    
    def __repr__(self):
        return f"Product({self.parent_asin[:8]}..., reward={self.reward_score:.3f}, reviews={self.num_reviews})"


class ReviewRewardCalculator:
    """
    Calcula rewards desde reviews usando múltiples señales.
    
    Fórmula:
        reward = α*rating_norm + β*log(1+helpful) + γ*verified + δ*recency
    """
    
    def __init__(
        self,
        alpha: float = 0.4,  # Peso rating
        beta: float = 0.3,   # Peso helpful votes
        gamma: float = 0.2,  # Peso verified purchase
        delta: float = 0.1   # Peso recency
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        logger.info(f"📊 Reward Calculator: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
    
    def calculate_product_reward(self, reviews: List[Dict]) -> float:
        """
        Calcula reward agregado para un producto desde sus reviews.
        
        Returns:
            float: Reward score normalizado [0, 1]
        """
        if not reviews:
            return 0.0
        
        # 1. Rating normalizado
        ratings = [r.get('rating', 0.0) for r in reviews]
        avg_rating = np.mean(ratings) if ratings else 0.0
        rating_norm = avg_rating / 5.0
        
        # 2. Helpful votes (log-scaled)
        total_helpful = sum(r.get('helpful_vote', 0) for r in reviews)
        helpful_score = np.log1p(total_helpful) / 10.0  # Normalizar ~[0, 1]
        helpful_score = min(1.0, helpful_score)
        
        # 3. Verified purchase ratio
        verified_count = sum(1 for r in reviews if r.get('verified_purchase', False))
        verified_ratio = verified_count / len(reviews) if reviews else 0.0
        
        # 4. Recency (opcional - timestamps)
        # Por ahora no implementado, pero se puede agregar
        recency_score = 0.5  # Neutral
        
        # Reward final
        reward = (
            self.alpha * rating_norm +
            self.beta * helpful_score +
            self.gamma * verified_ratio +
            self.delta * recency_score
        )
        
        return min(1.0, max(0.0, reward))


class PseudoQueryGenerator:
    """
    Genera queries sintéticas para productos.
    
    Estrategias:
    1. Usar categoría principal
    2. Extraer keywords del título
    3. Combinar categoría + atributos clave
    """
    
    def __init__(self):
        self.category_templates = {
            'All_Beauty': ['beauty products', 'makeup', 'skincare'],
            'Toys_and_Games': ['toys for kids', 'games', 'children toys'],
            'Video_Games': ['video games', 'gaming', 'playstation games'],
            'Electronics': ['electronics', 'gadgets', 'tech products'],
            'Automotive': ['car parts', 'auto accessories', 'vehicle parts'],
            'Sports_and_Outdoors': ['sports equipment', 'outdoor gear'],
        }
    
    def generate_queries(self, product: Dict, num_queries: int = 3) -> List[str]:
        """Genera queries para un producto"""
        queries = []
        
        # 1. Query desde categoría
        category = product.get('main_category', '')
        if category in self.category_templates:
            queries.extend(random.sample(
                self.category_templates[category],
                min(2, len(self.category_templates[category]))
            ))
        else:
            # Fallback genérico
            queries.append(category.lower().replace('_', ' '))
        
        # 2. Query desde título (primeras 3 palabras significativas)
        title = product.get('title', '')
        if title:
            words = [w for w in title.split() if len(w) > 3][:3]
            if words:
                queries.append(' '.join(words).lower())
        
        # Limitar a num_queries
        return queries[:num_queries]


class RLHFPairGenerator:
    """
    Genera pares (chosen, rejected) para RLHF desde reviews.
    """
    
    def __init__(
        self,
        products_file: Path,
        reviews_file: Path,
        output_file: Path,
        min_reviews: int = 5,
        pairs_per_query: int = 3
    ):
        self.products_file = products_file
        self.reviews_file = reviews_file
        self.output_file = output_file
        self.min_reviews = min_reviews
        self.pairs_per_query = pairs_per_query
        self.products_file = products_file
        self.reviews_file = reviews_file
        self.reward_calc = ReviewRewardCalculator()
        self.query_gen = PseudoQueryGenerator()
        
        logger.info("🔧 RLHF Pair Generator inicializado")
        logger.info(f"   • Min reviews: {min_reviews}")
        logger.info(f"   • Pairs/query: {pairs_per_query}")
    
    def load_products(self, limit: Optional[int] = None) -> Dict[str, Dict]:
        logger.info(f"📂 Cargando productos desde {self.products_file.name}...")
        
        products = {}
        count = 0
        
        # DETECTAR si es .gz o .jsonl normal
        if str(self.products_file).endswith('.gz'):
            open_fn = gzip.open
        else:
            open_fn = open
        
        with open_fn(self.products_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if limit and count >= limit:
                    break
                
                try:
                    product = json.loads(line)
                    parent_asin = product.get('parent_asin')
                    if parent_asin:
                        products[parent_asin] = product
                        count += 1
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"✅ Productos cargados: {len(products):,}")
        return products
    
    def aggregate_reviews_by_product(
        self,
        limit: Optional[int] = None
    ) -> Dict[str, List[Dict]]:
        """Agrupa reviews por parent_asin"""
        logger.info(f"📂 Agregando reviews desde {self.reviews_file.name}...")
        
        reviews_by_product = defaultdict(list)
        count = 0
        
         # DETECTAR si es .gz o .jsonl normal
        if str(self.reviews_file).endswith('.gz'):
            open_fn = gzip.open
        else:
            open_fn = open
        
        with open_fn(self.reviews_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if limit and count >= limit:
                    break
                
                try:
                    review = json.loads(line)
                    parent_asin = review.get('parent_asin')
                    if parent_asin:
                        reviews_by_product[parent_asin].append(review)
                        count += 1
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"✅ Reviews procesadas: {count:,}")
        logger.info(f"✅ Productos con reviews: {len(reviews_by_product):,}")
        
        return reviews_by_product
    
    def calculate_product_rewards(
        self,
        products: Dict[str, Dict],
        reviews_by_product: Dict[str, List[Dict]]
    ) -> List[ProductRewardStats]:
        """Calcula rewards para cada producto"""
        logger.info("📊 Calculando rewards...")
        
        product_rewards = []
        
        for parent_asin, product in tqdm(products.items(), desc="Rewards"):
            reviews = reviews_by_product.get(parent_asin, [])
            
            if len(reviews) < self.min_reviews:
                continue
            
            # Calcular reward
            reward = self.reward_calc.calculate_product_reward(reviews)
            
            # Estadísticas
            ratings = [r.get('rating', 0.0) for r in reviews]
            helpful_votes = sum(r.get('helpful_vote', 0) for r in reviews)
            verified = sum(1 for r in reviews if r.get('verified_purchase', False))
            
            stats = ProductRewardStats(
                parent_asin=parent_asin,
                title=product.get('title', '')[:100],
                category=product.get('main_category', ''),
                avg_rating=np.mean(ratings) if ratings else 0.0,
                num_reviews=len(reviews),
                num_helpful_votes=helpful_votes,
                verified_purchase_ratio=verified / len(reviews) if reviews else 0.0,
                reward_score=reward
            )
            
            product_rewards.append(stats)
        
        # Ordenar por reward
        product_rewards.sort(key=lambda x: x.reward_score, reverse=True)
        
        logger.info(f"✅ Productos con rewards: {len(product_rewards):,}")
        
        return product_rewards
    
    def generate_preference_pairs(
        self,
        products: Dict[str, Dict],
        product_rewards: List[ProductRewardStats]
    ) -> List[Dict]:
        """
        Genera pares (chosen, rejected) para RLHF.
        
        Estrategia:
        - Para cada query, tomar productos con reward alto vs bajo
        - Asegurar que chosen.reward > rejected.reward
        """
        logger.info("🎯 Generando pares de preferencia...")
        
        pairs = []
        
        # Agrupar por categoría para queries relevantes
        by_category = defaultdict(list)
        for stats in product_rewards:
            by_category[stats.category].append(stats)
        
        for category, category_products in tqdm(by_category.items(), desc="Categorías"):
            if len(category_products) < 2:
                continue
            
            # Generar queries para esta categoría
            sample_product = products.get(category_products[0].parent_asin, {})
            queries = self.query_gen.generate_queries(sample_product)
            
            for query in queries:
                # Crear pares: top vs bottom
                top_products = category_products[:10]  # Mejores
                bottom_products = category_products[-10:]  # Peores
                
                for _ in range(self.pairs_per_query):
                    if not top_products or not bottom_products:
                        break
                    
                    chosen = random.choice(top_products)
                    rejected = random.choice(bottom_products)
                    
                    # Asegurar chosen > rejected
                    if chosen.reward_score <= rejected.reward_score:
                        continue
                    
                    pair = {
                        'query': query,
                        'chosen': {
                            'parent_asin': chosen.parent_asin,
                            'title': chosen.title,
                            'reward_score': chosen.reward_score,
                            'avg_rating': chosen.avg_rating,
                            'num_reviews': chosen.num_reviews
                        },
                        'rejected': {
                            'parent_asin': rejected.parent_asin,
                            'title': rejected.title,
                            'reward_score': rejected.reward_score,
                            'avg_rating': rejected.avg_rating,
                            'num_reviews': rejected.num_reviews
                        },
                        'margin': chosen.reward_score - rejected.reward_score
                    }
                    
                    pairs.append(pair)
        
        logger.info(f"✅ Pares generados: {len(pairs):,}")
        
        return pairs
    
    def save_pairs(self, pairs: List[Dict]):
        """Guarda pares en formato JSONL"""
        logger.info(f"💾 Guardando pares en {self.output_file}...")
        
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ Guardado: {len(pairs):,} pares")
    
    def run(self, limit_products: Optional[int] = None, limit_reviews: Optional[int] = None):
        """Pipeline completo"""
        logger.info("="*60)
        logger.info("🚀 INICIANDO GENERACIÓN DE PARES RLHF")
        logger.info("="*60)
        
        # 1. Cargar datos
        products = self.load_products(limit=limit_products)
        reviews_by_product = self.aggregate_reviews_by_product(limit=limit_reviews)
        
        # 2. Calcular rewards
        product_rewards = self.calculate_product_rewards(products, reviews_by_product)
        
        # 3. Generar pares
        pairs = self.generate_preference_pairs(products, product_rewards)
        
        # 4. Guardar
        self.save_pairs(pairs)
        
        # Estadísticas finales
        logger.info("\n" + "="*60)
        logger.info("📊 ESTADÍSTICAS FINALES")
        logger.info("="*60)
        logger.info(f"Productos totales:       {len(products):,}")
        logger.info(f"Productos con rewards:   {len(product_rewards):,}")
        logger.info(f"Pares RLHF generados:    {len(pairs):,}")
        
        if product_rewards:
            logger.info(f"\nReward stats:")
            rewards = [p.reward_score for p in product_rewards]
            logger.info(f"  • Mean:   {np.mean(rewards):.3f}")
            logger.info(f"  • Median: {np.median(rewards):.3f}")
            logger.info(f"  • Std:    {np.std(rewards):.3f}")
        
        if pairs:
            margins = [p['margin'] for p in pairs]
            logger.info(f"\nPair margin stats:")
            logger.info(f"  • Mean:   {np.mean(margins):.3f}")
            logger.info(f"  • Median: {np.median(margins):.3f}")
        
        logger.info("="*60)


def main():
    """Ejemplo de uso"""
    
    # Configuración
    DATA_DIR = Path("data/raw")
    REVIEW_DIR = Path("data/reviews")
    OUTPUT_DIR = Path("data/rlhf_pairs")
    
    # Archivos (ajustar según tu estructura)
    products_file = DATA_DIR / "meta_All_Beauty.jsonl"
    reviews_file = REVIEW_DIR / "All_Beauty.jsonl"
    output_file = OUTPUT_DIR / "rlhf_pairs_beauty.jsonl"
    
    # Generador
    generator = RLHFPairGenerator(
        products_file=products_file,
        reviews_file=reviews_file,
        output_file=output_file,
        min_reviews=5,
        pairs_per_query=3
    )
    
    # Ejecutar (limitar para prueba)
    generator.run(
        limit_products=10000,   # Primeros 10K productos
        limit_reviews=100000    # Primeras 100K reviews
    )


if __name__ == "__main__":
    main()