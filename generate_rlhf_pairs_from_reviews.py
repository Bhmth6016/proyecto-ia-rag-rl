# generate_rlhf_pairs_from_reviews.py
#!/usr/bin/env python3
"""
Generador de Pares RLHF - VERSIÓN MEJORADA
==========================================

Mejoras:
1. Auto-detecta TODAS las categorías disponibles
2. Templates expandidos para las 9 categorías
3. Procesa múltiples categorías en un solo run
4. Mejor logging y estadísticas
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional
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
    avg_rating: float
    num_reviews: int
    num_helpful_votes: int
    verified_purchase_ratio: float
    reward_score: float


class ReviewRewardCalculator:
    """Calcula rewards desde reviews usando múltiples señales"""
    
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        logger.info(f"📊 Reward Calculator: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
    
    def calculate_product_reward(self, reviews: List[Dict]) -> float:
        """Calcula reward agregado para un producto"""
        if not reviews:
            return 0.0
        
        # 1. Rating normalizado
        ratings = [r.get('rating', 0.0) for r in reviews]
        avg_rating = np.mean(ratings) if ratings else 0.0
        rating_norm = avg_rating / 5.0
        
        # 2. Helpful votes (log-scaled)
        total_helpful = sum(r.get('helpful_vote', 0) for r in reviews)
        helpful_score = np.log1p(total_helpful) / 10.0
        helpful_score = min(1.0, helpful_score)
        
        # 3. Verified purchase ratio
        verified_count = sum(1 for r in reviews if r.get('verified_purchase', False))
        verified_ratio = verified_count / len(reviews) if reviews else 0.0
        
        # 4. Recency (neutral por ahora)
        recency_score = 0.5
        
        # Reward final
        reward = (
            self.alpha * rating_norm +
            self.beta * helpful_score +
            self.gamma * verified_ratio +
            self.delta * recency_score
        )
        
        return min(1.0, max(0.0, reward))


class PseudoQueryGenerator:
    """Genera queries sintéticas - EXPANDIDO para 9 categorías"""
    
    def __init__(self):
        # ✅ EXPANDIDO: Templates para TODAS las categorías
        self.category_templates = {
            'Automotive': [
                'car parts', 'auto accessories', 'vehicle parts',
                'car accessories', 'automotive tools'
            ],
            'Beauty_and_Personal_Care': [
                'beauty products', 'makeup', 'skincare',
                'hair care', 'personal care'
            ],
            'Books': [
                'books', 'novels', 'educational books',
                'fiction books', 'non-fiction'
            ],
            'Clothing_Shoes_and_Jewelry': [
                'clothing', 'shoes', 'jewelry',
                'fashion', 'apparel'
            ],
            'Electronics': [
                'electronics', 'gadgets', 'tech products',
                'electronic devices', 'tech accessories'
            ],
            'Home_and_Kitchen': [
                'home products', 'kitchen items', 'home decor',
                'kitchenware', 'home accessories'
            ],
            'Sports_and_Outdoors': [
                'sports equipment', 'outdoor gear', 'fitness products',
                'athletic gear', 'camping equipment'
            ],
            'Toys_and_Games': [
                'toys for kids', 'games', 'children toys',
                'board games', 'educational toys'
            ],
            'Video_Games': [
                'video games', 'gaming', 'pc games',
                'console games', 'playstation games', 'xbox games'
            ]
        }
    
    def generate_queries(self, product: Dict, num_queries: int = 3) -> List[str]:
        """Genera queries para un producto"""
        queries = []
        
        # 1. Query desde categoría
        category = product.get('main_category', '')
        
        # ✅ MEJORADO: Buscar template flexible
        template = None
        for cat_key, templates in self.category_templates.items():
            if cat_key.lower() in category.lower() or category.lower() in cat_key.lower():
                template = templates
                break
        
        if template:
            queries.extend(random.sample(template, min(2, len(template))))
        else:
            # Fallback genérico
            queries.append(category.lower().replace('_', ' '))
        
        # 2. Query desde título
        title = product.get('title', '')
        if title:
            words = [w for w in title.split() if len(w) > 3][:3]
            if words:
                queries.append(' '.join(words).lower())
        
        return queries[:num_queries]


class RLHFPairGenerator:
    """Genera pares (chosen, rejected) para RLHF"""
    
    def __init__(
        self,
        data_dir: Path = Path("data"),
        output_dir: Path = Path("data/rlhf_pairs"),
        min_reviews: int = 5,
        pairs_per_query: int = 3
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.min_reviews = min_reviews
        self.pairs_per_query = pairs_per_query
        
        self.reward_calc = ReviewRewardCalculator()
        self.query_gen = PseudoQueryGenerator()
        
        logger.info("🔧 RLHF Pair Generator inicializado")
        logger.info(f"   • Data dir: {data_dir}")
        logger.info(f"   • Min reviews: {min_reviews}")
        logger.info(f"   • Pairs/query: {pairs_per_query}")
    
    def find_available_categories(self) -> List[str]:
        """✅ NUEVO: Auto-detecta categorías disponibles"""
        raw_dir = self.data_dir / "raw"
        reviews_dir = self.data_dir / "reviews"
        
        # Buscar archivos meta_*.jsonl
        meta_files = list(raw_dir.glob("meta_*.jsonl"))
        
        categories = []
        for meta_file in meta_files:
            # Extraer nombre de categoría
            category = meta_file.stem.replace("meta_", "").replace("_10000", "")
            
            # Verificar que exista el archivo de reviews correspondiente
            review_file = reviews_dir / f"{category}_10000.jsonl"
            if not review_file.exists():
                review_file = reviews_dir / f"{category}.jsonl"
            
            if review_file.exists():
                categories.append(category)
                logger.info(f"   ✓ Categoría encontrada: {category}")
            else:
                logger.warning(f"   ✗ Sin reviews para: {category}")
        
        return categories
    
    def load_products(self, category: str, limit: Optional[int] = None) -> Dict[str, Dict]:
        """Carga productos de una categoría"""
        products_file = self.data_dir / "raw" / f"meta_{category}_10000.jsonl"
        if not products_file.exists():
            products_file = self.data_dir / "raw" / f"meta_{category}.jsonl"
        
        logger.info(f"📂 Cargando productos: {products_file.name}")
        
        products = {}
        count = 0
        
        with open(products_file, 'r', encoding='utf-8') as f:
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
    
    def aggregate_reviews(self, category: str, limit: Optional[int] = None) -> Dict[str, List[Dict]]:
        """Agrupa reviews por producto"""
        reviews_file = self.data_dir / "reviews" / f"{category}_10000.jsonl"
        if not reviews_file.exists():
            reviews_file = self.data_dir / "reviews" / f"{category}.jsonl"
        
        logger.info(f"📂 Agregando reviews: {reviews_file.name}")
        
        reviews_by_product = defaultdict(list)
        count = 0
        
        with open(reviews_file, 'r', encoding='utf-8') as f:
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
        
        logger.info(f"✅ Reviews: {count:,} | Productos: {len(reviews_by_product):,}")
        return reviews_by_product
    
    def calculate_rewards(
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
            
            reward = self.reward_calc.calculate_product_reward(reviews)
            
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
        
        product_rewards.sort(key=lambda x: x.reward_score, reverse=True)
        logger.info(f"✅ Productos con rewards: {len(product_rewards):,}")
        
        return product_rewards
    
    def generate_pairs(
        self,
        products: Dict[str, Dict],
        product_rewards: List[ProductRewardStats]
    ) -> List[Dict]:
        """Genera pares (chosen, rejected)"""
        logger.info("🎯 Generando pares de preferencia...")
        
        pairs = []
        
        # Generar queries
        sample_product = list(products.values())[0] if products else {}
        queries = self.query_gen.generate_queries(sample_product)
        
        for query in queries:
            # Top vs Bottom products
            top_products = product_rewards[:min(10, len(product_rewards))]
            bottom_products = product_rewards[-min(10, len(product_rewards)):]
            
            for _ in range(self.pairs_per_query):
                if not top_products or not bottom_products:
                    break
                
                chosen = random.choice(top_products)
                rejected = random.choice(bottom_products)
                
                if chosen.reward_score <= rejected.reward_score:
                    continue
                
                # Obtener imagen del producto original
                chosen_product = products.get(chosen.parent_asin, {})
                rejected_product = products.get(rejected.parent_asin, {})
                
                pair = {
                    'query': query,
                    'chosen': {
                        'parent_asin': chosen.parent_asin,
                        'title': chosen.title,
                        'image_url': chosen_product.get('imageURL') or chosen_product.get('imUrl'),  # NUEVO
                        'reward_score': chosen.reward_score,
                        'avg_rating': chosen.avg_rating,
                        'num_reviews': chosen.num_reviews
                    },
                    'rejected': {
                        'parent_asin': rejected.parent_asin,
                        'title': rejected.title,
                        'image_url': rejected_product.get('imageURL') or rejected_product.get('imUrl'),  # NUEVO
                        'reward_score': rejected.reward_score,
                        'avg_rating': rejected.avg_rating,
                        'num_reviews': rejected.num_reviews
                    },
                    'margin': chosen.reward_score - rejected.reward_score,
                    'category': chosen.category
                }
                
                pairs.append(pair)
        
        logger.info(f"✅ Pares generados: {len(pairs):,}")
        return pairs
    
    def save_pairs(self, pairs: List[Dict], category: str):
        """Guarda pares en formato JSONL"""
        output_file = self.output_dir / f"rlhf_pairs_{category.lower()}.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Guardando: {output_file.name}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ Guardado: {len(pairs):,} pares")
        return output_file
    
    def process_category(
        self,
        category: str,
        limit_products: Optional[int] = None,
        limit_reviews: Optional[int] = None
    ) -> Optional[Path]:
        """Procesa una categoría completa"""
        logger.info("\n" + "="*60)
        logger.info(f"PROCESANDO: {category}")
        logger.info("="*60)
        
        try:
            # 1. Cargar datos
            products = self.load_products(category, limit=limit_products)
            reviews_by_product = self.aggregate_reviews(category, limit=limit_reviews)
            
            if not products or not reviews_by_product:
                logger.warning(f"⚠️ Datos insuficientes para {category}")
                return None
            
            # 2. Calcular rewards
            product_rewards = self.calculate_rewards(products, reviews_by_product)
            
            if len(product_rewards) < 2:
                logger.warning(f"⚠️ Muy pocos productos con rewards para {category}")
                return None
            
            # 3. Generar pares
            pairs = self.generate_pairs(products, product_rewards)
            
            if not pairs:
                logger.warning(f"⚠️ No se generaron pares para {category}")
                return None
            
            # 4. Guardar
            output_file = self.save_pairs(pairs, category)
            
            # Estadísticas
            rewards = [p.reward_score for p in product_rewards]
            margins = [p['margin'] for p in pairs]
            
            logger.info("\n📊 ESTADÍSTICAS:")
            logger.info(f"   Productos:  {len(products):,}")
            logger.info(f"   Con rewards: {len(product_rewards):,}")
            logger.info(f"   Pares RLHF:  {len(pairs):,}")
            logger.info(f"   Reward medio: {np.mean(rewards):.3f}")
            logger.info(f"   Margin medio: {np.mean(margins):.3f}")
            
            return output_file
            
        except Exception as e:
            logger.error(f"❌ Error procesando {category}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_categories(
        self,
        limit_products: Optional[int] = None,
        limit_reviews: Optional[int] = None
    ):
        """✅ NUEVO: Procesa TODAS las categorías disponibles"""
        logger.info("\n" + "="*60)
        logger.info("🚀 PROCESANDO TODAS LAS CATEGORÍAS")
        logger.info("="*60)
        
        # Auto-detectar categorías
        categories = self.find_available_categories()
        
        if not categories:
            logger.error("❌ No se encontraron categorías")
            return
        
        logger.info(f"\n📂 Categorías a procesar: {len(categories)}")
        for cat in categories:
            logger.info(f"   • {cat}")
        
        # Procesar cada categoría
        results = {}
        total_pairs = 0
        
        for category in categories:
            output_file = self.process_category(
                category,
                limit_products=limit_products,
                limit_reviews=limit_reviews
            )
            
            if output_file:
                # Contar pares generados
                with open(output_file, 'r', encoding='utf-8') as f:  # ✅ FIX: encoding para Windows
                    num_pairs = sum(1 for _ in f)
                results[category] = num_pairs
                total_pairs += num_pairs
            else:
                results[category] = 0
        
        # Resumen final
        logger.info("\n" + "="*60)
        logger.info("📊 RESUMEN FINAL")
        logger.info("="*60)
        
        for category, num_pairs in results.items():
            status = "✓" if num_pairs > 0 else "✗"
            logger.info(f"{status} {category:40} {num_pairs:>6,} pares")
        
        logger.info(f"\nTOTAL: {total_pairs:,} pares RLHF generados")
        logger.info(f"Categorías exitosas: {sum(1 for n in results.values() if n > 0)}/{len(results)}")
        logger.info("="*60)


def main():
    """Ejemplo de uso"""
    
    # Inicializar generador
    generator = RLHFPairGenerator(
        data_dir=Path("data"),
        output_dir=Path("data/rlhf_pairs"),
        min_reviews=5,
        pairs_per_query=3
    )
    
    # ✅ OPCIÓN 1: Procesar TODAS las categorías (RECOMENDADO)
    generator.run_all_categories(
        limit_products=10000,   # Todos los productos disponibles
        limit_reviews=100000    # Primeras 100K reviews por categoría
    )

if __name__ == "__main__":
    main()