# scripts/balance_categories.py - VERSIÓN MEJORADA

import json
import random
from pathlib import Path
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class CategoryBalancer:
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.products = []
        
    def load_products(self):
        """Carga productos desde el archivo JSON"""
        logger.info(f"🔧 BALANCEANDO CATEGORÍAS EN: {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.products = json.load(f)
        
        logger.info(f"📦 Productos totales: {len(self.products)}")
        return self.products
    
    def analyze_distribution(self):
        """Analiza la distribución actual de categorías"""
        categories = Counter()
        for product in self.products:
            category = product.get('main_category', 'General') or 'General'
            categories[category] += 1
        
        total = len(self.products)
        logger.info(f"📊 DISTRIBUCIÓN ACTUAL ({total} productos):")
        
        for category, count in categories.most_common():
            percentage = (count / total) * 100
            logger.info(f"   • {category}: {count} ({percentage:.1f}%)")
        
        return dict(categories)
    
    def balance_categories(self, target_min: int = 800, target_max: int = 4000):
        """
        Balancea categorías moviendo productos de categorías sobre-representadas
        a categorías infra-representadas
        """
        # Identificar categorías objetivo principales
        target_categories = [
            'Electronics', 'Video Games', 'Home & Kitchen', 'Clothing', 'Books',
            'Sports & Outdoors', 'Beauty', 'Toys & Games', 'Office Products', 'Automotive',
            'Health & Personal Care', 'Grocery', 'Tools & Home Improvement'
        ]
        
        # Análisis inicial
        initial_dist = self.analyze_distribution()
        
        # Identificar categorías con exceso
        excess_categories = []
        for category, count in initial_dist.items():
            if count > target_max and category not in target_categories:
                excess_categories.append((category, count))
        
        # Identificar categorías que necesitan más productos
        need_categories = []
        for category in target_categories:
            count = initial_dist.get(category, 0)
            if count < target_min:
                need_categories.append((category, count, target_min - count))
        
        # Ordenar por prioridad (las que más necesitan primero)
        need_categories.sort(key=lambda x: x[2], reverse=True)
        
        logger.info("\n🎯 Categorías objetivo principales:")
        for category in target_categories[:10]:
            count = initial_dist.get(category, 0)
            logger.info(f"   • {category}: {count} productos")
        
        # Balancear categorías
        changed_count = 0
        
        for target_category, current_count, needed in need_categories:
            if not excess_categories:
                break
            
            # Tomar productos de categorías con exceso
            for source_category, source_count in excess_categories:
                if source_count <= target_max:
                    continue
                
                # Calcular cuántos podemos tomar (máximo 10% de la fuente)
                can_take = min(needed, int(source_count * 0.1), source_count - target_max)
                if can_take <= 0:
                    continue
                
                # Encontrar productos de la categoría fuente
                source_products = []
                for i, product in enumerate(self.products):
                    if product.get('main_category') == source_category:
                        source_products.append(i)
                        if len(source_products) >= can_take:
                            break
                
                # Cambiar categoría
                for idx in source_products:
                    self.products[idx]['main_category'] = target_category
                    changed_count += 1
                    needed -= 1
                
                # Actualizar contadores
                initial_dist[source_category] -= len(source_products)
                initial_dist[target_category] = initial_dist.get(target_category, 0) + len(source_products)
                
                logger.info(f"   • {source_category} → {target_category}: {len(source_products)} productos")
                
                if needed <= 0:
                    break
        
        logger.info(f"\n✅ Cambiadas {changed_count} categorías")
        
        # Crear backup
        backup_path = self.data_path.with_suffix('.json.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(self.products, f, ensure_ascii=False, indent=2)
        logger.info(f"📦 Backup: {backup_path}")
        
        return changed_count
    
    def save_balanced_data(self):
        """Guarda los datos balanceados"""
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(self.products, f, ensure_ascii=False, indent=2)
        
        # Verificar distribución final
        final_dist = Counter()
        for product in self.products:
            category = product.get('main_category', 'General') or 'General'
            final_dist[category] += 1
        
        total = len(self.products)
        logger.info(f"\n📊 NUEVA DISTRIBUCIÓN:")
        logger.info(f"📊 DISTRIBUCIÓN ACTUAL ({total} productos):")
        
        for category, count in final_dist.most_common(40):
            percentage = (count / total) * 100
            logger.info(f"   • {category}: {count} ({percentage:.1f}%)")
        
        return final_dist

def main():
    data_path = Path("data/processed/products.json")
    
    if not data_path.exists():
        logger.error(f"❌ No se encuentra el archivo: {data_path}")
        return
    
    balancer = CategoryBalancer(data_path)
    balancer.load_products()
    balancer.balance_categories(target_min=1000, target_max=5000)
    balancer.save_balanced_data()
    
    logger.info("\n🎯 RECOMENDACIÓN: Ahora ejecuta 'python main.py index' para reconstruir el índice")

if __name__ == "__main__":
    main()