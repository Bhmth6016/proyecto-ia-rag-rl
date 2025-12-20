# src/data/optimized_loader.py - VERSIÓN CORREGIDA
"""
Loader optimizado con caché para evitar reprocesamiento
"""
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class OptimizedDataLoader:
    """Cargador de datos con caché inteligente"""
    
    def __init__(self, cache_enabled=True):
        self.cache_manager = CacheManager() if cache_enabled else None
        self.raw_products = []
        self.canonical_products = []
        
    def load_all_with_cache(self, force_reload=False):
        """
        Carga TODOS los datos usando caché cuando sea posible
        
        Args:
            force_reload: Si True, ignora caché y reprocesa todo
        
        Returns:
            Lista de productos canonicalizados
        """
        if not force_reload and self.cache_manager:
            # Intentar cargar desde caché
            cached_products = self.cache_manager.load_canonical_products("amazon_full")
            
            if cached_products is not None:
                logger.info(f"✅ {len(cached_products):,} productos cargados desde caché")
                self.canonical_products = cached_products
                return self.canonical_products
        
        # Si no hay caché o force_reload=True, cargar desde raw
        logger.info("📥 Cargando productos desde archivos raw...")
        self.raw_products = self._load_raw_products()
        
        # Canonicalizar
        logger.info("🔧 Canonicalizando productos...")
        self.canonical_products = self._canonicalize_batch(self.raw_products)
        
        # Guardar en caché
        if self.cache_manager:
            self.cache_manager.save_canonical_products(
                self.canonical_products, 
                "amazon_full"
            )
        
        return self.canonical_products
    
    def _load_raw_products(self, limit=None):
        """Carga productos desde archivos JSONL"""
        all_products = []
        data_dir = Path("data/raw")
        
        files_to_load = [
            "meta_Automotive_10000.jsonl",
            "meta_Beauty_and_Personal_Care_10000.jsonl",
            "meta_Books_10000.jsonl",
            "meta_Clothing_Shoes_and_Jewelry_10000.jsonl", 
            "meta_Electronics_10000.jsonl",
            "meta_Home_and_Kitchen_10000.jsonl",
            "meta_Sports_and_Outdoors_10000.jsonl",
            "meta_Toys_and_Games_10000.jsonl",
            "meta_Video_Games_10000.jsonl"
        ]
        
        logger.info(f"📂 Procesando {len(files_to_load)} archivos...")
        
        for filename in files_to_load:
            filepath = data_dir / filename
            if not filepath.exists():
                logger.warning(f"⚠️  Archivo no encontrado: {filename}")
                continue
            
            logger.info(f"   📄 {filename}...")
            count = 0
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if limit is not None and len(all_products) >= limit:
                            break
                        
                        try:
                            product = json.loads(line.strip())
                            if isinstance(product, dict):
                                # Añadir metadata del archivo
                                product['_source_file'] = filename
                                product['_source_line'] = count
                                
                                # Asegurar ID único
                                if 'asin' in product:
                                    product['id'] = product['asin']
                                elif 'id' not in product:
                                    product['id'] = f"{filename}_{count}"
                                
                                all_products.append(product)
                                count += 1
                                
                        except json.JSONDecodeError:
                            continue
                
                logger.info(f"   ✅ {count:,} productos")
                
            except Exception as e:
                logger.error(f"   ❌ Error: {e}")
        
        logger.info(f"📈 TOTAL: {len(all_products):,} productos raw")
        return all_products
    
    def _canonicalize_batch(self, raw_products, batch_size=5000):
        """Canoniza productos en lotes"""
        # CORRECCIÓN: Importación absoluta en lugar de relativa
        from src.data.canonicalizer import ProductCanonicalizer
        
        logger.info(f"🔧 Canonicalizando {len(raw_products):,} productos...")
        
        # Inicializar canonicalizer
        canonicalizer = ProductCanonicalizer(
            embedding_model="all-MiniLM-L6-v2"
        )
        
        # Procesar en lotes
        canonical_products = []
        total_batches = (len(raw_products) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(raw_products), batch_size):
            batch = raw_products[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            logger.info(f"   Lote {batch_num}/{total_batches} ({len(batch):,} productos)...")
            
            batch_canonical = canonicalizer.batch_canonicalize(batch)
            canonical_products.extend(batch_canonical)
            
            # Guardar progreso parcial en caché
            if self.cache_manager and batch_num % 2 == 0:
                temp_key = f"amazon_partial_batch_{batch_num}"
                self.cache_manager.save_canonical_products(
                    canonical_products, 
                    temp_key
                )
        
        logger.info(f"✅ {len(canonical_products):,} productos canonicalizados")
        return canonical_products
    
    def get_stats(self):
        """Obtiene estadísticas de los datos cargados"""
        stats = {
            'raw_count': len(self.raw_products),
            'canonical_count': len(self.canonical_products),
            'sources': {}
        }
        
        # Contar por categoría
        categories = {}
        for product in self.canonical_products:
            cat = getattr(product, 'category', 'unknown')
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        
        stats['categories'] = categories
        
        return stats


# Función de conveniencia para backward compatibility
def load_all_products_cached(force_reload=False):
    """Carga todos los productos usando caché (para scripts existentes)"""
    loader = OptimizedDataLoader(cache_enabled=True)
    return loader.load_all_with_cache(force_reload=force_reload)