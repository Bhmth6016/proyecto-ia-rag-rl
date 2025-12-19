"""
Versión simplificada del orquestador principal
"""
import yaml
import logging
from pathlib import Path
import numpy as np
import sys
import os
import json

# Configurar path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importaciones
try:
    from data.canonicalizer import ProductCanonicalizer
    from data.vector_store import VectorStore
    from query.query_understanding import QueryUnderstanding
    from ranking.ranking_engine import StaticRankingEngine
    from consistency_checker import ConsistencyChecker
except ImportError as e:
    logger.error(f"Error de importación: {e}")
    sys.exit(1)


class SimpleExperimentRunner:
    """Ejecuta experimento simplificado"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> dict:
        """Carga configuración"""
        config_file = Path(config_path)
        if not config_file.exists():
            logger.error(f"Archivo de configuración no encontrado: {config_path}")
            logger.info("Usando configuración por defecto")
            return {
                'experiment': {'name': 'demo', 'seed': 42},
                'embedding': {'model': 'all-MiniLM-L6-v2', 'dimension': 384},
                'evaluation': {'test_queries': ['test query']}
            }
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def run_demo(self):
        """Ejecuta demostración básica"""
        logger.info("🚀 INICIANDO DEMOSTRACIÓN SIMPLIFICADA")
        
        # 1. Verificación de consistencia
        self._run_consistency_check()
        
        # 2. Cargar datos de ejemplo
        products = self._load_sample_products()
        
        # 3. Canonizar productos
        canonicalizer = ProductCanonicalizer(
            embedding_model=self.config.get('embedding', {}).get('model', 'all-MiniLM-L6-v2')
        )
        
        canonical_products = canonicalizer.batch_canonicalize(products)
        logger.info(f"✅ Productos canonizados: {len(canonical_products)}")
        
        # 4. Construir índice vectorial
        vector_store = VectorStore(
            dimension=self.config.get('embedding', {}).get('dimension', 384)
        )
        vector_store.build_index(canonical_products)
        logger.info("✅ Índice vectorial construido")
        
        # 5. Query Understanding
        query_understanding = QueryUnderstanding()
        
        # 6. Ranking Engine
        baseline_weights = self.config.get('ranking', {}).get('baseline_weights', {})
        ranking_engine = StaticRankingEngine(weights=baseline_weights)
        
        # 7. Probar con queries de prueba
        test_queries = self.config.get('evaluation', {}).get('test_queries', ['test query'])
        
        logger.info("\n🔍 PROBANDO SISTEMA CON QUERIES:")
        logger.info("="*50)
        
        for query in test_queries:
            logger.info(f"\nQuery: '{query}'")
            
            # Analizar query
            analysis = query_understanding.analyze(query)
            logger.info(f"  Análisis: Categoría='{analysis.get('category')}', "
                       f"Intención='{analysis.get('intent')}'")
            
            # Generar embedding de query
            query_embedding = canonicalizer.embedding_model.encode(
                query, normalize_embeddings=True
            )
            
            # Búsqueda en vector store
            retrieved = vector_store.search(query_embedding, k=5)
            logger.info(f"  Productos recuperados: {len(retrieved)}")
            
            # Ranking
            ranked = ranking_engine.rank_products(
                query_embedding=query_embedding,
                query_category=analysis.get('category', 'General'),
                products=retrieved,
                top_k=3
            )
            
            logger.info(f"  Top 3 resultados:")
            for i, product in enumerate(ranked[:3], 1):
                price_str = f"${product.price:.2f}" if product.price else "N/A"
                rating_str = f"⭐{product.rating}" if product.rating else "Sin rating"
                logger.info(f"    {i}. {product.title} ({price_str}, {rating_str})")
        
        logger.info("\n" + "="*50)
        logger.info("✅ DEMOSTRACIÓN COMPLETADA")
        
        # Guardar resultados
        self._save_demo_results(canonical_products)
    
    def _run_consistency_check(self):
        """Verificación de consistencia científica"""
        logger.info("\n🔍 VERIFICACIÓN DE CONSISTENCIA CIENTÍFICA")
        checker = ConsistencyChecker()
        checker.check_all()
    
    def _load_sample_products(self):
        """Carga productos de ejemplo"""
        sample_path = Path("data/raw/sample_products.json")
        
        if sample_path.exists():
            with open(sample_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Crear datos mock si no existe
            logger.warning(f"Archivo de muestra no encontrado: {sample_path}")
            logger.info("Creando datos de ejemplo...")
            
            mock_products = []
            for i in range(10):
                mock_products.append({
                    "id": f"mock_prod_{i+1:03d}",
                    "title": f"Producto de ejemplo {i+1}",
                    "description": f"Descripción del producto de ejemplo {i+1}",
                    "price": 99.99 + i * 20,
                    "main_category": ["Electronics", "Books", "Sports"][i % 3],
                    "average_rating": 4.0 + (i % 5) * 0.2,
                    "rating_count": 100 * (i + 1)
                })
            
            # Guardar para futuras ejecuciones
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            with open(sample_path, 'w', encoding='utf-8') as f:
                json.dump(mock_products, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Datos de ejemplo creados en: {sample_path}")
            return mock_products
    
    def _save_demo_results(self, products):
        """Guarda resultados de la demostración"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Guardar información básica
        results = {
            "total_products": len(products),
            "products_with_price": sum(1 for p in products if p.price),
            "products_with_rating": sum(1 for p in products if p.rating),
            "categories": list(set(p.category for p in products))
        }
        
        results_path = results_dir / "demo_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Resultados guardados en: {results_path}")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sistema E-commerce RAG+RLHF - Demostración Simplificada"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/experiment.yaml',
        help='Archivo de configuración'
    )
    
    args = parser.parse_args()
    
    # Ejecutar demostración
    runner = SimpleExperimentRunner(args.config)
    runner.run_demo()


if __name__ == "__main__":
    main()