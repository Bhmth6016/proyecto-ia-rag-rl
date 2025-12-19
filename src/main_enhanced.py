# src/main_enhanced.py
"""
Orquestador principal mejorado - Implementa todas las mejoras
"""
import yaml
import logging
from pathlib import Path
import numpy as np
import sys
from datetime import datetime
import os

# Configurar path absoluto
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importaciones absolutas
try:
    from data.canonicalizer import ProductCanonicalizer
    from data.vector_store import VectorStore
    from query.query_understanding import QueryUnderstanding
    from ranking.ranking_engine import StaticRankingEngine
    from ranking.rlhf_agent import RLHFAgent
    from evaluator.evaluator import ScientificEvaluator
    from consistency_checker import ConsistencyChecker
except ImportError as e:
    logger.error(f"Error de importación: {e}")
    logger.info("Asegúrate de que todos los módulos estén en src/")
    sys.exit(1)


class EnhancedExperimentRunner:
    """Ejecuta experimento completo con todas las mejoras"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Crear directorios de resultados
        self._create_results_dirs()
        
    def _load_config(self, config_path: str) -> dict:
        """Carga configuración"""
        config_file = Path(config_path)
        if not config_file.exists():
            # Buscar en config/ si no se encuentra
            config_file = Path("config") / Path(config_path).name
            if not config_file.exists():
                config_file = Path("config/experiment.yaml")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_results_dirs(self):
        """Crea directorios para resultados"""
        dirs = ["results", "results/plots", "results/logs", "results/checkpoints"]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def run_full_experiment(self):
        """Ejecuta experimento completo"""
        logger.info("🚀 INICIANDO EXPERIMENTO MEJORADO")
        logger.info(f"📋 ID del experimento: {self.experiment_id}")
        
        # 1. Verificación de consistencia
        self._run_consistency_check()
        
        # 2. Inicialización de componentes
        components = self._initialize_components()
        
        # 3. Demostración simple (sin validaciones complejas por ahora)
        self._run_demo_experiment(components)
        
        logger.info("✅ EXPERIMENTO DEMO COMPLETADO")
        logger.info(f"📁 Puedes encontrar resultados en: results/")
    
    def _run_consistency_check(self):
        """Verificación de consistencia científica"""
        logger.info("\n" + "="*60)
        logger.info("1️⃣ VERIFICACIÓN DE CONSISTENCIA CIENTÍFICA")
        logger.info("="*60)
        
        checker = ConsistencyChecker()
        if not checker.check_all():
            logger.error("❌ La verificación de consistencia falló")
            # Continuar de todos modos para demo
            logger.warning("⚠️ Continuando en modo demo...")
    
    def _initialize_components(self) -> Dict:
        """Inicializa todos los componentes"""
        logger.info("\n2️⃣ INICIALIZANDO COMPONENTES")
        
        # Canonicalizer
        canonicalizer = ProductCanonicalizer(
            embedding_model=self.config.get('embedding', {}).get('model', 'all-MiniLM-L6-v2')
        )
        
        # Vector Store (simulado para demo)
        vector_store = VectorStore(
            dimension=self.config.get('embedding', {}).get('dimension', 384)
        )
        
        # Query Understanding
        query_understanding = QueryUnderstanding()
        
        # Ranking Engines
        baseline_weights = self.config.get('ranking', {}).get('baseline_weights', {})
        baseline_engine = StaticRankingEngine(weights=baseline_weights)
        
        # RLHF Agent
        def simple_feature_extractor(query_features, product):
            """Extractor de características simple para demo"""
            return {
                "price_available": 1.0 if product.price else 0.0,
                "has_rating": 1.0 if product.rating else 0.0,
                "rating_value": product.rating if product.rating else 0.0
            }
        
        rlhf_agent = RLHFAgent(
            feature_extractor=simple_feature_extractor,
            alpha=self.config.get('rlhf', {}).get('alpha', 0.1)
        )
        
        # Evaluator
        evaluator = ScientificEvaluator(
            seed=self.config.get('experiment', {}).get('seed', 42)
        )
        
        logger.info("✅ Componentes inicializados correctamente")
        
        return {
            "canonicalizer": canonicalizer,
            "vector_store": vector_store,
            "query_understanding": query_understanding,
            "baseline_engine": baseline_engine,
            "rlhf_agent": rlhf_agent,
            "evaluator": evaluator
        }
    
    def _run_demo_experiment(self, components: Dict):
        """Ejecuta demostración simple del sistema"""
        logger.info("\n3️⃣ EJECUTANDO DEMOSTRACIÓN")
        logger.info("-"*40)
        
        # Datos de prueba simulados
        test_queries = self.config.get('evaluation', {}).get('test_queries', [
            "nintendo switch games",
            "laptop for programming"
        ])
        
        logger.info(f"Queries de prueba: {test_queries}")
        
        # Simular productos de prueba
        class MockProduct:
            def __init__(self, id, title, price=None, rating=None):
                self.id = id
                self.title = title
                self.price = price
                self.rating = rating
                self.description = f"Description of {title}"
                self.category = "Electronics"
                self.rating_count = 100 if rating else 0
                self.title_embedding = np.random.randn(384)
                self.content_embedding = np.random.randn(384)
        
        # Crear productos mock
        mock_products = [
            MockProduct("prod1", "Nintendo Switch OLED", 299.99, 4.5),
            MockProduct("prod2", "Nintendo Switch Lite", 199.99, 4.3),
            MockProduct("prod3", "Gaming Laptop", 1299.99, 4.7),
            MockProduct("prod4", "Programming Laptop", 999.99, 4.6),
            MockProduct("prod5", "Wireless Headphones", 149.99, 4.4),
        ]
        
        # Probar cada query
        for query in test_queries[:2]:  # Solo las primeras 2 para demo
            logger.info(f"\n🔍 Procesando query: '{query}'")
            
            # Analizar query
            analysis = components["query_understanding"].analyze(query)
            logger.info(f"  Análisis: {analysis.get('intent', 'search')}, Categoría: {analysis.get('category', 'General')}")
            
            # Embedding de query simulado
            query_embedding = np.random.randn(384)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Ranking baseline
            baseline_results = components["baseline_engine"].rank_products(
                query_embedding=query_embedding,
                query_category=analysis.get('category', 'General'),
                products=mock_products,
                top_k=3
            )
            
            logger.info(f"  Baseline top-3:")
            for i, product in enumerate(baseline_results[:3], 1):
                logger.info(f"    {i}. {product.title} (${product.price if product.price else 'N/A'})")
            
            # RLHF (simulado)
            logger.info("  RLHF: Agente inicializado y listo para aprendizaje")
        
        # Guardar configuración
        config_path = f"results/{self.experiment_id}_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        logger.info(f"\n📁 Configuración guardada en: {config_path}")
        logger.info("\n🎯 Para ejecutar experimentos completos, implementa:")
        logger.info("   - validation/rlhf_learning_validator.py")
        logger.info("   - evaluator/metrics_calculator.py")
        logger.info("   - experiments/ablation_study.py")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sistema E-commerce RAG+RLHF - Experimento Mejorado"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/experiment.yaml',
        help='Archivo de configuración'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Ejecutar solo demostración básica'
    )
    
    args = parser.parse_args()
    
    # Ejecutar experimento
    runner = EnhancedExperimentRunner(args.config)
    runner.run_full_experiment()


if __name__ == "__main__":
    main()