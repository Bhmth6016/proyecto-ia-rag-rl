# src/unified_system.py
"""
Sistema base RAG+RL - Versión simplificada para compatibilidad
"""
import yaml
import logging
from pathlib import Path
import sys
import os

# Añadir el directorio src al path para importaciones
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedRAGRLSystem:
    """Sistema base para compatibilidad con V2"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        
        # Componentes básicos
        self.canonical_products = []
        self.canonicalizer = None
        self.vector_store = None
        self.query_understanding = None
        self.feature_engineer = None
        self.rl_ranker = None
        self.baseline_ranker = None
        
        logger.info("Sistema base inicializado")
    
    def _load_config(self, config_path: str) -> dict:
        """Carga configuración"""
        config_file = Path(config_path)
        if not config_file.exists():
            # Configuración por defecto
            return {
                'embedding': {
                    'model': 'all-MiniLM-L6-v2',
                    'dimension': 384
                },
                'rlhf': {
                    'learning_rate': 0.3,
                    'match_rating_balance': 1.5
                }
            }
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def initialize_from_raw_all_files(self, limit=30000, batch_size=2000):
        """Inicializa sistema desde datos raw"""
        try:
            # Importar dinámicamente para evitar errores de importación
            from data.loader import load_raw_products
            from data.canonicalizer import ProductCanonicalizer
            from data.vector_store import ImmutableVectorStore
            from query.understanding import QueryUnderstanding
            from features.extractor import FeatureEngineer
            from ranking.baseline_ranker import BaselineRanker
            
            logger.info("Cargando productos de todos los archivos raw...")
            logger.info(f"Límite configurado: {limit} productos")
            
            # Cargar productos
            all_raw_products = load_raw_products(file_path=None, limit=limit)
            
            if not all_raw_products:
                logger.error("No se pudieron cargar productos")
                return False
            
            logger.info(f"Cargados {len(all_raw_products):,} productos")
            
            # Canonizar
            logger.info("Canonizando productos...")
            self.canonicalizer = ProductCanonicalizer(
                embedding_model=self.config['embedding']['model']
            )
            
            self.canonical_products = []
            total_batches = (len(all_raw_products) + batch_size - 1) // batch_size
            
            for i in range(0, len(all_raw_products), batch_size):
                batch = all_raw_products[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                logger.info(f"   Lote {batch_num}/{total_batches} ({len(batch)} productos)...")
                
                batch_canonical = self.canonicalizer.batch_canonicalize(batch)
                self.canonical_products.extend(batch_canonical)
                
                if batch_num % 5 == 0 or batch_num == total_batches:
                    logger.info(f"     Progreso: {len(self.canonical_products):,}/{len(all_raw_products):,} productos canonizados")
                
                del batch
            
            logger.info(f"Canonizados {len(self.canonical_products):,} productos")
            
            del all_raw_products
            
            # Construir vector store
            logger.info("Inicializando vector store...")
            self.vector_store = ImmutableVectorStore(
                dimension=self.config['embedding']['dimension']
            )
            self.vector_store.build_index(self.canonical_products)
            
            # Inicializar otros componentes
            logger.info("Inicializando componentes de ranking...")
            self.query_understanding = QueryUnderstanding()
            self.feature_engineer = FeatureEngineer()
            
            # Intentar crear RL ranker
            try:
                from ranking.rl_ranker_fixed import RLHFRankerFixed
                self.rl_ranker = RLHFRankerFixed(
                    learning_rate=self.config.get('rlhf', {}).get('learning_rate', 0.3),
                    match_rating_balance=self.config.get('rlhf', {}).get('match_rating_balance', 1.5)
                )
                logger.info("RL ranker creado")
            except ImportError as e:
                self.rl_ranker = None
                logger.warning(f"RLHFRankerFixed no disponible: {e}")
            
            self.baseline_ranker = BaselineRanker()
            
            logger.info(f"Sistema inicializado: {len(self.canonical_products):,} productos")
            
            logger.info(f"ESTADÍSTICAS DEL SISTEMA:")
            logger.info(f"   Productos canonizados: {len(self.canonical_products):,}")
            logger.info(f"   Dimensión embeddings: {self.config['embedding']['dimension']}")
            logger.info(f"   RL ranker disponible: {self.rl_ranker is not None}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_system_stats(self):
        """Obtiene estadísticas completas del sistema"""
        return {
            'canonical_products': len(self.canonical_products) if self.canonical_products else 0,
            'has_canonicalizer': self.canonicalizer is not None,
            'has_vector_store': self.vector_store is not None,
            'has_rl_ranker': self.rl_ranker is not None,
            'has_baseline_ranker': self.baseline_ranker is not None
        }