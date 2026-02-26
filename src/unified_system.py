# src/unified_system.py
import yaml
import logging
from pathlib import Path
import sys

current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedRAGRLSystem:

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.canonical_products = []
        self.canonicalizer = None
        self.vector_store = None
        self.query_understanding = None
        self.feature_engineer = None
        # rl_ranker eliminado — era RLHFRankerFixed, heurística no RLHF
        self.baseline_ranker = None
        logger.info("Sistema base inicializado")

    def _load_config(self, config_path: str) -> dict:
        config_file = Path(config_path)
        if not config_file.exists():
            return {
                'embedding': {'model': 'all-MiniLM-L6-v2', 'dimension': 384},
                'rlhf': {}
            }
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def initialize_from_raw_all_files(self, limit=300000, batch_size=2000):
        try:
            from data.loader import load_raw_products
            from data.canonicalizer import ProductCanonicalizer
            from data.vector_store import ImmutableVectorStore
            from query.understanding import QueryUnderstanding
            from features.extractor import FeatureEngineer
            from ranking.baseline_ranker import BaselineRanker

            logger.info(f"Cargando productos (límite: {limit})...")
            all_raw = load_raw_products(file_path=None, limit=limit)
            if not all_raw:
                return False

            self.canonicalizer = ProductCanonicalizer(
                embedding_model=self.config['embedding']['model']
            )
            self.canonical_products = []
            total_batches = (len(all_raw) + batch_size - 1) // batch_size

            for i in range(0, len(all_raw), batch_size):
                batch = all_raw[i:i + batch_size]
                bn = (i // batch_size) + 1
                logger.info(f"   Lote {bn}/{total_batches}...")
                self.canonical_products.extend(
                    self.canonicalizer.batch_canonicalize(batch)
                )
                del batch

            del all_raw

            self.vector_store = ImmutableVectorStore(
                dimension=self.config['embedding']['dimension']
            )
            self.vector_store.build_index(self.canonical_products)

            self.query_understanding = QueryUnderstanding()
            self.feature_engineer = FeatureEngineer()
            self.baseline_ranker = BaselineRanker()

            logger.info(f"Sistema inicializado: {len(self.canonical_products):,} productos")
            return True

        except Exception as e:
            logger.error(f"Error inicializando: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_system_stats(self):
        return {
            'canonical_products': len(self.canonical_products) if self.canonical_products else 0,
            'has_canonicalizer': self.canonicalizer is not None,
            'has_vector_store': self.vector_store is not None,
            'has_baseline_ranker': self.baseline_ranker is not None,
        }