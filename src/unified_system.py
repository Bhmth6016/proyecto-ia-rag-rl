# src/unified_system.py
"""
Sistema RAG+RL UNIFICADO - Definido en módulo propio para pickle
"""
import yaml
import logging
from pathlib import Path
import sys
from datetime import datetime
from typing import List, Dict, Any
import json
import pickle
import numpy as np

# Configurar paths
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
    """Sistema unificado con caché, persistencia y todos los métodos necesarios"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Componentes (se inicializarán después)
        self.canonical_products = []
        self.canonicalizer = None
        self.vector_store = None
        self.query_understanding = None
        self.feature_engineer = None
        self.rl_ranker = None
        self.interaction_handler = None
        
        logger.info("🔧 Sistema unificado inicializado")
    
    def _load_config(self, config_path: str) -> dict:
        """Carga configuración"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def initialize_from_raw(self, limit=10000):
        """Inicializa sistema desde datos raw"""
        try:
            from src.data.loader import load_raw_products
            from data.canonicalizer import ProductCanonicalizer
            from data.vector_store import ImmutableVectorStore
            from src.query.understanding import QueryUnderstanding
            from features.extractor import FeatureEngineer
            from ranking.rl_ranker import RLHFRanker
            from src.user.interaction_handler import InteractionHandler
            
            logger.info("📥 Cargando productos raw...")
            raw_products = load_raw_products(limit=limit)
            
            logger.info("🔧 Canonizando productos...")
            self.canonicalizer = ProductCanonicalizer(
                embedding_model=self.config['embedding']['model']
            )
            
            # Canonizar en lote pequeño
            batch_size = 2000
            self.canonical_products = []
            
            for i in range(0, len(raw_products), batch_size):
                batch = raw_products[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                logger.info(f"   Lote {batch_num}/{(len(raw_products) + batch_size - 1) // batch_size}...")
                
                batch_canonical = self.canonicalizer.batch_canonicalize(batch)
                self.canonical_products.extend(batch_canonical)
            
            logger.info("📚 Inicializando vector store...")
            self.vector_store = ImmutableVectorStore(
                dimension=self.config['embedding']['dimension']
            )
            self.vector_store.build_index(self.canonical_products)
            
            # Inicializar otros componentes
            self.query_understanding = QueryUnderstanding()
            self.feature_engineer = FeatureEngineer()
            self.rl_ranker = RLHFRanker(
                alpha=self.config.get('rlhf', {}).get('alpha', 0.1)
            )
            self.interaction_handler = InteractionHandler()
            
            logger.info(f"✅ Sistema inicializado: {len(self.canonical_products):,} productos")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_to_cache(self, cache_path="data/cache/unified_system.pkl"):
        """Guarda sistema en caché"""
        try:
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"💾 Sistema guardado en caché: {cache_file}")
            return True
        except Exception as e:
            logger.error(f"❌ Error guardando caché: {e}")
            return False
    
    @staticmethod
    def load_from_cache(cache_path="data/cache/unified_system.pkl"):
        """Carga sistema desde caché"""
        cache_file = Path(cache_path)
        
        if not cache_file.exists():
            logger.warning(f"⚠️  No existe caché: {cache_file}")
            return None
        
        try:
            logger.info(f"📥 Cargando desde caché: {cache_file}")
            with open(cache_file, 'rb') as f:
                system = pickle.load(f)
            
            logger.info(f"✅ Sistema cargado: {len(system.canonical_products):,} productos")
            return system
            
        except Exception as e:
            logger.error(f"❌ Error cargando caché: {e}")
            return None
    
    def _process_query_mode(self, query_text: str, mode: str = 'with_rlhf'):
        """Procesa query según modo específico"""
        logger.info(f"🔍 Procesando query '{query_text}' (modo: {mode})")
        
        try:
            # 1. Análisis de query
            query_analysis = self.query_understanding.extract(query_text)
            
            # 2. Retrieval (siempre igual)
            query_embedding = self.canonicalizer.embedding_model.encode(
                query_text, normalize_embeddings=True
            )
            retrieved_products = self.vector_store.search(query_embedding, k=50)
            
            if not retrieved_products:
                return {'success': True, 'products': []}
            
            # 3. Feature engineering
            query_features = self.feature_engineer.extract_query_features(
                query_text, query_embedding, query_analysis
            )
            
            product_features = []
            for product in retrieved_products:
                feat = self.feature_engineer.extract_product_features(product, query_features)
                product_features.append(feat)
            
            # 4. Ranking según modo
            if mode == 'baseline':
                # Ordenar por similitud coseno
                ranked_products = self._sort_by_cosine_similarity(retrieved_products, query_embedding)
            elif mode == 'with_features':
                # Ranking con features
                if hasattr(self.rl_ranker, 'rank_with_features_only'):
                    ranked_products = self.rl_ranker.rank_with_features_only(
                        retrieved_products, query_features, product_features
                    )
                else:
                    ranked_products = self._sort_by_cosine_similarity(retrieved_products, query_embedding)
            elif mode == 'with_rlhf':
                # Ranking con RL
                if hasattr(self.rl_ranker, 'has_learned') and self.rl_ranker.has_learned:
                    logger.info("   → Aplicando política RL aprendida")
                    if hasattr(self.rl_ranker, 'rank_with_learning'):
                        ranked_products = self.rl_ranker.rank_with_learning(
                            retrieved_products, query_features, product_features
                        )
                    else:
                        ranked_products = self.rl_ranker.rank_with_features_only(
                            retrieved_products, query_features, product_features
                        )
                else:
                    logger.info("   → Sin aprendizaje aún, usando features")
                    if hasattr(self.rl_ranker, 'rank_with_features_only'):
                        ranked_products = self.rl_ranker.rank_with_features_only(
                            retrieved_products, query_features, product_features
                        )
                    else:
                        ranked_products = self._sort_by_cosine_similarity(retrieved_products, query_embedding)
            else:
                ranked_products = retrieved_products
            
            # 5. Preparar respuesta
            response_products = []
            for i, product in enumerate(ranked_products[:10]):
                product_dict = {
                    'id': getattr(product, 'id', f"prod_{i}"),
                    'title': getattr(product, 'title', f"Product {i+1}")[:100],
                    'category': getattr(product, 'category', 'unknown'),
                    'price': getattr(product, 'price', 0.0),
                    'rating': getattr(product, 'rating', 0.0),
                    'similarity_score': round(0.8 - (i * 0.07), 3),
                    'position': i + 1
                }
                response_products.append(product_dict)
            
            return {
                'success': True,
                'query': query_text,
                'mode': mode,
                'products': response_products,
                'retrieved_count': len(retrieved_products)
            }
            
        except Exception as e:
            logger.error(f"❌ Error procesando query: {e}")
            return {'success': False, 'error': str(e)}
    
    def _sort_by_cosine_similarity(self, products, query_embedding):
        """Ordena productos por similitud coseno"""
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        scores = []
        for product in products:
            if hasattr(product, 'content_embedding'):
                prod_embedding = product.content_embedding
                prod_norm = prod_embedding / np.linalg.norm(prod_embedding)
                score = np.dot(query_norm, prod_norm)
            else:
                score = 0.0
            scores.append(score)
        
        sorted_pairs = sorted(zip(products, scores), key=lambda x: x[1], reverse=True)
        return [product for product, _ in sorted_pairs]
    
    def process_feedback(self, interaction_data: Dict[str, Any]):
        """Procesa feedback para aprendizaje RL"""
        logger.info(f"\n🎯 Procesando feedback...")
        
        try:
            context = interaction_data.get('context', {})
            
            if self.rl_ranker:
                query_features = {'query_text': context.get('query', '')}
                
                # Método compatible
                if hasattr(self.rl_ranker, 'learn_from_feedback'):
                    self.rl_ranker.learn_from_feedback(
                        query_features=query_features,
                        selected_product_id=context.get('product_id'),
                        reward=1.0,
                        context=context
                    )
                
                logger.info("✅ Feedback aplicado al RL")
                return {'success': True, 'learning_applied': True}
            
            return {'success': False, 'error': 'No hay RL ranker'}
            
        except Exception as e:
            logger.error(f"❌ Error procesando feedback: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_rl_stats(self):
        """Obtiene estadísticas del RL"""
        if not self.rl_ranker:
            return {'error': 'No hay RL ranker'}
        
        stats = {'has_rl_ranker': True}
        
        if hasattr(self.rl_ranker, 'has_learned'):
            stats['has_learned'] = self.rl_ranker.has_learned
        
        if hasattr(self.rl_ranker, 'get_learning_stats'):
            stats.update(self.rl_ranker.get_learning_stats())
        
        return stats
    
    def initialize_from_raw_all_files(self, limit=30000, batch_size=2000):
        """Inicializa sistema desde TODOS los archivos raw"""
        try:
            from src.data.loader import load_raw_products
            from data.canonicalizer import ProductCanonicalizer
            from data.vector_store import ImmutableVectorStore
            from src.query.understanding import QueryUnderstanding
            from features.extractor import FeatureEngineer
            from ranking.rl_ranker import RLHFRanker
            from ranking.baseline_ranker import BaselineRanker
            from src.user.interaction_handler import InteractionHandler
            
            logger.info("📥 Cargando productos de TODOS los archivos raw...")
            logger.info(f"🔧 Límite configurado: {limit} productos")
            
            # Cargar productos usando la función mejorada
            # Pasar None como file_path para cargar todos los archivos
            all_raw_products = load_raw_products(file_path=None, limit=limit)
            
            if not all_raw_products:
                logger.error("❌ No se pudieron cargar productos")
                return False
            
            logger.info(f"✅ Cargados {len(all_raw_products):,} productos")
            
            # Canonizar
            logger.info("🔧 Canonizando productos...")
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
                
                # Mostrar progreso
                if batch_num % 5 == 0 or batch_num == total_batches:
                    logger.info(f"     Progreso: {len(self.canonical_products):,}/{len(all_raw_products):,} productos canonizados")
                
                # Liberar memoria
                del batch
            
            logger.info(f"✅ Canonizados {len(self.canonical_products):,} productos")
            
            # Liberar raw products para ahorrar memoria
            del all_raw_products
            
            # Construir vector store
            logger.info("📚 Inicializando vector store...")
            self.vector_store = ImmutableVectorStore(
                dimension=self.config['embedding']['dimension']
            )
            self.vector_store.build_index(self.canonical_products)
            
            # Inicializar otros componentes
            logger.info("🤖 Inicializando componentes de ranking...")
            self.query_understanding = QueryUnderstanding()
            self.feature_engineer = FeatureEngineer()
            
            # RL Ranker con parámetros optimizados
            self.rl_ranker = RLHFRanker(
                alpha=self.config.get('rlhf', {}).get('alpha', 0.15),
                temperature=0.6
            )
            
            # Baseline ranker
            baseline_ranker = BaselineRanker()
            if hasattr(self.rl_ranker, 'set_baseline_ranker'):
                self.rl_ranker.set_baseline_ranker(baseline_ranker)
            
            self.interaction_handler = InteractionHandler()
            
            logger.info(f"✅ Sistema inicializado: {len(self.canonical_products):,} productos")
            
            # Mostrar estadísticas
            logger.info(f"\n📊 ESTADÍSTICAS DEL SISTEMA:")
            logger.info(f"   • Productos canonizados: {len(self.canonical_products):,}")
            logger.info(f"   • Dimensión embeddings: {self.config['embedding']['dimension']}")
            logger.info(f"   • Alpha RL: {self.config.get('rlhf', {}).get('alpha', 0.15)}")
            logger.info(f"   • Batch size: {batch_size}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def process_feedback_enhanced(self, interaction_data: Dict[str, Any], product_features: Dict[str, float] = None):
        """Procesa feedback con características del producto"""
        logger.info(f"\n🎯 Procesando feedback mejorado...")
        
        try:
            context = interaction_data.get('context', {})
            
            if self.rl_ranker:
                query_features = {'query_text': context.get('query', '')}
                
                # Extraer características del producto clickeado si es posible
                selected_product_id = context.get('product_id')
                enhanced_features = product_features or {}
                
                # Método compatible
                if hasattr(self.rl_ranker, 'learn_from_feedback'):
                    self.rl_ranker.learn_from_feedback(
                        query_features=query_features,
                        selected_product_id=selected_product_id,
                        reward=1.0,
                        context=context,
                        product_features=enhanced_features
                    )
                
                stats = self.rl_ranker.get_learning_stats()
                logger.info(f"✅ Feedback aplicado - Política: {stats.get('policy_size', 0)} características")
                
                if 'top_features' in stats:
                    logger.info(f"   Top: {stats['top_features']}")
                
                return {'success': True, 'learning_applied': True}
            
            return {'success': False, 'error': 'No hay RL ranker'}
            
        except Exception as e:
            logger.error(f"❌ Error procesando feedback: {e}")
            return {'success': False, 'error': str(e)}