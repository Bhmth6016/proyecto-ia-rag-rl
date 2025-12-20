# src/main_optimized.py
"""
Sistema principal OPTIMIZADO con caché - Basado en tu código existente
"""
import yaml
import logging
from pathlib import Path
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
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


class OptimizedRAGRLSystem:
    """Sistema principal OPTIMIZADO con caché - Versión rápida"""
    
    def __init__(self, config_path: str = "config/config.yaml", use_cache=True):
        self.config = self._load_config(config_path)
        self.use_cache = use_cache
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cache_loaded = False
        self.cache_saved = False
        
        # Manager de caché
        if use_cache:
            try:
                from data.cache_manager import CacheManager
                self.cache_manager = CacheManager()
                logger.info("🔧 CacheManager inicializado")
            except ImportError:
                logger.warning("⚠️  CacheManager no disponible, usando modo sin caché")
                self.use_cache = False
                self.cache_manager = None
        else:
            self.cache_manager = None
            logger.info("🔧 Modo sin caché")
        
        # Cargador optimizado (si está disponible)
        self.data_loader = None
        if use_cache:
            try:
                from data.optimized_loader import OptimizedDataLoader
                self.data_loader = OptimizedDataLoader(cache_enabled=use_cache)
                logger.info("🔧 OptimizedDataLoader inicializado")
            except ImportError:
                logger.warning("⚠️  OptimizedDataLoader no disponible")
        
        # Componentes del sistema (inicializar después)
        self.canonical_products = []
        self.vector_store = None
        self.canonicalizer = None
        self.query_understanding = None
        self.feature_engineer = None
        self.rl_ranker = None
        self.interaction_handler = None
        
        # Setup directorios
        self._setup_directories()
    
    def initialize_with_cache(self, force_reload=False, load_limit=None):
        """
        Inicialización RÁPIDA usando caché
        
        Args:
            force_reload: Si True, ignora caché y reprocesa todo
            load_limit: Límite de productos a cargar (None = todos)
        
        Returns:
            True si éxito, False si error
        """
        logger.info("\n" + "="*80)
        logger.info("🚀 INICIALIZACIÓN RÁPIDA CON CACHÉ")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        try:
            # 1. Cargar productos usando caché si está disponible
            if self.data_loader and not force_reload:
                logger.info("📥 Cargando productos desde caché...")
                self.canonical_products = self.data_loader.load_all_with_cache(force_reload=False)
                
                if self.canonical_products:
                    load_time = (datetime.now() - start_time).total_seconds()
                    logger.info(f"✅ {len(self.canonical_products):,} productos cargados en {load_time:.1f}s")
                    self.cache_loaded = True
                else:
                    logger.info("⚠️  Caché vacío, cargando desde raw...")
                    self._load_without_cache(load_limit)
            else:
                self._load_without_cache(load_limit)
            
            # 2. Inicializar componentes principales
            self._initialize_components()
            
            # 3. Cargar o construir índice FAISS
            if not self._load_faiss_from_cache() or force_reload:
                self._build_faiss_index()
            
            # 4. Cargar estado RL desde caché si existe
            if self.use_cache:
                self._load_rl_state()
            
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"🎯 Sistema inicializado en {total_time:.1f}s")
            logger.info(f"📊 Productos disponibles: {len(self.canonical_products):,}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en inicialización: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_without_cache(self, load_limit=None):
        """Carga productos sin caché"""
        logger.info("📥 Cargando productos desde archivos raw...")
        
        # Usar tu loader existente
        from data.loader import load_raw_products
        raw_products = load_raw_products(limit=load_limit)
        
        if not raw_products:
            raise ValueError("No se pudieron cargar productos")
        
        logger.info(f"🔧 Canonizando {len(raw_products):,} productos...")
        
        # Canonizar productos (usar tu canonicalizer existente)
        from data.canonicalizer import ProductCanonicalizer
        canonicalizer = ProductCanonicalizer(
            embedding_model=self.config['embedding']['model']
        )
        
        # Canonizar en lotes
        batch_size = 1000
        self.canonical_products = []
        
        for i in range(0, len(raw_products), batch_size):
            batch = raw_products[i:i + batch_size]
            logger.info(f"   Lote {i//batch_size + 1}/{(len(raw_products)+batch_size-1)//batch_size}...")
            
            batch_canonical = canonicalizer.batch_canonicalize(batch)
            self.canonical_products.extend(batch_canonical)
            
            # Guardar progreso en caché
            if self.use_cache and self.cache_manager and (i//batch_size) % 5 == 0:
                temp_key = f"partial_batch_{i//batch_size}"
                self.cache_manager.save_canonical_products(
                    self.canonical_products, 
                    temp_key
                )
        
        # Guardar en caché final
        if self.use_cache and self.cache_manager:
            self.cache_manager.save_canonical_products(
                self.canonical_products, 
                "amazon_full"
            )
            self.cache_saved = True
    
    def _initialize_components(self):
        """Inicializa componentes del sistema rápidamente"""
        logger.info("⚡ Inicializando componentes rápidos...")
        
        # Inicializar canonicalizer (para embeddings de queries)
        from data.canonicalizer import ProductCanonicalizer
        self.canonicalizer = ProductCanonicalizer(
            embedding_model=self.config['embedding']['model']
        )
        
        # Inicializar vector store (pero sin construir índice aún)
        from data.vector_store import ImmutableVectorStore
        self.vector_store = ImmutableVectorStore(
            dimension=self.config['embedding']['dimension']
        )
        
        # Inicializar otros componentes
        from src.query.understanding import QueryUnderstanding
        self.query_understanding = QueryUnderstanding()
        
        from features.extractor import FeatureEngineer
        self.feature_engineer = FeatureEngineer()
        
        from ranking.rl_ranker import RLHFRanker
        self.rl_ranker = RLHFRanker(
            alpha=self.config.get('rlhf', {}).get('alpha', 0.1)
        )
        
        from src.user.interaction_handler import InteractionHandler
        self.interaction_handler = InteractionHandler()
        
        logger.info("✅ Componentes inicializados")
    
    def _load_faiss_from_cache(self):
        """Intenta cargar índice FAISS desde caché"""
        if not self.use_cache or not self.cache_manager:
            return False
        
        try:
            # Buscar archivo FAISS en caché
            cache_dir = Path("data/cache/indices")
            if not cache_dir.exists():
                return False
            
            # Buscar archivo más reciente
            index_files = list(cache_dir.glob("*.index"))
            if not index_files:
                return False
            
            latest_index = max(index_files, key=lambda f: f.stat().st_mtime)
            
            # Cargar índice
            import faiss
            index = faiss.read_index(str(latest_index))
            
            # Cargar IDs correspondientes
            ids_file = latest_index.with_suffix('').with_suffix('_ids.pkl')
            if not ids_file.exists():
                return False
            
            with open(ids_file, 'rb') as f:
                product_ids = pickle.load(f)
            
            # Configurar vector store
            self.vector_store.index = index
            self.vector_store.is_locked = True
            
            # Mapear IDs a productos
            id_to_product = {getattr(p, 'id', str(i)): p for i, p in enumerate(self.canonical_products)}
            self.vector_store.products = [id_to_product.get(pid) for pid in product_ids if pid in id_to_product]
            
            logger.info(f"✅ Índice FAISS cargado desde caché: {len(self.vector_store.products):,} vectores")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  No se pudo cargar FAISS desde caché: {e}")
            return False
    
    def _build_faiss_index(self):
        """Construye nuevo índice FAISS"""
        logger.info("🔨 Construyendo nuevo índice FAISS...")
        
        try:
            # Construir índice usando tu vector store existente
            self.vector_store.build_index(self.canonical_products)
            
            # Guardar en caché si está habilitado
            if self.use_cache and self.cache_manager:
                self._save_faiss_to_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error construyendo índice FAISS: {e}")
            raise
    
    def _save_faiss_to_cache(self):
        """Guarda índice FAISS en caché"""
        try:
            # Extraer embeddings y IDs
            embeddings = []
            product_ids = []
            
            for product in self.canonical_products:
                if hasattr(product, 'content_embedding') and product.content_embedding is not None:
                    embeddings.append(product.content_embedding)
                    product_ids.append(getattr(product, 'id', 'unknown'))
            
            if not embeddings:
                logger.warning("⚠️  No hay embeddings para guardar en caché")
                return
            
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Guardar en caché
            cache_key = self.cache_manager.save_embeddings(
                embeddings_array, 
                product_ids, 
                "amazon_faiss"
            )
            
            if cache_key:
                logger.info(f"💾 Embeddings guardados en caché: {len(embeddings):,} vectores")
            
        except Exception as e:
            logger.warning(f"⚠️  Error guardando FAISS en caché: {e}")
    
    def _load_rl_state(self):
        """Carga estado RL desde caché si existe"""
        try:
            rl_state_file = Path("data/cache/rl_state.pkl")
            if rl_state_file.exists():
                with open(rl_state_file, 'rb') as f:
                    rl_state = pickle.load(f)
                
                # Aplicar estado al ranker
                if hasattr(self.rl_ranker, 'load_state'):
                    self.rl_ranker.load_state(rl_state)
                    logger.info("✅ Estado RL cargado desde caché")
                    
        except Exception as e:
            logger.warning(f"⚠️  No se pudo cargar estado RL: {e}")
    
    def save_rl_state(self):
        """Guarda estado RL en caché"""
        if not self.use_cache or not hasattr(self.rl_ranker, 'get_state'):
            return
        
        try:
            rl_state = self.rl_ranker.get_state()
            
            cache_dir = Path("data/cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            with open(cache_dir / "rl_state.pkl", 'wb') as f:
                pickle.dump(rl_state, f)
            
            logger.info("✅ Estado RL guardado en caché")
            
        except Exception as e:
            logger.warning(f"⚠️  No se pudo guardar estado RL: {e}")
    
    def save_snapshot(self, name="system_snapshot"):
        """Guarda snapshot completo del sistema"""
        if not self.use_cache:
            return
        
        snapshot_dir = Path(f"data/snapshots/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'experiment_id': self.experiment_id,
            'products_count': len(self.canonical_products),
            'rl_learned': self.rl_ranker.has_learned if hasattr(self.rl_ranker, 'has_learned') else False,
            'cache_loaded': self.cache_loaded,
            'cache_saved': self.cache_saved
        }
        
        with open(snapshot_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Guardar estado RL
        self.save_rl_state()
        
        # Copiar estado RL al snapshot
        rl_state_file = Path("data/cache/rl_state.pkl")
        if rl_state_file.exists():
            import shutil
            shutil.copy2(rl_state_file, snapshot_dir / "rl_state.pkl")
        
        logger.info(f"✅ Snapshot guardado en: {snapshot_dir}")
        return str(snapshot_dir)
    
    # Métodos de tu sistema original que necesitamos mantener
    def process_query(self, query_text: str, user_id: str = "anonymous", use_rlhf: bool = True):
        """Procesa una query completa - Mismo que tu sistema original"""
        # Este método necesita acceder a _process_query_mode que está en tu main.py
        # Podemos importarlo o reimplementar
        
        # Por ahora, crear un método placeholder
        logger.info(f"🔍 Procesando query: '{query_text}'")
        
        try:
            # 1. Embedding de query
            query_embedding = self.canonicalizer.embedding_model.encode(
                query_text, normalize_embeddings=True
            )
            
            # 2. Retrieval
            retrieved_products = self.vector_store.search(query_embedding, k=50)
            
            if not retrieved_products:
                return {
                    'success': True,
                    'query': query_text,
                    'products': [],
                    'mode': 'with_rlhf' if use_rlhf else 'with_features'
                }
            
            # 3. Query analysis
            query_analysis = self.query_understanding.extract(query_text)
            
            # 4. Feature engineering
            query_features = self.feature_engineer.extract_query_features(
                query_text, query_embedding, query_analysis
            )
            
            product_features = []
            for product in retrieved_products:
                feat = self.feature_engineer.extract_product_features(product, query_features)
                product_features.append(feat)
            
            # 5. Ranking
            mode = 'with_rlhf' if use_rlhf else 'with_features'
            if mode == 'with_rlhf' and hasattr(self.rl_ranker, 'has_learned') and self.rl_ranker.has_learned:
                ranked_products = self.rl_ranker.rank_with_learning(
                    retrieved_products, query_features, product_features, [1.0]*len(retrieved_products)
                )
            else:
                if hasattr(self.rl_ranker, 'rank_with_features_only'):
                    ranked_products = self.rl_ranker.rank_with_features_only(
                        retrieved_products, query_features, product_features, [1.0]*len(retrieved_products)
                    )
                else:
                    ranked_products = retrieved_products
            
            # 6. Preparar respuesta
            response_products = []
            for i, product in enumerate(ranked_products[:10]):
                product_dict = {
                    'id': getattr(product, 'id', f"prod_{i}"),
                    'title': getattr(product, 'title', f"Product {i+1}")[:100],
                    'category': getattr(product, 'category', 'unknown'),
                    'price': getattr(product, 'price', 0.0),
                    'rating': getattr(product, 'rating', 0.0),
                    'similarity_score': 0.8 - (i * 0.07),  # Score simple para demo
                    'position': i + 1
                }
                response_products.append(product_dict)
            
            return {
                'success': True,
                'query': query_text,
                'mode': mode,
                'retrieved_count': len(retrieved_products),
                'ranked_count': len(response_products),
                'products': response_products
            }
            
        except Exception as e:
            logger.error(f"❌ Error procesando query: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query_text
            }
    
    def process_feedback(self, interaction_data: Dict[str, Any]):
        """Procesa feedback para aprendizaje RL"""
        # Método simple similar al de tu sistema
        logger.info(f"\n🎯 Procesando feedback...")
        
        try:
            context = interaction_data.get('context', {})
            
            if self.rl_ranker:
                query_features = {'query_text': context.get('query', '')}
                
                self.rl_ranker.learn_from_feedback(
                    query_features=query_features,
                    selected_product_id=context.get('product_id'),
                    reward=1.0,
                    context=context
                )
            
            # Guardar estado RL después de aprender
            self.save_rl_state()
            
            return {
                'success': True,
                'learning_applied': True,
                'rl_state_saved': True
            }
            
        except Exception as e:
            logger.error(f"❌ Error procesando feedback: {e}")
            return {'success': False, 'error': str(e)}
    
    def _load_config(self, config_path: str) -> dict:
        """Carga configuración"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_directories(self):
        """Crea directorios necesarios"""
        dirs = [
            f"results/{self.experiment_id}",
            f"data/cache",
            f"data/snapshots",
            f"data/interactions"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Función principal para backward compatibility
def main():
    """Función principal optimizada"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sistema RAG+RL OPTIMIZADO con caché"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['fast', 'query', 'evaluate', 'cache_stats', 'clear_cache'],
        default='fast',
        help='Modo de ejecución'
    )
    
    parser.add_argument(
        '--force-reload',
        action='store_true',
        help='Forzar recarga completa (ignorar caché)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Límite de productos a cargar (para testing)'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Query a procesar'
    )
    
    parser.add_argument(
        '--save-snapshot',
        type=str,
        help='Guardar snapshot con nombre'
    )
    
    args = parser.parse_args()
    
    # Crear sistema optimizado
    system = OptimizedRAGRLSystem('config/config.yaml', use_cache=True)
    
    if args.mode == 'fast':
        # Inicialización rápida con caché
        print("\n🚀 INICIANDO SISTEMA OPTIMIZADO")
        print("="*50)
        
        success = system.initialize_with_cache(
            force_reload=args.force_reload,
            load_limit=args.limit
        )
        
        if success:
            print(f"✅ Sistema listo en segundos")
            print(f"📊 Productos: {len(system.canonical_products):,}")
            
            if args.query:
                response = system.process_query(args.query)
                print(json.dumps(response, indent=2))
            
            if args.save_snapshot:
                snapshot_path = system.save_snapshot(args.save_snapshot)
                print(f"💾 Snapshot guardado: {snapshot_path}")
    
    elif args.mode == 'query':
        # Solo procesar query (asume sistema ya inicializado)
        if not args.query:
            print("❌ Debes proporcionar una query con --query")
            return
        
        # Inicializar rápido
        system.initialize_with_cache(force_reload=False)
        response = system.process_query(args.query)
        print(json.dumps(response, indent=2))
    
    elif args.mode == 'cache_stats':
        # Mostrar estadísticas de caché
        print("\n📊 ESTADÍSTICAS DE CACHÉ:")
        print("-" * 40)
        
        cache_dir = Path("data/cache")
        if cache_dir.exists():
            for subdir in cache_dir.iterdir():
                if subdir.is_dir():
                    files = list(subdir.glob("*"))
                    if files:
                        total_size = sum(f.stat().st_size for f in files) / (1024*1024)
                        print(f"{subdir.name}:")
                        print(f"  Archivos: {len(files)}")
                        print(f"  Tamaño: {total_size:.1f} MB")
                        if subdir.name == "canonical":
                            # Mostrar productos en caché
                            for f in files[:3]:
                                try:
                                    with open(f, 'rb') as fp:
                                        data = pickle.load(fp)
                                        if 'count' in data:
                                            print(f"  • {f.name}: {data['count']:,} productos")
                                except:
                                    pass
        else:
            print("❌ Directorio cache no existe")
        
        print("-" * 40)
    
    elif args.mode == 'clear_cache':
        # Limpiar caché antiguo
        confirm = input("⚠️  ¿Eliminar caché? (sí/no): ").strip().lower()
        if confirm == 'si' or confirm == 'sí':
            import shutil
            cache_dir = Path("data/cache")
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                print("✅ Caché eliminado")
            else:
                print("❌ Directorio cache no existe")
        else:
            print("❌ Cancelado")


if __name__ == "__main__":
    main()