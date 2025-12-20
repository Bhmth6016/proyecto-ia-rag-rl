# run_simple_cache.py
"""
Sistema simple que usa caché - Sin problemas de importación
"""
import sys
from pathlib import Path
import pickle
import json
from datetime import datetime
import logging

# Configurar paths
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleCacheSystem:
    """Sistema simple con caché manual"""
    
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.canonical_products = []
        self.system = None
    
    def load_with_cache(self):
        """Carga sistema con caché simple"""
        print("\n" + "="*80)
        print("🚀 SISTEMA CON CACHÉ SIMPLE")
        print("="*80)
        
        # Archivo de caché
        cache_file = self.cache_dir / "system_cache.pkl"
        
        # Intentar cargar desde caché
        if self.use_cache and cache_file.exists():
            print("📥 Intentando cargar desde caché...")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                self.canonical_products = cached_data.get('products', [])
                
                if self.canonical_products:
                    print(f"✅ {len(self.canonical_products):,} productos cargados desde caché")
                    print(f"🕐 Caché creado: {cached_data.get('timestamp', 'desconocido')}")
                    
                    # Inicializar sistema rápidamente
                    self._initialize_system_fast()
                    return True
                    
            except Exception as e:
                print(f"⚠️  Error cargando caché: {e}")
        
        # Si no hay caché, cargar normalmente
        print("📥 Cargando sistema normalmente...")
        return self._load_without_cache(cache_file)
    
    def _load_without_cache(self, cache_file):
        """Carga sistema sin caché y luego guarda"""
        try:
            # Importar tu sistema original
            from src.main import RAGRLSystem
            from src.data.loader import load_raw_products
            
            print("📂 Cargando productos raw...")
            raw_products = load_raw_products(limit=None)
            
            print("🔧 Inicializando sistema...")
            self.system = RAGRLSystem('config/config.yaml')
            self.system.initialize_system(raw_products)
            
            # Guardar productos en caché
            self.canonical_products = self.system.canonical_products
            
            # Guardar en caché
            cache_data = {
                'products': self.canonical_products,
                'timestamp': datetime.now().isoformat(),
                'count': len(self.canonical_products)
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"💾 {len(self.canonical_products):,} productos guardados en caché")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando sistema: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _initialize_system_fast(self):
        """Inicializa sistema rápidamente desde caché"""
        try:
            from src.main import RAGRLSystem
            
            print("⚡ Inicializando sistema rápidamente...")
            
            # Crear sistema
            self.system = RAGRLSystem('config/config.yaml')
            
            # Inyectar productos canonicalizados
            self.system.canonical_products = self.canonical_products
            
            # Inicializar componentes mínimos
            self._init_minimal_components()
            
            print(f"✅ Sistema listo con {len(self.canonical_products):,} productos")
            
        except Exception as e:
            print(f"❌ Error inicializando sistema: {e}")
            import traceback
            traceback.print_exc()
    
    def _init_minimal_components(self):
        """Inicializa componentes mínimos del sistema"""
        try:
            # Importar componentes necesarios
            from data.canonicalizer import ProductCanonicalizer
            from src.query.understanding import QueryUnderstanding
            from features.extractor import FeatureEngineer
            from ranking.rl_ranker import RLHFRanker
            
            # Inicializar canonicalizer (para embeddings de queries)
            self.system.canonicalizer = ProductCanonicalizer(
                embedding_model="all-MiniLM-L6-v2"
            )
            
            # Inicializar otros componentes
            self.system.query_understanding = QueryUnderstanding()
            self.system.feature_engineer = FeatureEngineer()
            
            self.system.rl_ranker = RLHFRanker(
                alpha=0.1
            )
            
            # Construir índice FAISS desde caché si existe
            self._load_faiss_from_cache()
            
        except Exception as e:
            print(f"⚠️  Error inicializando componentes: {e}")
    
    def _load_faiss_from_cache(self):
        """Intenta cargar FAISS desde caché"""
        faiss_cache = self.cache_dir / "faiss_cache.index"
        if faiss_cache.exists():
            try:
                import faiss
                
                print("🔍 Cargando índice FAISS desde caché...")
                self.system.vector_store.index = faiss.read_index(str(faiss_cache))
                self.system.vector_store.is_locked = True
                
                # Cargar IDs de productos
                ids_cache = self.cache_dir / "faiss_product_ids.pkl"
                if ids_cache.exists():
                    with open(ids_cache, 'rb') as f:
                        product_ids = pickle.load(f)
                    
                    # Mapear IDs a productos
                    id_to_product = {getattr(p, 'id', str(i)): p 
                                   for i, p in enumerate(self.canonical_products)}
                    
                    self.system.vector_store.products = [
                        id_to_product.get(pid) for pid in product_ids if pid in id_to_product
                    ]
                
                print(f"✅ Índice FAISS cargado: {len(self.system.vector_store.products):,} vectores")
                
            except Exception as e:
                print(f"⚠️  Error cargando FAISS: {e}")
    
    def save_faiss_to_cache(self):
        """Guarda FAISS en caché"""
        if not self.system or not self.system.vector_store or not self.system.vector_store.index:
            return
        
        try:
            import faiss
            import numpy as np
            
            print("💾 Guardando índice FAISS en caché...")
            
            # Guardar índice
            faiss.write_index(self.system.vector_store.index, str(self.cache_dir / "faiss_cache.index"))
            
            # Guardar IDs de productos
            product_ids = []
            for product in self.system.canonical_products:
                product_ids.append(getattr(product, 'id', 'unknown'))
            
            with open(self.cache_dir / "faiss_product_ids.pkl", 'wb') as f:
                pickle.dump(product_ids, f)
            
            print("✅ FAISS guardado en caché")
            
        except Exception as e:
            print(f"⚠️  Error guardando FAISS: {e}")
    
    def process_query(self, query_text):
        """Procesa una query"""
        if not self.system:
            print("❌ Sistema no inicializado")
            return None
        
        try:
            print(f"\n🔍 Procesando query: '{query_text}'")
            
            # Usar el método de procesamiento del sistema
            response = self.system.process_query(query_text)
            
            return response
            
        except Exception as e:
            print(f"❌ Error procesando query: {e}")
            return None
    
    def show_stats(self):
        """Muestra estadísticas"""
        print("\n📊 ESTADÍSTICAS:")
        print("-" * 40)
        print(f"   Productos: {len(self.canonical_products):,}")
        print(f"   Caché: {'HABILITADO' if self.use_cache else 'DESHABILITADO'}")
        
        if self.system and self.system.vector_store:
            print(f"   FAISS: {'CARGADO' if self.system.vector_store.index else 'NO CARGADO'}")
        
        print("-" * 40)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema con caché simple")
    parser.add_argument('--query', type=str, help='Query a procesar')
    parser.add_argument('--no-cache', action='store_true', help='No usar caché')
    parser.add_argument('--stats', action='store_true', help='Mostrar estadísticas de caché')
    
    args = parser.parse_args()
    
    if args.stats:
        # Mostrar estadísticas de caché
        cache_dir = Path("data/cache")
        if cache_dir.exists():
            print("\n📊 ESTADÍSTICAS DE CACHÉ:")
            print("-" * 40)
            
            cache_files = list(cache_dir.glob("*"))
            if cache_files:
                total_size = sum(f.stat().st_size for f in cache_files) / (1024*1024)
                print(f"Archivos: {len(cache_files)}")
                print(f"Tamaño total: {total_size:.1f} MB")
                
                for f in cache_files:
                    size_mb = f.stat().st_size / (1024*1024)
                    print(f"  • {f.name}: {size_mb:.1f} MB")
            else:
                print("❌ No hay archivos en caché")
        else:
            print("❌ Directorio cache no existe")
        
        print("-" * 40)
        return
    
    # Crear sistema
    system = SimpleCacheSystem(use_cache=not args.no_cache)
    
    # Cargar sistema
    success = system.load_with_cache()
    
    if not success:
        print("❌ No se pudo cargar el sistema")
        return
    
    # Mostrar estadísticas
    system.show_stats()
    
    # Procesar query si se proporciona
    if args.query:
        response = system.process_query(args.query)
        if response:
            print(json.dumps(response, indent=2))
    
    # Guardar FAISS en caché si se cargó normalmente
    if args.no_cache or not Path("data/cache/faiss_cache.index").exists():
        system.save_faiss_to_cache()


if __name__ == "__main__":
    main()