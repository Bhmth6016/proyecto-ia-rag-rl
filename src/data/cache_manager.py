# src/data/cache_manager.py
"""
Sistema de caché y persistencia para evitar recargar 90K productos cada vez
"""
import pickle
import json
from pathlib import Path
from datetime import datetime
import hashlib
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """Gestiona caché de productos canonicalizados y embeddings"""
    
    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Directorios específicos
        self.canonical_cache = self.cache_dir / "canonical"
        self.embedding_cache = self.cache_dir / "embeddings" 
        self.index_cache = self.cache_dir / "indices"
        
        for d in [self.canonical_cache, self.embedding_cache, self.index_cache]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Metadata
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Carga metadatos del caché"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """Guarda metadatos del caché"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_cache_key(self, data_source, version="v1"):
        """Genera una clave única para el caché"""
        key_data = f"{data_source}_{version}_{datetime.now().strftime('%Y%m')}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def save_canonical_products(self, products, source_name):
        """Guarda productos canonicalizados en caché"""
        cache_key = self.get_cache_key(source_name)
        cache_file = self.canonical_cache / f"{cache_key}.pkl"
        
        try:
            # Guardar productos
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'products': products,
                    'timestamp': datetime.now().isoformat(),
                    'count': len(products),
                    'source': source_name
                }, f)
            
            # Actualizar metadatos
            self.metadata['last_canonical'] = {
                'file': str(cache_file),
                'timestamp': datetime.now().isoformat(),
                'count': len(products),
                'source': source_name
            }
            self._save_metadata()
            
            logger.info(f"✅ Caché guardado: {len(products):,} productos en {cache_file}")
            return cache_key
            
        except Exception as e:
            logger.error(f"❌ Error guardando caché: {e}")
            return None
    
    def load_canonical_products(self, source_name=None):
        """Carga productos canonicalizados desde caché"""
        if source_name:
            cache_key = self.get_cache_key(source_name)
            cache_file = self.canonical_cache / f"{cache_key}.pkl"
        else:
            # Cargar el más reciente
            cache_files = list(self.canonical_cache.glob("*.pkl"))
            if not cache_files:
                return None
            cache_file = max(cache_files, key=lambda f: f.stat().st_mtime)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                logger.info(f"✅ Caché cargado: {data['count']:,} productos de {data['source']}")
                logger.info(f"   Timestamp: {data['timestamp']}")
                
                return data['products']
                
            except Exception as e:
                logger.error(f"❌ Error cargando caché: {e}")
                return None
        
        return None
    
    def save_embeddings(self, embeddings, product_ids, source_name):
        """Guarda embeddings en caché"""
        cache_key = self.get_cache_key(source_name + "_embeddings")
        cache_file = self.embedding_cache / f"{cache_key}.pkl"
        
        try:
            # Guardar embeddings (numpy array optimizado)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'embeddings': embeddings,
                    'product_ids': product_ids,
                    'dimension': embeddings.shape[1] if len(embeddings.shape) > 1 else embeddings.shape[0],
                    'count': len(product_ids),
                    'timestamp': datetime.now().isoformat()
                }, f, protocol=4)  # Protocolo 4 para compatibilidad
            
            logger.info(f"✅ Embeddings guardados: {len(product_ids):,} vectores")
            return cache_key
            
        except Exception as e:
            logger.error(f"❌ Error guardando embeddings: {e}")
            return None
    
    def load_embeddings(self, source_name=None):
        """Carga embeddings desde caché"""
        if source_name:
            cache_key = self.get_cache_key(source_name + "_embeddings")
            cache_file = self.embedding_cache / f"{cache_key}.pkl"
        else:
            # Cargar el más reciente
            cache_files = list(self.embedding_cache.glob("*.pkl"))
            if not cache_files:
                return None, None
            cache_file = max(cache_files, key=lambda f: f.stat().st_mtime)
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                logger.info(f"✅ Embeddings cargados: {data['count']:,} vectores (d={data['dimension']})")
                return data['embeddings'], data['product_ids']
                
            except Exception as e:
                logger.error(f"❌ Error cargando embeddings: {e}")
                return None, None
        
        return None, None
    
    def save_faiss_index(self, index, product_ids, source_name):
        """Guarda índice FAISS en caché"""
        cache_key = self.get_cache_key(source_name + "_faiss")
        index_file = self.index_cache / f"{cache_key}.index"
        ids_file = self.index_cache / f"{cache_key}_ids.pkl"
        
        try:
            # Guardar índice FAISS
            import faiss
            faiss.write_index(index, str(index_file))
            
            # Guardar IDs
            with open(ids_file, 'wb') as f:
                pickle.dump(product_ids, f)
            
            logger.info(f"✅ Índice FAISS guardado: {len(product_ids):,} vectores")
            return cache_key
            
        except Exception as e:
            logger.error(f"❌ Error guardando índice FAISS: {e}")
            return None
    
    def load_faiss_index(self, source_name=None):
        """Carga índice FAISS desde caché"""
        if source_name:
            cache_key = self.get_cache_key(source_name + "_faiss")
            index_file = self.index_cache / f"{cache_key}.index"
            ids_file = self.index_cache / f"{cache_key}_ids.pkl"
        else:
            # Cargar el más reciente
            index_files = list(self.index_cache.glob("*.index"))
            if not index_files:
                return None, None
            index_file = max(index_files, key=lambda f: f.stat().st_mtime)
            
            # Encontrar archivo de IDs correspondiente
            base_name = index_file.stem.replace("_faiss", "")
            ids_file = self.index_cache / f"{base_name}_ids.pkl"
        
        if index_file.exists() and ids_file.exists():
            try:
                import faiss
                index = faiss.read_index(str(index_file))
                
                with open(ids_file, 'rb') as f:
                    product_ids = pickle.load(f)
                
                logger.info(f"✅ Índice FAISS cargado: {len(product_ids):,} vectores")
                return index, product_ids
                
            except Exception as e:
                logger.error(f"❌ Error cargando índice FAISS: {e}")
                return None, None
        
        return None, None
    
    def clear_cache(self, days_old=30):
        """Limpia caché antiguo"""
        import time
        current_time = time.time()
        deleted = 0
        
        for cache_dir in [self.canonical_cache, self.embedding_cache, self.index_cache]:
            for file_path in cache_dir.glob("*"):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > days_old * 86400:  # días a segundos
                        file_path.unlink()
                        deleted += 1
        
        if deleted > 0:
            logger.info(f"🧹 Caché limpiado: {deleted} archivos antiguos eliminados")
        
        return deleted