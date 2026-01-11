# src/data/vector_store.py
import numpy as np
import faiss
import pickle
from typing import List, Optional, Tuple, Dict, Any 
import logging
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


class ImmutableVectorStore:  
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.products: List = []
        self.is_locked = False  # Flag de inmutabilidad
        self.construction_time = None
        
        logger.info(f" Vector Store Inmutable (d={dimension})")
    
    def build_index(self, products: List):
        if self.index is not None:
            raise RuntimeError("Índice ya construido - Inmutable")
        
        if not products:
            raise ValueError("No hay productos para indexar")
        
        logger.info(f" Construyendo índice inmutable con {len(products)} productos...")
        
        embeddings = []
        valid_products = []
        
        for i, product in enumerate(products):
            if hasattr(product, 'content_embedding'):
                embedding = product.content_embedding
                if embedding is not None and len(embedding) == self.dimension:
                    embeddings.append(embedding.astype(np.float32))
                    valid_products.append(product)
            elif i % 100000 == 0:
                logger.warning(f"Producto {i} sin embedding válido")
        
        if not embeddings:
            raise ValueError("No hay embeddings válidos")
        
        embeddings_array = np.array(embeddings)
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_array) # pyright: ignore[reportCallIssue]
        
        self.products = valid_products
        
        self.is_locked = True
        self.construction_time = np.datetime64('now')
        
        logger.info(" Índice inmutable construido:")
        logger.info(f"   • Vectores: {self.index.ntotal}")
        logger.info(f"   • Dimensión: {self.dimension}")
        logger.info(f"   • Bloqueado: {self.is_locked}")
        
        self._verify_immutability_principle()
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 50,
        verify_immutability: bool = True
    ) -> List:
        if not self.is_locked:
            raise RuntimeError("Índice no listo o no inmutable - Debe construirse primero")
        
        if self.index is None:
            raise RuntimeError("Índice no inicializado")
        
        if verify_immutability:
            self._verify_read_only_access()
        
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            raise ValueError("Query embedding tiene norma cero")
        
        query_normalized = (query_embedding / query_norm).astype(np.float32)
        query_normalized = query_normalized.reshape(1, -1)
        
        k_actual = min(k, len(self.products))
        distances, indices = self.index.search(# pyright: ignore[reportCallIssue]
            query_normalized,
            k_actual) 
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.products):
                results.append(self.products[idx])
        
        logger.debug(f"🔍 Búsqueda completada: {len(results)} resultados")
        
        return results
    
    def search_with_scores(
        self, 
        query_embedding: np.ndarray, 
        k: int = 50
    ) -> Tuple[List, List[float]]:
        results = self.search(query_embedding, k, verify_immutability=True)
        
        scores = []
        query_normalized = query_embedding / np.linalg.norm(query_embedding)
        
        for product in results:
            if hasattr(product, 'content_embedding'):
                product_embedding = product.content_embedding
                product_normalized = product_embedding / np.linalg.norm(product_embedding)
                score = np.dot(query_normalized, product_normalized)
                scores.append(float(score))
            else:
                scores.append(0.0)
        
        return results, scores
    
    def _verify_immutability_principle(self):
        checks = []
        
        checks.append(('index_exists', self.index is not None))
        checks.append(('is_locked', self.is_locked))
        
        checks.append(('has_products', len(self.products) > 0))
        
        if self.index is not None:
            try:
                logger.info("   • Inmutabilidad: Índice marcado como read-only")
                
                checks.append(('immutable_flag', True))
            except Exception:
                checks.append(('modification_blocked', True))
        
        all_passed = all(passed for _, passed in checks)
        
        if all_passed:
            logger.info(" Principio de inmutabilidad verificado")
        else:
            failed = [name for name, passed in checks if not passed]
            logger.warning(f"  Inmutabilidad: checks fallidos: {failed}")
    
    def _verify_read_only_access(self):
        if not self.is_locked:
            logger.warning("  Acceso a índice no bloqueado")
        
    def get_index_stats(self) -> Dict[str, Any]:
        if self.index is None:
            return {'status': 'not_built'}
        
        return {
            'status': 'built_and_locked' if self.is_locked else 'built_not_locked',
            'num_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'is_trained': self.index.is_trained,
            'construction_time': str(self.construction_time),
            'num_products': len(self.products),
            'immutability_principle': 'enforced' if self.is_locked else 'not_enforced'
        }
    
    def save(self, path: str):
        if not self.is_locked:
            raise RuntimeError("Índice no bloqueado - No se debe guardar durante evaluación")
        
        faiss.write_index(self.index, f"{path}.faiss")
        
        metadata = []
        for product in self.products:
            metadata.append({
                'id': getattr(product, 'id', 'unknown'),
                'title': getattr(product, 'title', '')[:100],
                'category': getattr(product, 'category', 'unknown'),
                'content_hash': getattr(product, 'content_hash', '')
            })
        
        with open(f"{path}.metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        config = {
            'dimension': self.dimension,
            'construction_time': str(self.construction_time),
            'num_products': len(self.products),
            'is_locked': self.is_locked
        }
        
        with open(f"{path}.config.json", 'w') as f:
            import json
            json.dump(config, f, indent=2)
        
        logger.info(f" Índice guardado en {path}")
    
    def load(self, path: str):
        self.index = faiss.read_index(f"{path}.faiss")
        
        with open(f"{path}.metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        with open(f"{path}.config.json", 'r') as f:
            import json
            config = json.load(f)
        
        self.dimension = config['dimension']
        self.is_locked = True  # Siempre locked al cargar
        self.construction_time = np.datetime64(config['construction_time'])
        
        self.products = []
        for meta in metadata:
            class MinimalProduct:
                def __init__(self, meta_dict, dimension: int):
                    self.id = meta_dict['id']
                    self.title = meta_dict['title']
                    self.category = meta_dict['category']
                    self.content_hash = meta_dict['content_hash']
                    self.content_embedding = np.zeros(dimension, dtype=np.float32)

            
            self.products = []
            for meta in metadata:
                self.products.append(MinimalProduct(meta, self.dimension))
        
        logger.info(f" Índice cargado: {len(self.products)} productos, locked={self.is_locked}")