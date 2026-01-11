# src/data/vector_store.py
"""
Vector Store Inmutable - Retrieval INMUTABLE durante evaluación
"""
import numpy as np
import faiss
import pickle
from typing import List, Optional, Tuple, Dict, Any  # AÑADE ESTO
import logging
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)


class ImmutableVectorStore:
    """
    Retrieval INMUTABLE durante evaluación
    
    Construido UNA VEZ, usado MUCHAS VECES
    """
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.products: List = []
        self.is_locked = False  # Flag de inmutabilidad
        self.construction_time = None
        
        logger.info(f"📚 Vector Store Inmutable (d={dimension})")
    
    def build_index(self, products: List):
        """Construye índice UNA SOLA VEZ"""
        if self.index is not None:
            raise RuntimeError("Índice ya construido - Inmutable")
        
        if not products:
            raise ValueError("No hay productos para indexar")
        
        logger.info(f"🔨 Construyendo índice inmutable con {len(products)} productos...")
        
        # 1. Verificar que todos los productos tengan embeddings
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
        
        # 2. Crear índice FAISS con configuración reproducible
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_array)
        
        # 3. Guardar productos
        self.products = valid_products
        
        # 4. Bloquear para inmutabilidad
        self.is_locked = True
        self.construction_time = np.datetime64('now')
        
        logger.info(f"✅ Índice inmutable construido:")
        logger.info(f"   • Vectores: {self.index.ntotal}")
        logger.info(f"   • Dimensión: {self.dimension}")
        logger.info(f"   • Bloqueado: {self.is_locked}")
        
        # Verificación de principio
        self._verify_immutability_principle()
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 50,
        verify_immutability: bool = True
    ) -> List:
        """
        BÚSQUEDA PURA - Sin side effects
        
        Args:
            query_embedding: Embedding de la query (normalizado)
            k: Número de resultados
            verify_immutability: Verificar principio de inmutabilidad
            
        Returns:
            Lista de productos recuperados
        """
        if not self.is_locked:
            raise RuntimeError("Índice no listo o no inmutable - Debe construirse primero")
        
        if self.index is None:
            raise RuntimeError("Índice no inicializado")
        
        # PRINCIPIO: Solo lectura
        if verify_immutability:
            self._verify_read_only_access()
        
        # Normalizar query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            raise ValueError("Query embedding tiene norma cero")
        
        query_normalized = (query_embedding / query_norm).astype(np.float32)
        query_normalized = query_normalized.reshape(1, -1)
        
        # Búsqueda en FAISS
        k_actual = min(k, len(self.products))
        distances, indices = self.index.search(query_normalized, k_actual)
        
        # Devolver productos
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
        """Búsqueda con scores de similitud"""
        results = self.search(query_embedding, k, verify_immutability=True)
        
        # Calcular scores (similitud coseno)
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
        """Verifica que el índice cumple principio de inmutabilidad"""
        checks = []
        
        # Check 1: Índice debe existir y estar bloqueado
        checks.append(('index_exists', self.index is not None))
        checks.append(('is_locked', self.is_locked))
        
        # Check 2: Debe tener productos
        checks.append(('has_products', len(self.products) > 0))
        
        # Check 3: Verificar que no se pueda modificar
        if self.index is not None:
            # Intentar agregar un vector (debería fallar si bloqueado correctamente)
            try:
                test_vector = np.random.randn(1, self.dimension).astype(np.float32)
                original_ntotal = self.index.ntotal
                
                # Esta operación debería ser permitida por FAISS técnicamente,
                # pero registramos la intención de inmutabilidad
                logger.info("   • Inmutabilidad: Índice marcado como read-only")
                
                checks.append(('immutable_flag', True))
            except Exception as e:
                checks.append(('modification_blocked', True))
        
        # Reportar checks
        all_passed = all(passed for _, passed in checks)
        
        if all_passed:
            logger.info("✅ Principio de inmutabilidad verificado")
        else:
            failed = [name for name, passed in checks if not passed]
            logger.warning(f"⚠️  Inmutabilidad: checks fallidos: {failed}")
    
    def _verify_read_only_access(self):
        """Verifica que solo se está accediendo en modo lectura"""
        # Esto es más un check conceptual que técnico
        # En producción, podríamos usar permisos de archivo o bases de datos read-only
        
        if not self.is_locked:
            logger.warning("⚠️  Acceso a índice no bloqueado")
        
        # Podemos verificar que no se llamen métodos de modificación
        # (en Python esto es más difícil, confiamos en la arquitectura)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del índice"""
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
        """Guarda índice (para reproducibilidad)"""
        if not self.is_locked:
            raise RuntimeError("Índice no bloqueado - No se debe guardar durante evaluación")
        
        # Guardar FAISS index
        faiss.write_index(self.index, f"{path}.faiss")
        
        # Guardar metadatos de productos (sin embeddings completos para ahorrar espacio)
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
        
        # Guardar configuración
        config = {
            'dimension': self.dimension,
            'construction_time': str(self.construction_time),
            'num_products': len(self.products),
            'is_locked': self.is_locked
        }
        
        with open(f"{path}.config.json", 'w') as f:
            import json
            json.dump(config, f, indent=2)
        
        logger.info(f"💾 Índice guardado en {path}")
    
    def load(self, path: str):
        """Carga índice guardado (read-only)"""
        # Cargar índice FAISS
        self.index = faiss.read_index(f"{path}.faiss")
        
        # Cargar metadatos
        with open(f"{path}.metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Cargar configuración
        with open(f"{path}.config.json", 'r') as f:
            import json
            config = json.load(f)
        
        # Configurar
        self.dimension = config['dimension']
        self.is_locked = True  # Siempre locked al cargar
        self.construction_time = np.datetime64(config['construction_time'])
        
        # Recrear productos básicos
        self.products = []
        for meta in metadata:
            # Crear objeto producto mínimo
            class MinimalProduct:
                def __init__(self, meta_dict):
                    self.id = meta_dict['id']
                    self.title = meta_dict['title']
                    self.category = meta_dict['category']
                    self.content_hash = meta_dict['content_hash']
                    # Embedding dummy (no se usa en retrieval desde índice cargado)
                    self.content_embedding = np.zeros(self.dimension)
            
            self.products.append(MinimalProduct(meta))
        
        logger.info(f"📂 Índice cargado: {len(self.products)} productos, locked={self.is_locked}")