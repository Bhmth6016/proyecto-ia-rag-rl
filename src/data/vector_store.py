# src/data/vector_store.py
"""
Almacenamiento vectorial simple con FAISS
"""
import numpy as np
import faiss
import pickle
from typing import List, Optional, Tuple
import logging
from .canonicalizer import CanonicalProduct

logger = logging.getLogger(__name__)


class VectorStore:
    """Almacenamiento vectorial simple - SOLO retrieval"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.products: List[CanonicalProduct] = []
        self._built = False
    
    def build_index(self, products: List[CanonicalProduct]):
        """Construye índice FAISS una sola vez"""
        if not products:
            raise ValueError("No hay productos para indexar")
        
        logger.info(f"🔨 Construyendo índice con {len(products)} productos")
        
        # Extraer embeddings
        embeddings = np.array([p.content_embedding for p in products], dtype=np.float32)
        
        # Crear índice FAISS plano (más simple y reproducible)
        self.index = faiss.IndexFlatIP(self.dimension)  # Producto interno
        self.index.add(embeddings)
        
        # Guardar productos
        self.products = products
        self._built = True
        
        logger.info(f"✅ Índice construido: {self.index.ntotal} vectores")
    
    def search(self, query_embedding: np.ndarray, k: int = 50) -> List[CanonicalProduct]:
        """Búsqueda por similitud coseno - NO modifica nada"""
        if not self._built:
            raise RuntimeError("Índice no construido")
        
        # Normalizar query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Búsqueda
        distances, indices = self.index.search(query_embedding, min(k, len(self.products)))
        
        # Devolver productos
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.products):
                results.append(self.products[idx])
        
        return results
    
    def save(self, path: str):
        """Guarda índice y productos"""
        if not self._built:
            raise RuntimeError("No hay índice para guardar")
        
        # Guardar FAISS index
        faiss.write_index(self.index, f"{path}.faiss")
        
        # Guardar productos (sin embeddings)
        product_data = []
        for p in self.products:
            product_data.append({
                "id": p.id,
                "title": p.title,
                "category": p.category,
                "price": p.price,
                "rating": p.rating
            })
        
        with open(f"{path}.products.pkl", 'wb') as f:
            pickle.dump(product_data, f)
        
        logger.info(f"💾 VectorStore guardado en {path}")
    
    def load(self, path: str):
        """Carga índice guardado"""
        # Cargar índice FAISS
        self.index = faiss.read_index(f"{path}.faiss")
        
        # Cargar productos
        with open(f"{path}.products.pkl", 'rb') as f:
            product_data = pickle.load(f)
        
        # Reconstruir productos básicos
        self.products = []
        for data in product_data:
            # Crear producto básico (sin embeddings)
            self.products.append(CanonicalProduct(
                id=data["id"],
                title=data["title"],
                description="",  # No se carga para ahorrar memoria
                price=data["price"],
                category=data["category"],
                rating=data["rating"],
                rating_count=None,
                title_embedding=np.zeros(self.dimension),
                content_embedding=np.zeros(self.dimension)
            ))
        
        self._built = True
        logger.info(f"📂 VectorStore cargado: {len(self.products)} productos")