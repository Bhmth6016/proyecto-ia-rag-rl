# src/core/embeddings/sentence_transformer_wrapper.py
"""
Wrapper para adaptar SentenceTransformer a la interfaz de LangChain/Chroma.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class SentenceTransformerWrapper:
    """
    Wrapper que convierte SentenceTransformer a la interfaz LangChain.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model: SentenceTransformer = SentenceTransformer(model_name, device=device)
        logger.info(f"✅ SentenceTransformer '{model_name}' inicializado en {device}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        try:
            # Siempre convertir a numpy array explícitamente
            embeddings: np.ndarray = np.array(self.model.encode(
                texts,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=True,  # Crucial: fuerza numpy array
                normalize_embeddings=True,
                batch_size=32
            ))
            
            # Asegurar que sea 2D
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # Convertir a lista de listas de floats
            return embeddings.tolist()
                
        except Exception as e:
            logger.error(f"❌ Error en embed_documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Implementa el método requerido por LangChain para queries."""
        try:
            # Siempre convertir a numpy array explícitamente
            embedding: np.ndarray = np.array(self.model.encode(
                text,
                convert_to_numpy=True,  # Crucial: fuerza numpy array
                normalize_embeddings=True
            ))
            
            # Aplanar a 1D si es necesario
            if embedding.ndim == 2:
                embedding = embedding.flatten()
            
            # Convertir a lista de floats
            return embedding.tolist()
                
        except Exception as e:
            logger.error(f"❌ Error en embed_query: {e}")
            raise
    
    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)