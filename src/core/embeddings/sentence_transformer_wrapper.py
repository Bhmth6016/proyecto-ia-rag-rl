# src/core/embeddings/sentence_transformer_wrapper.py
"""
Wrapper para adaptar SentenceTransformer a la interfaz de LangChain/Chroma.
"""

from typing import List, Optional, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class SentenceTransformerWrapper:
    """
    Wrapper que convierte SentenceTransformer a la interfaz LangChain.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa el modelo una sola vez."""
        logger.info(f"🔄 Inicializando SentenceTransformer: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"✅ SentenceTransformer inicializado en {self.device}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Implementa el método requerido por LangChain."""
        if not texts:
            return []
        
        try:
            # Usar encode() de SentenceTransformer
            embeddings = self.model.encode(
                texts,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"❌ Error en embed_documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Implementa el método requerido por LangChain para queries."""
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"❌ Error en embed_query: {e}")
            raise
    
    # Métodos adicionales para compatibilidad
    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)