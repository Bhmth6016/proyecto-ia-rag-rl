# serialization_utils.py (nuevo archivo)
import json
import pickle
import base64
import numpy as np
from typing import List, Optional

class EmbeddingSerializer:
    """Utilidades para serializar embeddings de productos"""
    
    @staticmethod
    def serialize_embedding(embedding: List[float]) -> str:
        """Convierte embedding a string para JSON"""
        if not embedding:
            return ""
        
        # Opción 1: Base64 + pickle (eficiente)
        try:
            arr = np.array(embedding, dtype=np.float32)
            pickled = pickle.dumps(arr)
            return base64.b64encode(pickled).decode('utf-8')
        except:
            # Opción 2: JSON (menos eficiente pero legible)
            return json.dumps([float(x) for x in embedding])
    
    @staticmethod
    def deserialize_embedding(embedding_str: str) -> Optional[List[float]]:
        """Convierte string a embedding"""
        if not embedding_str:
            return None
        
        try:
            # Intentar base64 primero
            pickled = base64.b64decode(embedding_str.encode('utf-8'))
            arr = pickle.loads(pickled)
            return arr.tolist()
        except:
            # Fallback a JSON
            try:
                return json.loads(embedding_str)
            except:
                return None
    
    @staticmethod
    def optimize_for_storage(product_dict: dict) -> dict:
        """Optimiza diccionario de producto para almacenamiento"""
        optimized = product_dict.copy()
        
        if 'embedding' in optimized:
            embedding = optimized['embedding']
            if isinstance(embedding, list) and len(embedding) > 10:
                # Comprimir embedding grande
                optimized['embedding'] = EmbeddingSerializer.serialize_embedding(embedding)
        
        return optimized