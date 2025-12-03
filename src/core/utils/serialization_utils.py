# src/core/utils/serialization_utils.py
import json
import pickle
import base64
import numpy as np
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EmbeddingSerializer:
    """Utilidades para serializar embeddings de productos"""
    
    @staticmethod
    def serialize_embedding(embedding: List[float]) -> str:
        """Convierte embedding a string para JSON (base64 + pickle)"""
        if not embedding:
            return ""
        
        try:
            # Opción 1: Base64 + pickle (más eficiente)
            arr = np.array(embedding, dtype=np.float32)
            pickled = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)
            encoded = base64.b64encode(pickled).decode('utf-8')
            return f"b64pickle:{encoded}"
        except Exception as e:
            logger.warning(f"Base64 serialization failed, using JSON: {e}")
            # Opción 2: JSON (menos eficiente pero compatible)
            return f"json:{json.dumps([float(x) for x in embedding])}"
    
    @staticmethod
    def deserialize_embedding(embedding_str: str) -> Optional[List[float]]:
        """Convierte string a embedding"""
        if not embedding_str:
            return None
        
        try:
            if embedding_str.startswith("b64pickle:"):
                # Decodificar base64 + pickle
                encoded = embedding_str[len("b64pickle:"):]
                pickled = base64.b64decode(encoded.encode('utf-8'))
                arr = pickle.loads(pickled)
                return arr.tolist()
            elif embedding_str.startswith("json:"):
                # Decodificar JSON
                json_str = embedding_str[len("json:"):]
                return json.loads(json_str)
            else:
                # Intentar detectar formato
                if embedding_str.startswith('['):
                    return json.loads(embedding_str)
                else:
                    # Asumir base64 sin prefijo (backward compatibility)
                    pickled = base64.b64decode(embedding_str.encode('utf-8'))
                    arr = pickle.loads(pickled)
                    return arr.tolist()
        except Exception as e:
            logger.error(f"Error deserializing embedding: {e}")
            return None
    
    @staticmethod
    def optimize_for_storage(product_dict: dict) -> dict:
        """Optimiza diccionario de producto para almacenamiento"""
        optimized = product_dict.copy()
        
        if 'embedding' in optimized and optimized['embedding']:
            embedding = optimized['embedding']
            if isinstance(embedding, list) and len(embedding) > 10:
                # Comprimir embedding grande
                optimized['embedding'] = EmbeddingSerializer.serialize_embedding(embedding)
                optimized['embedding_compressed'] = True
        
        # También optimizar otros campos grandes
        if 'extracted_entities' in optimized and optimized['extracted_entities']:
            if isinstance(optimized['extracted_entities'], dict):
                optimized['extracted_entities'] = json.dumps(optimized['extracted_entities'])
        
        return optimized
    
    @staticmethod
    def restore_from_storage(product_dict: dict) -> dict:
        """Restaura diccionario de producto desde almacenamiento"""
        restored = product_dict.copy()
        
        if 'embedding' in restored and isinstance(restored['embedding'], str):
            embedding = EmbeddingSerializer.deserialize_embedding(restored['embedding'])
            if embedding:
                restored['embedding'] = embedding
                restored.pop('embedding_compressed', None)
        
        if 'extracted_entities' in restored and isinstance(restored['extracted_entities'], str):
            try:
                restored['extracted_entities'] = json.loads(restored['extracted_entities'])
            except:
                pass
        
        return restored

# Alias para compatibilidad
serialize_embedding = EmbeddingSerializer.serialize_embedding
deserialize_embedding = EmbeddingSerializer.deserialize_embedding