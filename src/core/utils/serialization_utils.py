# src/core/utils/serialization_utils.py - VERSIÓN MEJORADA
"""
Utilidades para serialización de embeddings con validación robusta.
Problema 3: Serialización de embeddings inconsistentes - SOLUCIÓN COMPLETA
"""
import json
import pickle
import base64
import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple
import logging
import struct
import zlib
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# 🔥 SOLUCIÓN PROBLEMA 3: Constantes para validación
# ============================================================================

class EmbeddingFormat(Enum):
    """Formatos de embedding soportados."""
    LIST = "list"
    NUMPY = "numpy"
    BASE64_PICKLE = "b64pickle"
    BASE64_JSON = "b64json"
    JSON = "json"
    COMPRESSED = "compressed"
    UNKNOWN = "unknown"


class EmbeddingValidation:
    """Configuración para validación de embeddings."""
    
    # Límites de dimensiones razonables
    MIN_DIMENSIONS = 10
    MAX_DIMENSIONS = 10000
    
    # Límites de valores
    MIN_VALUE = -100.0
    MAX_VALUE = 100.0
    
    # Límites de strings serializados
    MIN_SERIALIZED_LENGTH = 20
    MAX_SERIALIZED_LENGTH = 1000000
    
    # Tolerancia para validación
    MAX_NAN_RATIO = 0.01  # Máximo 1% de valores NaN
    MAX_ZERO_RATIO = 0.95  # Máximo 95% de ceros


# ============================================================================
# 🔥 CLASE PRINCIPAL MEJORADA
# ============================================================================

class EmbeddingSerializer:
    """
    Utilidades para serializar embeddings de productos con validación robusta.
    
    Problema 3: Solución completa para serialización de embeddings inconsistentes
    """
    
    @staticmethod
    def detect_embedding_format(embedding: Any) -> Tuple[EmbeddingFormat, Any]:
        """
        Detecta el formato de un embedding.
        
        Returns:
            Tuple[formato, embedding procesado]
        """
        if embedding is None:
            return EmbeddingFormat.UNKNOWN, None
        
        # String serializado
        if isinstance(embedding, str):
            if embedding.startswith("b64pickle:"):
                return EmbeddingFormat.BASE64_PICKLE, embedding
            elif embedding.startswith("b64json:"):
                return EmbeddingFormat.BASE64_JSON, embedding
            elif embedding.startswith("compressed:"):
                return EmbeddingFormat.COMPRESSED, embedding
            elif embedding.startswith('[') and embedding.endswith(']'):
                return EmbeddingFormat.JSON, embedding
            elif len(embedding) > EmbeddingValidation.MIN_SERIALIZED_LENGTH:
                # Podría ser base64 sin prefijo
                return EmbeddingFormat.BASE64_PICKLE, f"b64pickle:{embedding}"
            else:
                return EmbeddingFormat.UNKNOWN, embedding
        
        # Lista
        elif isinstance(embedding, list):
            return EmbeddingFormat.LIST, embedding
        
        # Numpy array
        elif isinstance(embedding, np.ndarray):
            return EmbeddingFormat.NUMPY, embedding
        
        # Otros tipos
        else:
            return EmbeddingFormat.UNKNOWN, embedding
    
    # 🔥 SOLUCIÓN PROBLEMA 3: Método de validación mejorado
    @staticmethod
    def validate_embedding_structure(embedding: Any, expected_dim: Optional[int] = None) -> bool:
        if embedding is None:
            logger.debug("Embedding es None")
            return False
        
        # 🔥 DETECCIÓN DE FORMATO
        format_type, processed_embedding = EmbeddingSerializer.detect_embedding_format(embedding)
        
        # 🔥 VALIDACIÓN POR FORMATO
        if format_type == EmbeddingFormat.LIST:
            return EmbeddingSerializer._validate_list_embedding(processed_embedding, expected_dim)
        
        elif format_type == EmbeddingFormat.NUMPY:
            return EmbeddingSerializer._validate_numpy_embedding(processed_embedding, expected_dim)
        
        elif format_type in [EmbeddingFormat.BASE64_PICKLE, 
                           EmbeddingFormat.BASE64_JSON,
                           EmbeddingFormat.JSON,
                           EmbeddingFormat.COMPRESSED]:
            # Validar string serializado
            if not isinstance(processed_embedding, str):
                logger.debug("String serializado no es string")
                return False
            
            if len(processed_embedding) < EmbeddingValidation.MIN_SERIALIZED_LENGTH:
                logger.debug(f"String serializado muy corto: {len(processed_embedding)}")
                return False
            
            if len(processed_embedding) > EmbeddingValidation.MAX_SERIALIZED_LENGTH:
                logger.debug(f"String serializado muy largo: {len(processed_embedding)}")
                return False
            
            # Intentar deserializar para validar contenido
            try:
                deserialized = EmbeddingSerializer.deserialize_embedding(processed_embedding)
                if deserialized is None:
                    logger.debug("No se pudo deserializar")
                    return False
                
                # Validar el embedding deserializado
                return EmbeddingSerializer._validate_list_embedding(deserialized, expected_dim)
                
            except Exception as e:
                logger.debug(f"Error deserializando para validación: {e}")
                return False
        
        else:
            logger.debug(f"Formato desconocido: {type(embedding)}")
            return False
    
    @staticmethod
    def _validate_list_embedding(embedding: List[float], expected_dim: Optional[int] = None) -> bool:
        if not isinstance(embedding, list):
            return False
        
        # Validar longitud
        if len(embedding) < EmbeddingValidation.MIN_DIMENSIONS:
            logger.debug(f"Embedding muy corto: {len(embedding)} dimensiones")
            return False
        
        if len(embedding) > EmbeddingValidation.MAX_DIMENSIONS:
            logger.debug(f"Embedding muy largo: {len(embedding)} dimensiones")
            return False
        
        # Validar dimensión esperada
        if expected_dim is not None and len(embedding) != expected_dim:
            logger.debug(f"Dimensión incorrecta: {len(embedding)} != {expected_dim}")
            return False
        
        # Convertir a numpy para validación numérica
        try:
            arr = np.array(embedding, dtype=np.float32)
            return EmbeddingSerializer._validate_numeric_embedding(arr)
        except Exception as e:
            logger.debug(f"Error convirtiendo a numpy: {e}")
            return False
    
    @staticmethod
    def _validate_numpy_embedding(embedding: np.ndarray, expected_dim: Optional[int] = None) -> bool:
        """Valida un embedding en formato numpy array."""
        if not isinstance(embedding, np.ndarray):
            return False
        
        # Validar dimensión
        if embedding.ndim != 1:
            logger.debug(f"Dimensión numpy incorrecta: {embedding.ndim}")
            return False
        
        if embedding.size < EmbeddingValidation.MIN_DIMENSIONS:
            logger.debug(f"Embedding numpy muy corto: {embedding.size}")
            return False
        
        if embedding.size > EmbeddingValidation.MAX_DIMENSIONS:
            logger.debug(f"Embedding numpy muy largo: {embedding.size}")
            return False
        
        # Validar dimensión esperada
        if expected_dim is not None and embedding.size != expected_dim:
            logger.debug(f"Dimensión numpy incorrecta: {embedding.size} != {expected_dim}")
            return False
        
        # Validación numérica
        return EmbeddingSerializer._validate_numeric_embedding(embedding)
    
    @staticmethod
    def _validate_numeric_embedding(arr: np.ndarray) -> bool:
        """Valida el contenido numérico de un embedding."""
        try:
            # Verificar que no sea todo ceros
            if np.all(arr == 0):
                logger.debug("Embedding es todo ceros")
                return False
            
            # Verificar valores NaN
            nan_count = np.sum(np.isnan(arr))
            nan_ratio = nan_count / len(arr)
            if nan_ratio > EmbeddingValidation.MAX_NAN_RATIO:
                logger.debug(f"Demasiados NaNs: {nan_count}/{len(arr)}")
                return False
            
            # Verificar valores infinitos
            inf_count = np.sum(np.isinf(arr))
            if inf_count > 0:
                logger.debug(f"Valores infinitos: {inf_count}")
                return False
            
            # Verificar rango de valores
            if np.any(arr < EmbeddingValidation.MIN_VALUE):
                logger.debug(f"Valores por debajo de {EmbeddingValidation.MIN_VALUE}")
                return False
            
            if np.any(arr > EmbeddingValidation.MAX_VALUE):
                logger.debug(f"Valores por encima de {EmbeddingValidation.MAX_VALUE}")
                return False
            
            # Verificar varianza (embeddings no deberían ser constantes)
            if np.std(arr) < 0.001:
                logger.debug("Embedding tiene varianza muy baja")
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error en validación numérica: {e}")
            return False
    
    # 🔥 SOLUCIÓN PROBLEMA 3: Alias para compatibilidad (tu método existente)
    @staticmethod
    def validate_embedding(embedding: Any, expected_dim: Optional[int] = None) -> bool:
        return EmbeddingSerializer.validate_embedding_structure(embedding, expected_dim)
    
    # ============================================================================
    # 🔥 MÉTODOS DE SERIALIZACIÓN MEJORADOS
    # ============================================================================
    
    @staticmethod
    def serialize_embedding(embedding: List[float], method: str = "auto") -> str:
        """
        Convierte embedding a string para JSON.
        
        Args:
            embedding: Lista de floats
            method: "auto", "b64pickle", "b64json", "json", "compressed"
            
        Returns:
            String serializado con prefijo de formato
        """
        if not embedding:
            return ""
        
        # Validar antes de serializar
        if not EmbeddingSerializer.validate_embedding_structure(embedding):
            logger.warning("Embedding no válido para serialización")
            return ""
        
        try:
            # Determinar mejor método si es auto
            if method == "auto":
                method = EmbeddingSerializer._select_best_method(len(embedding))
            
            # Serializar según método
            if method == "b64pickle":
                return EmbeddingSerializer._serialize_b64pickle(embedding)
            elif method == "b64json":
                return EmbeddingSerializer._serialize_b64json(embedding)
            elif method == "json":
                return EmbeddingSerializer._serialize_json(embedding)
            elif method == "compressed":
                return EmbeddingSerializer._serialize_compressed(embedding)
            else:
                logger.warning(f"Método de serialización desconocido: {method}, usando b64pickle")
                return EmbeddingSerializer._serialize_b64pickle(embedding)
                
        except Exception as e:
            logger.error(f"Error serializando embedding: {e}")
            # Fallback a JSON simple
            try:
                return f"json:{json.dumps([float(x) for x in embedding])}"
            except:
                return ""
    
    @staticmethod
    def _select_best_method(dimension: int) -> str:
        """Selecciona el mejor método de serialización basado en la dimensión."""
        if dimension > 1000:
            return "compressed"
        elif dimension > 100:
            return "b64pickle"
        else:
            return "b64json"
    
    @staticmethod
    def _serialize_b64pickle(embedding: List[float]) -> str:
        """Serializa usando Base64 + Pickle (más eficiente)."""
        arr = np.array(embedding, dtype=np.float32)
        pickled = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)
        encoded = base64.b64encode(pickled).decode('utf-8')
        return f"b64pickle:{encoded}"
    
    @staticmethod
    def _serialize_b64json(embedding: List[float]) -> str:
        """Serializa usando Base64 + JSON (balanceado)."""
        json_str = json.dumps([float(x) for x in embedding])
        encoded = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
        return f"b64json:{encoded}"
    
    @staticmethod
    def _serialize_json(embedding: List[float]) -> str:
        """Serializa usando JSON directo (compatible pero grande)."""
        json_str = json.dumps([float(x) for x in embedding])
        return f"json:{json_str}"
    
    @staticmethod
    def _serialize_compressed(embedding: List[float]) -> str:
        """Serializa comprimiendo los datos."""
        arr = np.array(embedding, dtype=np.float32)
        
        # Convertir a bytes
        bytes_data = arr.tobytes()
        
        # Comprimir
        compressed = zlib.compress(bytes_data, level=9)
        
        # Codificar base64
        encoded = base64.b64encode(compressed).decode('utf-8')
        return f"compressed:{encoded}"
    
    # ============================================================================
    # 🔥 MÉTODOS DE DESERIALIZACIÓN MEJORADOS
    # ============================================================================
    
    @staticmethod
    def deserialize_embedding(embedding_str: str) -> Optional[List[float]]:
        """
        Convierte string a embedding.
        
        Args:
            embedding_str: String serializado
            
        Returns:
            Lista de floats o None si hay error
        """
        if not embedding_str:
            return None
        
        try:
            # Detectar formato
            if embedding_str.startswith("b64pickle:"):
                return EmbeddingSerializer._deserialize_b64pickle(embedding_str)
            elif embedding_str.startswith("b64json:"):
                return EmbeddingSerializer._deserialize_b64json(embedding_str)
            elif embedding_str.startswith("json:"):
                return EmbeddingSerializer._deserialize_json(embedding_str)
            elif embedding_str.startswith("compressed:"):
                return EmbeddingSerializer._deserialize_compressed(embedding_str)
            else:
                # Intentar detectar automáticamente
                return EmbeddingSerializer._deserialize_auto(embedding_str)
                
        except Exception as e:
            logger.error(f"Error deserializando embedding: {e}")
            return None
    
    @staticmethod
    def _deserialize_b64pickle(embedding_str: str) -> List[float]:
        """Deserializa Base64 + Pickle."""
        encoded = embedding_str[len("b64pickle:"):]
        pickled = base64.b64decode(encoded.encode('utf-8'))
        arr = pickle.loads(pickled)
        return arr.tolist()
    
    @staticmethod
    def _deserialize_b64json(embedding_str: str) -> List[float]:
        """Deserializa Base64 + JSON."""
        encoded = embedding_str[len("b64json:"):]
        json_str = base64.b64decode(encoded.encode('utf-8')).decode('utf-8')
        return json.loads(json_str)
    
    @staticmethod
    def _deserialize_json(embedding_str: str) -> List[float]:
        """Deserializa JSON directo."""
        json_str = embedding_str[len("json:"):]
        return json.loads(json_str)
    
    @staticmethod
    def _deserialize_compressed(embedding_str: str) -> List[float]:
        """Deserializa datos comprimidos."""
        encoded = embedding_str[len("compressed:"):]
        compressed = base64.b64decode(encoded.encode('utf-8'))
        bytes_data = zlib.decompress(compressed)
        
        # Reconstruir numpy array
        arr = np.frombuffer(bytes_data, dtype=np.float32)
        return arr.tolist()
    
    @staticmethod
    def _deserialize_auto(embedding_str: str) -> Optional[List[float]]:
        """Intenta deserializar automáticamente."""
        # Si parece JSON
        if embedding_str.startswith('[') and embedding_str.endswith(']'):
            try:
                return json.loads(embedding_str)
            except:
                pass
        
        # Si es largo, podría ser base64
        if len(embedding_str) > 50:
            try:
                # Intentar como base64 pickle
                pickled = base64.b64decode(embedding_str.encode('utf-8'))
                arr = pickle.loads(pickled)
                return arr.tolist()
            except:
                try:
                    # Intentar como base64 json
                    json_str = base64.b64decode(embedding_str.encode('utf-8')).decode('utf-8')
                    return json.loads(json_str)
                except:
                    pass
        
        # No se pudo deserializar
        return None
    
    # ============================================================================
    # 🔥 MÉTODOS DE OPTIMIZACIÓN MEJORADOS
    # ============================================================================
    
    @staticmethod
    def optimize_for_storage(product_dict: dict, compress_large: bool = True) -> dict:
        """
        Optimiza diccionario de producto para almacenamiento.
        
        Args:
            product_dict: Diccionario del producto
            compress_large: Si True, comprime embeddings grandes
            
        Returns:
            Diccionario optimizado
        """
        optimized = product_dict.copy()
        
        # Optimizar embedding
        if 'embedding' in optimized:
            embedding = optimized['embedding']
            
            # Validar antes de optimizar
            if EmbeddingSerializer.validate_embedding_structure(embedding):
                # Si es string ya serializado, mantenerlo
                if isinstance(embedding, str):
                    format_type, _ = EmbeddingSerializer.detect_embedding_format(embedding)
                    if format_type != EmbeddingFormat.UNKNOWN:
                        optimized['embedding_format'] = format_type.value
                
                # Si es lista/numpy, serializar
                elif isinstance(embedding, (list, np.ndarray)):
                    # Seleccionar método basado en tamaño
                    if compress_large and len(embedding) > 500:
                        method = "compressed"
                    else:
                        method = "b64pickle"
                    
                    serialized = EmbeddingSerializer.serialize_embedding(
                        embedding if isinstance(embedding, list) else embedding.tolist(),
                        method=method
                    )
                    if serialized:
                        optimized['embedding'] = serialized
                        optimized['embedding_format'] = method
                        optimized['embedding_compressed'] = True
            else:
                # Embedding no válido, eliminarlo
                logger.warning(f"Embedding no válido en producto {optimized.get('id', 'unknown')}, eliminando")
                optimized.pop('embedding', None)
        
        # Optimizar otros campos grandes
        large_fields = ['extracted_entities', 'ml_features', 'metadata', 'features']
        for field in large_fields:
            if field in optimized and optimized[field]:
                if isinstance(optimized[field], (dict, list)):
                    try:
                        optimized[field] = json.dumps(optimized[field])
                        optimized[f'{field}_json'] = True
                    except Exception as e:
                        logger.debug(f"Error serializando {field}: {e}")
        
        return optimized
    
    @staticmethod
    def restore_from_storage(product_dict: dict) -> dict:
        """
        Restaura diccionario de producto desde almacenamiento.
        
        Args:
            product_dict: Diccionario optimizado
            
        Returns:
            Diccionario restaurado
        """
        restored = product_dict.copy()
        
        # Restaurar embedding
        if 'embedding' in restored and isinstance(restored['embedding'], str):
            embedding = EmbeddingSerializer.deserialize_embedding(restored['embedding'])
            if embedding:
                restored['embedding'] = embedding
            else:
                # Si no se puede deserializar, eliminarlo
                restored.pop('embedding', None)
        
        # Eliminar campos de formato
        restored.pop('embedding_format', None)
        restored.pop('embedding_compressed', None)
        
        # Restaurar otros campos JSON
        json_fields = ['extracted_entities', 'ml_features', 'metadata', 'features']
        for field in json_fields:
            json_flag = f'{field}_json'
            if json_flag in restored and restored[json_flag]:
                if field in restored and isinstance(restored[field], str):
                    try:
                        restored[field] = json.loads(restored[field])
                    except:
                        pass  # Mantener como string si falla
                restored.pop(json_flag, None)
        
        return restored
    
    # ============================================================================
    # 🔥 MÉTODOS DE DIAGNÓSTICO
    # ============================================================================
    
    @staticmethod
    def diagnose_embedding(embedding: Any) -> Dict[str, Any]:
        """
        Diagnóstico completo de un embedding.
        
        Args:
            embedding: El embedding a diagnosticar
            
        Returns:
            Diccionario con información de diagnóstico
        """
        result = {
            'exists': embedding is not None,
            'valid': False,
            'format': 'unknown',
            'dimensions': 0,
            'issues': []
        }
        
        if embedding is None:
            result['issues'].append('Embedding es None')
            return result
        
        # Detectar formato
        format_type, processed = EmbeddingSerializer.detect_embedding_format(embedding)
        result['format'] = format_type.value
        
        # Validar estructura
        if EmbeddingSerializer.validate_embedding_structure(embedding):
            result['valid'] = True
            
            # Obtener dimensiones
            try:
                if format_type == EmbeddingFormat.LIST:
                    result['dimensions'] = len(processed)
                elif format_type == EmbeddingFormat.NUMPY:
                    result['dimensions'] = processed.size
                elif format_type in [EmbeddingFormat.BASE64_PICKLE, 
                                   EmbeddingFormat.BASE64_JSON,
                                   EmbeddingFormat.JSON,
                                   EmbeddingFormat.COMPRESSED]:
                    deserialized = EmbeddingSerializer.deserialize_embedding(processed)
                    if deserialized:
                        result['dimensions'] = len(deserialized)
            except:
                result['dimensions'] = 0
        else:
            # Diagnosticar problemas
            if isinstance(embedding, str):
                if len(embedding) < EmbeddingValidation.MIN_SERIALIZED_LENGTH:
                    result['issues'].append(f'String muy corto: {len(embedding)}')
                if len(embedding) > EmbeddingValidation.MAX_SERIALIZED_LENGTH:
                    result['issues'].append(f'String muy largo: {len(embedding)}')
            
            elif isinstance(embedding, list):
                if len(embedding) < EmbeddingValidation.MIN_DIMENSIONS:
                    result['issues'].append(f'Lista muy corta: {len(embedding)}')
                if len(embedding) > EmbeddingValidation.MAX_DIMENSIONS:
                    result['issues'].append(f'Lista muy larga: {len(embedding)}')
            
            elif isinstance(embedding, np.ndarray):
                if embedding.size < EmbeddingValidation.MIN_DIMENSIONS:
                    result['issues'].append(f'Array muy pequeño: {embedding.size}')
                if embedding.size > EmbeddingValidation.MAX_DIMENSIONS:
                    result['issues'].append(f'Array muy grande: {embedding.size}')
        
        return result
    
    @staticmethod
    def compare_embeddings(embedding1: Any, embedding2: Any) -> Dict[str, Any]:
        """
        Compara dos embeddings.
        
        Returns:
            Diccionario con resultados de comparación
        """
        # Deserializar si es necesario
        if isinstance(embedding1, str):
            embedding1 = EmbeddingSerializer.deserialize_embedding(embedding1)
        if isinstance(embedding2, str):
            embedding2 = EmbeddingSerializer.deserialize_embedding(embedding2)
        
        if embedding1 is None or embedding2 is None:
            return {'similarity': 0.0, 'comparable': False, 'reason': 'Uno o ambos embeddings son None'}
        
        # Convertir a numpy arrays
        try:
            arr1 = np.array(embedding1, dtype=np.float32)
            arr2 = np.array(embedding2, dtype=np.float32)
            
            # Verificar dimensiones
            if arr1.shape != arr2.shape:
                return {
                    'similarity': 0.0,
                    'comparable': False,
                    'reason': f'Dimensiones diferentes: {arr1.shape} != {arr2.shape}'
                }
            
            # Calcular similitud coseno
            norm1 = np.linalg.norm(arr1)
            norm2 = np.linalg.norm(arr2)
            
            if norm1 == 0 or norm2 == 0:
                return {'similarity': 0.0, 'comparable': False, 'reason': 'Uno o ambos embeddings son cero'}
            
            similarity = np.dot(arr1, arr2) / (norm1 * norm2)
            
            return {
                'similarity': float(similarity),
                'comparable': True,
                'dimensions': len(arr1),
                'norm1': float(norm1),
                'norm2': float(norm2)
            }
            
        except Exception as e:
            return {'similarity': 0.0, 'comparable': False, 'reason': str(e)}


# ============================================================================
# 🔥 ALIASES Y FUNCIONES DE CONVENIENCIA
# ============================================================================

# Alias para compatibilidad
serialize_embedding = EmbeddingSerializer.serialize_embedding
deserialize_embedding = EmbeddingSerializer.deserialize_embedding
validate_embedding = EmbeddingSerializer.validate_embedding
optimize_for_storage = EmbeddingSerializer.optimize_for_storage
restore_from_storage = EmbeddingSerializer.restore_from_storage


def normalize_embedding(embedding: Any) -> Optional[List[float]]:
    """
    Normaliza cualquier formato de embedding a lista de floats.
    
    Args:
        embedding: Embedding en cualquier formato
        
    Returns:
        Lista de floats normalizada o None
    """
    if embedding is None:
        return None
    
    # Si ya es lista de floats
    if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
        return [float(x) for x in embedding]
    
    # Si es string serializado
    if isinstance(embedding, str):
        return EmbeddingSerializer.deserialize_embedding(embedding)
    
    # Si es numpy array
    if isinstance(embedding, np.ndarray):
        return embedding.astype(np.float32).tolist()
    
    # Intentar convertir
    try:
        return [float(x) for x in embedding]
    except:
        return None


def validate_product_embeddings(products: List[Dict]) -> Dict[str, Any]:
    """
    Valida embeddings en una lista de productos.
    
    Args:
        products: Lista de diccionarios de productos
        
    Returns:
        Estadísticas de validación
    """
    stats = {
        'total_products': len(products),
        'products_with_embedding': 0,
        'valid_embeddings': 0,
        'invalid_embeddings': 0,
        'embedding_formats': {},
        'common_issues': []
    }
    
    issue_counts = {}
    
    for product in products:
        if 'embedding' not in product or product['embedding'] is None:
            continue
        
        stats['products_with_embedding'] += 1
        
        # Diagnosticar embedding
        diagnosis = EmbeddingSerializer.diagnose_embedding(product['embedding'])
        
        # Contar formato
        fmt = diagnosis['format']
        stats['embedding_formats'][fmt] = stats['embedding_formats'].get(fmt, 0) + 1
        
        # Contar válidos/inválidos
        if diagnosis['valid']:
            stats['valid_embeddings'] += 1
        else:
            stats['invalid_embeddings'] += 1
            for issue in diagnosis['issues']:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    # Ordenar issues por frecuencia
    sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
    stats['common_issues'] = [{'issue': issue, 'count': count} for issue, count in sorted_issues[:5]]
    
    return stats


# ============================================================================
# 🔥 EJECUCIÓN DE PRUEBAS
# ============================================================================

if __name__ == "__main__":
    # Ejecutar pruebas si se llama directamente
    print("🧪 Probando sistema de serialización de embeddings...")
    
    # Crear embedding de prueba
    test_embedding = [0.1 * i for i in range(384)]
    
    # Probar serialización/deserialización
    for method in ["b64pickle", "b64json", "json", "compressed"]:
        print(f"\n🔧 Método: {method}")
        
        serialized = EmbeddingSerializer.serialize_embedding(test_embedding, method=method)
        print(f"   Longitud serializada: {len(serialized)}")
        
        deserialized = EmbeddingSerializer.deserialize_embedding(serialized)
        print(f"   Deserializado correctamente: {deserialized is not None}")
        
        if deserialized:
            print(f"   Dimensiones: {len(deserialized)}")
            print(f"   Válido: {EmbeddingSerializer.validate_embedding(deserialized)}")
    
    # Probar diagnóstico
    print(f"\n🔍 Diagnóstico del embedding original:")
    diagnosis = EmbeddingSerializer.diagnose_embedding(test_embedding)
    for key, value in diagnosis.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Pruebas completadas")