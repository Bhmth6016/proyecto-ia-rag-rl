# src/core/utils/parsers.py
from typing import Union, Optional, Dict, Any, List
import re
from enum import Enum
import logging
from pydantic import BaseModel, validator

logger = logging.getLogger(__name__)

class BinaryScore(str, Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"

class ParsingResult(BaseModel):
    success: bool
    value: Optional[Any] = None
    error: Optional[str] = None
    original: Optional[str] = None

def parse_binary_score(text: str, strict: bool = False) -> BinaryScore:
    """
    Parsea texto a resultado binario estandarizado con opción de modo estricto.
    
    Args:
        text: Texto a parsear
        strict: Si True, solo acepta "yes"/"no" exactos
        
    Returns:
        BinaryScore (Enum): YES, NO o UNKNOWN
    """
    if not isinstance(text, str):
        logger.warning(f"Input no es texto: {type(text)}")
        return BinaryScore.UNKNOWN
    
    clean_text = text.strip().lower()
    
    if strict:
        if clean_text == "yes":
            return BinaryScore.YES
        if clean_text == "no":
            return BinaryScore.NO
        return BinaryScore.UNKNOWN
    
    # Lógica no estricta (por defecto)
    if "yes" in clean_text:
        return BinaryScore.YES
    if "no" in clean_text:
        return BinaryScore.NO
    
    # Detección de negaciones
    negations = {"nunca", "no", "not", "ningún", "ninguna", "incorrecto"}
    if any(neg in clean_text for neg in negations):
        return BinaryScore.NO
        
    return BinaryScore.UNKNOWN

def parse_score_range(
    text: str,
    min_val: Union[int, float],
    max_val: Union[int, float],
    default: Optional[Union[int, float]] = None
) -> ParsingResult:
    """
    Extrae puntuación numérica de texto con manejo robusto de errores.
    
    Args:
        text: Texto a analizar
        min_val: Valor mínimo aceptable
        max_val: Valor máximo aceptable
        default: Valor por defecto si no se puede parsear (opcional)
        
    Returns:
        ParsingResult con:
        - success: bool
        - value: Valor parseado (si success)
        - error: Mensaje de error (si no success)
        - original: Texto original
    """
    if default is None:
        default = min_val
    
    try:
        # Buscar números en el texto
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        
        for num_str in numbers:
            try:
                score = float(num_str)
                if min_val <= score <= max_val:
                    return ParsingResult(
                        success=True,
                        value=score,
                        original=text
                    )
            except ValueError:
                continue
                
        return ParsingResult(
            success=False,
            value=default,
            error=f"No se encontró valor en rango {min_val}-{max_val}",
            original=text
        )
        
    except Exception as e:
        logger.error(f"Error parseando score: {str(e)}")
        return ParsingResult(
            success=False,
            value=default,
            error=str(e),
            original=text
        )

def parse_list(text: str, delimiter: str = ",") -> ParsingResult:
    """
    Parsea texto a lista, manejando diferentes formatos.
    
    Args:
        text: Texto con elementos separados
        delimiter: Separador (por defecto ",")
        
    Returns:
        ParsingResult con lista parseada
    """
    try:
        if not text.strip():
            return ParsingResult(
                success=True,
                value=[],
                original=text
            )
            
        items = [item.strip() for item in text.split(delimiter) if item.strip()]
        return ParsingResult(
            success=True,
            value=items,
            original=text
        )
    except Exception as e:
        return ParsingResult(
            success=False,
            error=str(e),
            original=text
        )

def parse_dict(text: str) -> ParsingResult:
    """
    Intenta parsear texto como diccionario JSON.
    
    Args:
        text: Texto en formato JSON
        
    Returns:
        ParsingResult con diccionario parseado
    """
    import json
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("El texto no representa un diccionario")
            
        return ParsingResult(
            success=True,
            value=data,
            original=text
        )
    except Exception as e:
        return ParsingResult(
            success=False,
            error=str(e),
            original=text
        )

def parse_boolean(text: str) -> ParsingResult:
    """
    Parsea texto a booleano con múltiples formatos soportados.
    
    Args:
        text: Texto a evaluar
        
    Returns:
        ParsingResult con valor booleano
    """
    true_values = {"true", "yes", "si", "sí", "1", "verdadero"}
    false_values = {"false", "no", "0", "falso"}
    
    clean_text = text.strip().lower()
    
    if clean_text in true_values:
        return ParsingResult(
            success=True,
            value=True,
            original=text
        )
    elif clean_text in false_values:
        return ParsingResult(
            success=True,
            value=False,
            original=text
        )
        
    return ParsingResult(
        success=False,
        error="Valor booleano no reconocido",
        original=text
    )

def safe_parse(parse_func, text: str, *args, **kwargs) -> ParsingResult:
    """
    Wrapper seguro para funciones de parseo.
    
    Args:
        parse_func: Función de parseo
        text: Texto a parsear
        *args, **kwargs: Argumentos adicionales
        
    Returns:
        ParsingResult estandarizado
    """
    try:
        return parse_func(text, *args, **kwargs)
    except Exception as e:
        logger.error(f"Error en safe_parse: {str(e)}")
        return ParsingResult(
            success=False,
            error=str(e),
            original=text
        )