from enum import Enum
from typing import Union

class BinaryScore(Enum):
    YES = "yes"
    NO = "no"

def parse_binary_score(text: str) -> BinaryScore:
    """
    Parsea entrada del usuario para feedback binario (sí/no).
    Compatible con cli.py y el sistema de logging existente.
    """
    if not text or not isinstance(text, str):
        return BinaryScore.NO

    text_lower = text.strip().lower()

    # Afirmativos
    affirmative = {'y', 'yes', 'sí', 'si', '1', 'true', 'ok', 'correcto'}
    if text_lower in affirmative:
        return BinaryScore.YES

    # Negativos
    negative = {'n', 'no', '0', 'false', 'incorrecto', 'mal'}
    if text_lower in negative:
        return BinaryScore.NO

    # Por defecto → NO
    return BinaryScore.NO

def safe_int_parse(value: Union[str, int, float], default: int = 0) -> int:
    """
    Conversión segura a entero para ratings.
    Compatible con agent.log_feedback() que espera un INT.
    """
    try:
        if isinstance(value, (int, float)):
            return int(value)
        return int(str(value).strip())
    except (ValueError, TypeError):
        return default