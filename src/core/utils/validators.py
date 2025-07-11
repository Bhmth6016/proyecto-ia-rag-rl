# src/core/utils/validators.py
from typing import Dict, List, Any, Optional, TypeVar
from pydantic import BaseModel, ValidationError, validator, create_model
import logging
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = []
    data: Optional[Any] = None

def validate_product(product: Dict) -> ValidationResult:
    """
    Valida la estructura de un producto con mensajes detallados.
    
    Args:
        product: Diccionario con datos del producto
        
    Returns:
        ValidationResult con:
        - is_valid: bool
        - errors: Lista de mensajes de error
        - data: Datos validados (si aplica)
    """
    errors = []
    
    # Validación básica de campos requeridos
    required_fields = {"title", "main_category"}
    missing_fields = [field for field in required_fields if field not in product]
    
    if missing_fields:
        errors.append(f"Campos requeridos faltantes: {', '.join(missing_fields)}")
    
    # Validación de tipos
    if "title" in product and not isinstance(product["title"], str):
        errors.append("El título debe ser una cadena de texto")
    
    if "main_category" in product and not isinstance(product["main_category"], str):
        errors.append("La categoría principal debe ser una cadena de texto")
    
    # Validación de estructura de detalles
    if "details" in product and not isinstance(product["details"], dict):
        errors.append("Los detalles deben ser un diccionario")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        data=product if not errors else None
    )

def validate_document(doc: Dict) -> ValidationResult:
    """
    Valida un documento para sistemas RAG.
    
    Args:
        doc: Diccionario con documento LangChain
        
    Returns:
        ValidationResult con resultados de validación
    """
    errors = []
    
    # Campos requeridos
    required_fields = {"page_content", "metadata"}
    missing_fields = [field for field in required_fields if field not in doc]
    
    if missing_fields:
        errors.append(f"Campos de documento faltantes: {', '.join(missing_fields)}")
    
    # Validación de tipos
    if "metadata" in doc and not isinstance(doc["metadata"], dict):
        errors.append("El metadata debe ser un diccionario")
    
    # Validación adicional del contenido
    if "page_content" in doc and not doc["page_content"]:
        errors.append("El contenido de la página no puede estar vacío")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        data=doc if not errors else None
    )

def validation_handler(func):
    """
    Decorador para manejar errores de validación.
    
    Ejemplo:
    @validation_handler
    def process_product(product: dict) -> bool:
        # ... lógica de procesamiento
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            logger.error(f"Error de validación en {func.__name__}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error inesperado en {func.__name__}: {str(e)}")
            raise
    return wrapper

# Modelos Pydantic para validación estructural
class ProductBase(BaseModel):
    title: str
    main_category: str
    details: Optional[Dict[str, Any]] = None

    @validator('title')
    def title_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("El título no puede estar vacío")
        return v.strip()

class DocumentBase(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

    @validator('page_content')
    def content_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("El contenido del documento no puede estar vacío")
        return v.strip()

def validate_with_model(data: Dict, model: T) -> ValidationResult:
    """
    Valida datos contra un modelo Pydantic.
    
    Args:
        data: Datos a validar
        model: Clase del modelo Pydantic
        
    Returns:
        ValidationResult con resultados
    """
    try:
        model_instance = model(**data)
        return ValidationResult(
            is_valid=True,
            data=model_instance.dict()
        )
    except ValidationError as e:
        return ValidationResult(
            is_valid=False,
            errors=[str(err) for err in e.errors()]
        )