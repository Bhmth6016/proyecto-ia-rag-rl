# src/core/utils/validators.py
"""
Unified validation utilities built on Pydantic v2.

Features
--------
• Domain-aware validators (Product, Document, etc.)
• Runtime validation without boilerplate
• Decorator for automatic error logging
• JSON-serialisable results
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Dict, List, Optional, TypeVar, Callable

from pydantic import BaseModel, ValidationError, Field, validator

from src.core.utils.logger import get_logger
from src.core.data.product import Product

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------
class ValidationResult(BaseModel):
    """Standard return type for all validators."""
    is_valid: bool
    errors: List[str] = []
    data: Optional[Any] = None


# ------------------------------------------------------------------
# Domain models
# ------------------------------------------------------------------
class ValidatedProduct(BaseModel):
    """Pydantic schema for a product coming from any source."""
    title: str
    main_category: str
    details: Optional[Dict[str, Any]] = None

    @validator("title", pre=True)
    def _clean_title(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("title must be a non-empty string")
        return v.strip()

    @validator("main_category", pre=True)
    def _clean_category(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("main_category must be a non-empty string")
        return v.strip()

    class Config:
        extra = "allow"  # accept unknown keys gracefully


class ValidatedDocument(BaseModel):
    """Pydantic schema for LangChain documents."""
    page_content: str
    metadata: Dict[str, Any]

    @validator("page_content", pre=True)
    def _non_empty_content(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("page_content must be a non-empty string")
        return v.strip()

    @validator("metadata", pre=True)
    def _dict_metadata(cls, v: Any) -> Dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError("metadata must be a dictionary")
        return v


# ------------------------------------------------------------------
# Convenience validators
# ------------------------------------------------------------------
def validate_product_dict(data: Dict[str, Any]) -> ValidationResult:
    """Validate a raw dictionary against the product schema."""
    try:
        product = ValidatedProduct(**data)
        return ValidationResult(is_valid=True, data=product.dict())
    except ValidationError as e:
        return ValidationResult(is_valid=False, errors=[str(err) for err in e.errors()])


def validate_document_dict(data: Dict[str, Any]) -> ValidationResult:
    """Validate a raw dictionary against the document schema."""
    try:
        doc = ValidatedDocument(**data)
        return ValidationResult(is_valid=True, data=doc.dict())
    except ValidationError as e:
        return ValidationResult(is_valid=False, errors=[str(err) for err in e.errors()])


# ------------------------------------------------------------------
# Legacy dict validators (kept for backward-compatibility)
# ------------------------------------------------------------------
def validate_product_legacy(data: Dict[str, Any]) -> ValidationResult:
    """Legacy validator for dicts without Pydantic models."""
    errors = []

    required = {"title", "main_category"}
    missing = required - data.keys()
    if missing:
        errors.append(f"Missing required keys: {', '.join(missing)}")

    if not isinstance(data.get("title"), str) or not data.get("title", "").strip():
        errors.append("title must be a non-empty string")

    if not isinstance(data.get("main_category"), str) or not data.get("main_category", "").strip():
        errors.append("main_category must be a non-empty string")

    details = data.get("details")
    if details is not None and not isinstance(details, dict):
        errors.append("details must be a dictionary if provided")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        data=data if not errors else None,
    )


def validate_document_legacy(data: Dict[str, Any]) -> ValidationResult:
    """Legacy validator for document dicts."""
    errors = []
    if "page_content" not in data or not str(data.get("page_content", "")).strip():
        errors.append("page_content missing or empty")
    if "metadata" not in data or not isinstance(data.get("metadata"), dict):
        errors.append("metadata missing or not a dict")
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        data=data if not errors else None,
    )


# ------------------------------------------------------------------
# Decorator
# ------------------------------------------------------------------
def validation_handler(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that logs ValidationError or any other exception
    and returns False on failure (handy for CLI scripts).
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            logger.error("ValidationError in %s: %s", func.__name__, e.errors())
            return False
        except Exception as e:
            logger.exception("Unexpected error in %s", func.__name__)
            return False
    return wrapper


# ------------------------------------------------------------------
# Quick factory
# ------------------------------------------------------------------
def pydantic_validator(model: type[T]) -> Callable[[Dict[str, Any]], ValidationResult]:
    """
    Create a validator function for any Pydantic model at runtime.

    >>> validate_foo = pydantic_validator(MyModel)
    >>> result = validate_foo({"x": 1})
    """
    def _validate(data: Dict[str, Any]) -> ValidationResult:
        try:
            instance = model(**data)
            return ValidationResult(is_valid=True, data=instance.dict())
        except ValidationError as e:
            return ValidationResult(is_valid=False, errors=[str(err) for err in e.errors()])

    return _validate