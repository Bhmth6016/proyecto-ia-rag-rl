# validaciones.py
import re
from typing import Optional, Union
from pydantic import BaseModel, field_validator

class ProductModel(BaseModel):
    main_category: Optional[str] = None
    title: Optional[str] = None
    average_rating: Optional[float] = None
    price: Optional[float] = None
    images: Optional[dict] = {}
    categories: Optional[list] = []
    details: Optional[dict] = {}

    @field_validator('price', mode='before')
    def parse_price(cls, value: Optional[Union[str, int, float]]) -> Optional[float]:
        """Convierte un precio en str, int o float a float válido."""
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value) if value >= 0 else None

        # Limpieza de strings (ej: "$10.99" → 10.99)
        price_str = str(value).lower()
        price_str = re.sub(r'[^\d.]', '', price_str)
        return float(price_str) if price_str else None

    @field_validator('average_rating', mode='before')
    def parse_rating(cls, value: Optional[Union[str, int, float]]) -> Optional[float]:
        """Valida que el rating esté entre 0 y 5."""
        if value is None or isinstance(value, bool):
            return None
        try:
            rating = float(value)
            return rating if 0 <= rating <= 5 else None
        except (ValueError, TypeError):
            return None

    @field_validator('categories', mode='before')
    def parse_categories(cls, value: Optional[list]) -> list:
        """Limpia y valida la lista de categorías."""
        if value is None:
            return []
        if not isinstance(value, (list, tuple, set)):
            return []
        return [str(cat).strip() for cat in value if cat is not None]

    @field_validator('details', mode='before')
    def parse_details(cls, value: Optional[dict]) -> dict:
        """Limpia y valida los detalles del producto."""
        if value is None or not isinstance(value, dict):
            return {}
        return {
            str(k).strip(): str(v).strip() if v is not None else ""
            for k, v in value.items()
            if k is not None
        }

def clean_string(value: Optional[Union[str, int, float]]) -> Optional[str]:
    """Limpia strings (elimina espacios, convierte a str)."""
    if value is None:
        return None
    try:
        cleaned = str(value).strip()
        return cleaned if cleaned else None
    except Exception:
        return None