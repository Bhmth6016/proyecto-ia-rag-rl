# src/core/data/product.py

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List


class ProductImage(BaseModel):
    large: Optional[str]
    medium: Optional[str]
    small: Optional[str]


class ProductDetails(BaseModel):
    brand: Optional[str] = Field(None, alias="Brand")
    model: Optional[str] = Field(None, alias="Model")
    features: Optional[List[str]] = Field(default_factory=list)
    specifications: Optional[Dict[str, str]] = Field(default_factory=dict)


class Product(BaseModel):
    id: str
    title: str
    main_category: str
    categories: List[str] = Field(default_factory=list)
    price: Optional[float]
    average_rating: Optional[float] = Field(None, ge=0, le=5)
    rating_count: Optional[int] = Field(None, alias="rating_number", ge=0)
    images: Optional[ProductImage]
    details: Optional[ProductDetails]

    @validator('price')
    def validate_price(cls, v):
        if v is not None and v < 0:
            raise ValueError("Price cannot be negative")
        return
    
    @classmethod
    def from_raw(cls, raw: dict) -> "Product":
        # Si hay campos anidados como 'details' o 'images', se convierten también
        if 'details' in raw and isinstance(raw['details'], dict):
            raw['details'] = ProductDetails(**raw['details'])

        if 'images' in raw and isinstance(raw['images'], dict):
            raw['images'] = ProductImage(**raw['images'])

        return cls(**raw)

        

    def to_document(self) -> Dict:
        """Convierte el producto a formato documento para indexación"""
        return {
            "page_content": self._generate_content(),
            "metadata": self._generate_metadata()
        }

    def _generate_content(self) -> str:
        parts = [
            f"Título: {self.title}",
            f"Categoría: {self.main_category}",
            f"Precio: {self.price if self.price is not None else 'N/A'}"
        ]

        if self.details:
            if self.details.brand:
                parts.append(f"Marca: {self.details.brand}")
            if self.details.model:
                parts.append(f"Modelo: {self.details.model}")
            if self.details.features:
                parts.append(f"Características: {', '.join(self.details.features)}")
            if self.details.specifications:
                specs_text = "\n".join(
                    f"{k}: {v}" for k, v in self.details.specifications.items()
                )
                parts.append(f"Specs:\n{specs_text}")

        return "\n".join(parts)

    def _generate_metadata(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.main_category,
            "price": self.price,
            "rating": self.average_rating,
            "rating_count": self.rating_count
        }
