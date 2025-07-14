from __future__ import annotations

# src/core/data/product.py
"""
Canonical Amazon product representation.

This model is shared by:
- Indexers (vector stores)
- CategoryTree / ProductFilter
- Feedback pipeline
- RLHF training
"""

import hashlib

from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field, validator

# ------------------------------------------------------------------
# Nested models
# ------------------------------------------------------------------
class ProductImage(BaseModel):
    large: Optional[str] = None
    medium: Optional[str] = None
    small: Optional[str] = None


class ProductDetails(BaseModel):
    brand: Optional[str] = Field(None, alias="Brand")
    model: Optional[str] = Field(None, alias="Model")
    features: List[str] = Field(default_factory=list)
    specifications: Dict[str, str] = Field(default_factory=dict)


# ------------------------------------------------------------------
# Main product entity
# ------------------------------------------------------------------
class Product(BaseModel):
    id: str
    title: str
    main_category: str
    categories: List[str] = Field(default_factory=list)
    price: Optional[float] = None
    average_rating: Optional[float] = Field(None, ge=0, le=5)
    rating_count: Optional[int] = Field(None, alias="rating_number", ge=0)
    images: Optional[ProductImage] = None
    details: Optional[ProductDetails] = None

    # --------------------------------------------------
    # Validators
    # --------------------------------------------------
    @validator("price")
    def non_negative_price(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("price cannot be negative")
        return v

    # --------------------------------------------------
    # Constructors
    # --------------------------------------------------
    
    @classmethod
    def from_dict(cls, raw: Dict) -> "Product":
        """Build Product from raw dict (handles nested objects and aliases)."""

        # 1. Crear un id si no existe
        if "id" not in raw:
            base = (raw.get("title") or "") + (raw.get("main_category") or "")
            raw["id"] = hashlib.md5(base.encode("utf-8")).hexdigest()

        # 2. Convertir images: list → ProductImage
        if isinstance(raw.get("images"), list) and raw["images"]:
            main = raw["images"][0]
            raw["images"] = {
                "large": main.get("large"),
                "medium": main.get("thumb"),
                "small": main.get("thumb"),
            }
        else:
            raw["images"] = {
                "large": None,
                "medium": None,
                "small": None,
            }

        # 3. Normalizar details
        if isinstance(raw.get("details"), dict):
            details = raw["details"]
            extracted = {
                "Brand": details.get("Brand") or details.get("brand"),
                "Model": details.get("Model") or details.get("model"),
                "features": raw.get("features", []),
                "specifications": {
                    k: v for k, v in details.items() if k not in ["Brand", "brand", "Model", "model"]
                },
            }
            raw["details"] = extracted

        # 4. Validación Pydantic de campos anidados
        if "details" in raw:
            raw["details"] = ProductDetails(**raw["details"])
        if "images" in raw:
            raw["images"] = ProductImage(**raw["images"])

        return cls(**raw)

    # --------------------------------------------------
    # Export helpers
    # --------------------------------------------------
    def to_document(self) -> Dict[str, Any]:
        """
        Return LangChain-compatible Document dict for indexing.

        Returns
        -------
        dict
            {"page_content": str, "metadata": dict}
        """
        return {
            "page_content": self._build_content(),
            "metadata": self._build_metadata(),
        }

    def _build_content(self) -> str:
        """Human-readable textual description."""
        parts = [
            f"Título: {self.title}",
            f"Categoría: {self.main_category}",
            f"Precio: ${self.price}" if self.price is not None else "Precio: N/A",
        ]

        if self.details:
            if self.details.brand:
                parts.append(f"Marca: {self.details.brand}")
            if self.details.model:
                parts.append(f"Modelo: {self.details.model}")
            if self.details.features:
                parts.append(f"Características: {', '.join(self.details.features)}")
            if self.details.specifications:
                specs = "\n".join(f"{k}: {v}" for k, v in self.details.specifications.items())
                parts.append(f"Especificaciones:\n{specs}")

        return "\n".join(parts)

    def _build_metadata(self) -> Dict[str, Any]:
        """Flat metadata dict for vector-store filtering."""
        return {
            "id": self.id,
            "title": self.title,
            "category": self.main_category,
            "price": self.price,
            "rating": self.average_rating,
            "rating_count": self.rating_count,
            "brand": self.details.brand if self.details else None,
            **(self.details.specifications if self.details else {}),
        }

    # --------------------------------------------------
    # Dunder utilities
    # --------------------------------------------------
    def __str__(self) -> str:  # noqa: D401
        """Concise representation."""
        return f"{self.title} (${self.price}) | {self.main_category}"