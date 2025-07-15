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

import re
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
    main_category: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    price: Optional[float] = None
    average_rating: Optional[float] = Field(None, ge=0, le=5)
    rating_count: Optional[int] = Field(None, alias="rating_number", ge=0)
    images: Optional[ProductImage] = None
    details: Optional[ProductDetails] = None

    # --- NEW FIELDS -------------------------------------------------
    product_type: Optional[str] = None
    compatible_devices: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    attributes: Dict[str, str] = Field(default_factory=dict)

    # --------------------------------------------------
    # Validators
    # --------------------------------------------------
    @validator("price")
    def non_negative_price(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("price cannot be negative")
        return v

    # --- NEW VALIDATORS --------------------------------------------
    _SPANISH_TO_ENGLISH = {
        "mochila": "backpack",
        "bolso": "bag",
        "maleta": "luggage",
        "auriculares": "headphones",
        "altavoz": "speaker",
        "teclado": "keyboard",
        "ratón": "mouse",
        "monitor": "monitor",
        "cámara": "camera",
    }

    @validator("product_type", pre=True)
    def normalize_product_type(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        v = v.lower().strip()
        v = cls._SPANISH_TO_ENGLISH.get(v, v)
        return v.title()

    @validator("tags", pre=True)
    def normalize_tags(cls, v: List[str]) -> List[str]:
        if not isinstance(v, list):
            return []
        normalized = []
        for tag in v:
            tag = tag.lower().strip()
            tag = cls._SPANISH_TO_ENGLISH.get(tag, tag)
            normalized.append(tag.title())
        return normalized

    @validator("compatible_devices", pre=True)
    def normalize_compatible_devices(cls, v: List[str]) -> List[str]:
        if not isinstance(v, list):
            v = []
        devices = []
        for item in v:
            item = item.lower().strip()
            if item in {"laptop", "laptops", "portátil", "ordenador portátil"}:
                devices.append("Laptop")
            elif item in {"tablet", "tablets"}:
                devices.append("Tablet")
            elif item in {"smartphone", "phone", "móvil", "teléfono"}:
                devices.append("Smartphone")
            else:
                devices.append(item.title())
        return sorted(set(devices))

    @validator("attributes", pre=True)
    def extract_attributes(cls, v: Dict[str, str]) -> Dict[str, str]:
        if not isinstance(v, dict):
            v = {}
        return v

    # --------------------------------------------------
    # Constructors
    # --------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Dict) -> "Product":
        """Build Product from raw dict (handles nested objects and aliases)."""

        # 1. Create id if it doesn't exist
        if "id" not in raw:
            base = (raw.get("title") or "") + (raw.get("main_category") or "")
            raw["id"] = hashlib.md5(base.encode("utf-8")).hexdigest()

        # 2. Convert images: list → ProductImage
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

        # 3. Normalize details
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

        # 4. Extract key attributes automatically
        specs = (raw.get("details") or {}).get("specifications") or {}
        raw.setdefault("attributes", {})
        raw.setdefault("compatible_devices", [])

        # Handle description - convert list to string if needed
        description = raw.get("description", "")
        if isinstance(description, list):
            description = " ".join(desc for desc in description if isinstance(desc, str))
        
        # 4a. Build text blob for analysis
        text_blob = " ".join([
            raw.get("title", ""), 
            description,
            *[str(v) for v in specs.values()]
        ]).lower()

        # 4b. Search for sizes
        for k, v in specs.items():
            if "size" in k.lower() or "dimension" in k.lower():
                match = re.search(r'(\d+(?:\.\d+)?(?:\s?-?\s?\d+)?(?:\s?inch|in|"))', str(v), re.IGNORECASE)
                if match:
                    raw["attributes"]["screen_size"] = match.group(0).strip()
            elif "pulgadas" in str(v).lower():
                match = re.search(r'(\d+(?:\.\d+)?)\s*pulgadas?', str(v), re.IGNORECASE)
                if match:
                    raw["attributes"]["screen_size"] = f"{match.group(1).strip()}-inch"

        # 4c. Infer compatible devices
        if any(kw in text_blob for kw in ["laptop", "notebook", "portátil"]):
            raw["compatible_devices"].append("laptop")
        if any(kw in text_blob for kw in ["tablet", "ipad"]):
            raw["compatible_devices"].append("tablet")
        if any(kw in text_blob for kw in ["smartphone", "phone", "teléfono", "móvil"]):
            raw["compatible_devices"].append("smartphone")

        # 4d. Infer product_type if empty
        if not raw.get("product_type"):
            title_lower = raw.get("title", "").lower()
            for sp, en in cls._SPANISH_TO_ENGLISH.items():
                if sp in title_lower:
                    raw["product_type"] = en
                    break

        # 5. Pydantic validation of nested fields
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

        if self.product_type:
            parts.append(f"Tipo de producto: {self.product_type.title()}")
        if self.compatible_devices:
            parts.append(f"Dispositivos compatibles: {', '.join(self.compatible_devices)}")
        if self.tags:
            parts.append(f"Etiquetas: {', '.join(self.tags)}")
        if self.attributes:
            specs = "\n".join(f"{k}: {v}" for k, v in self.attributes.items())
            parts.append(f"Atributos:\n{specs}")

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
            "product_type": self.product_type,
            "compatible_devices": self.compatible_devices,
            "tags": self.tags,
            **(self.attributes or {}),
            **(self.details.specifications if self.details else {}),
        }

    @staticmethod
    def clean_url(url_str):
        """Clean URL from HTML markup"""
        import re
        if not url_str:
            return ""
        match = re.search(r'https?://[^\s<>"\']+', str(url_str))
        return match.group(0) if match else ""

    def clean_image_urls(self):
        """Clean all image URLs in the product"""
        if hasattr(self, 'images') and self.images:
            for key in ['thumb', 'large', 'hi_res', 'medium', 'small']:
                if hasattr(self.images, key) and getattr(self.images, key):
                    cleaned = self.clean_url(getattr(self.images, key))
                    setattr(self.images, key, cleaned)

    # --------------------------------------------------
    # Dunder utilities
    # --------------------------------------------------
    def __str__(self) -> str:  # noqa: D401
        """Concise representation."""
        return f"{self.title} (${self.price}) | {self.main_category}"