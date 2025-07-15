from __future__ import annotations

# src/core/data/product.py}

import re
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
    specifications: Dict[str, Any] = Field(default_factory=dict)

    @validator("specifications", pre=True)
    def normalize_specifications(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(v, dict):
            return {}
        
        normalized = {}
        for key, value in v.items():
            if key.lower() == "best sellers rank" and isinstance(value, dict):
                normalized[key] = ", ".join(f"{k}: {v}" for k, v in value.items())
            else:
                normalized[key] = str(value) if value is not None else ""
        return normalized


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
    @validator("price", pre=True)
    def parse_price(cls, v: Any) -> Optional[float]:
        if v is None or isinstance(v, float):
            return v
        if isinstance(v, str):
            # Remove common non-numeric characters and extract numbers
            v = re.sub(r'[^\d.]', '', v)
            if v:
                try:
                    return float(v)
                except ValueError:
                    pass
        return None

    @validator("price")
    def non_negative_price(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("price cannot be negative")
        return v

    # --- NEW VALIDATORS --------------------------------------------
    SPANISH_TO_ENGLISH: ClassVar[Dict[str, str]] = {
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

    @classmethod
    def get_spanish_to_english(cls) -> Dict[str, str]:
        return cls.SPANISH_TO_ENGLISH

    @validator("product_type", pre=True)
    def normalize_product_type(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        v = v.lower().strip()
        v = cls.get_spanish_to_english().get(v, v)
        return v.title()

    @validator("tags", pre=True)
    def normalize_tags(cls, v: List[str]) -> List[str]:
        if not isinstance(v, list):
            return []
        normalized = []
        for tag in v:
            tag = tag.lower().strip()
            tag = cls.get_spanish_to_english().get(tag, tag)
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
            if "Best Sellers Rank" in details and isinstance(details["Best Sellers Rank"], dict):
                details["Best Sellers Rank"] = ", ".join(
                    f"{k}: {v}" for k, v in details["Best Sellers Rank"].items()
                )
            extracted = {
                "Brand": details.get("Brand") or details.get("brand"),
                "Model": details.get("Model") or details.get("model"),
                "features": raw.get("features", []),
                "specifications": {
                    k: v for k, v in details.items() 
                    if k not in ["Brand", "brand", "Model", "model"]
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
            for sp, en in cls.get_spanish_to_english().items():
                if sp in title_lower:
                    raw["product_type"] = en
                    break

        # 5. Pydantic validation of nested fields
        if "details" in raw:
            raw["details"] = ProductDetails(**raw["details"])
        if "images" in raw:
            raw["images"] = ProductImage(**raw["images"])

        return cls(**raw)

    def clean_image_urls(self):
        """Limpia todas las URLs de imágenes en el producto."""
        if self.images:
            self.images.large = self.clean_url(self.images.large)
            self.images.medium = self.clean_url(self.images.medium)
            self.images.small = self.clean_url(self.images.small)

    @staticmethod
    def clean_url(url_str):
        """Limpia la URL de HTML."""
        if not url_str:
            return ""
        match = re.search(r'https?://[^\s<>"\']+', str(url_str))
        return match.group(0) if match else ""