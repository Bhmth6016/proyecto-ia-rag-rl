from __future__ import annotations
# src/core/data/product.py
import hashlib
import re
from typing import Optional, Dict, List, Any, ClassVar
from pydantic import BaseModel, Field, validator
from langchain_core.documents import Document
import uuid
import logging

logger = logging.getLogger(__name__)

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
    color: Optional[str] = None
    weight: Optional[str] = None
    wireless: Optional[bool] = None
    bluetooth: Optional[bool] = None

    @validator('specifications', pre=True)
    def normalize_specifications(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        for key, value in v.items():
            key = key.strip().lower()
            
            # Normalize common attributes
            if 'color' in key:
                normalized['color'] = str(value)
            elif 'weight' in key:
                normalized['weight'] = str(value)
            elif any(w in key for w in ['wireless', 'inalámbrico']):
                normalized['wireless'] = bool(value)
            elif 'bluetooth' in key:
                normalized['bluetooth'] = bool(value)
            else:
                normalized[key] = str(value) if value is not None else ""
                
        return normalized

    @classmethod
    def safe_create(cls, details_data: Optional[Dict]) -> "ProductDetails":
        """Crea una instancia de ProductDetails manejando datos inválidos"""
        if not details_data or not isinstance(details_data, dict):
            return cls()  # Retorna una instancia con valores por defecto
        return cls(**details_data)

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
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # ID por defecto
    title: str = "Unknown Product"  # Valor por defecto
    main_category: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    price: Optional[float] = None
    average_rating: Optional[float] = Field(None, ge=0, le=5)
    rating_count: Optional[int] = Field(None, alias="rating_number", ge=0)
    images: Optional[ProductImage] = None
    details: Optional[ProductDetails] = None
    product_type: Optional[str] = None
    compatible_devices: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    attributes: Dict[str, str] = Field(default_factory=dict)
    description: Optional[str] = None

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
        """Versión más robusta del constructor."""
        try:
            # Asegurar campos mínimos
            raw.setdefault('title', 'Unknown Product')
            raw.setdefault('id', str(uuid.uuid4()))
            
            # Manejar detalles vacíos
            if 'details' not in raw or raw['details'] is None:
                raw['details'] = {}
                
            # Create ProductDetails instance safely
            details = ProductDetails.safe_create(raw.get('details', {}))
            raw['details'] = details.dict()

            # Normalize details
            if isinstance(details.specifications.get("Best Sellers Rank"), dict):
                details.specifications["Best Sellers Rank"] = ", ".join(
                    f"{k}: {v}" for k, v in details.specifications["Best Sellers Rank"].items()
                )

            # Extract key attributes automatically
            specs = raw['details'].get('specifications', {})
            raw.setdefault('attributes', {})
            raw.setdefault('compatible_devices', [])

            # Convert description list to string if needed
            description = raw.get('description', '')
            if isinstance(description, list):
                description = ' '.join(filter(lambda x: isinstance(x, str), description))
            raw['description'] = description

            # Build text blob
            text_blob = ' '.join([
                raw.get('title', ''),
                raw.get('description', ''),
                *[str(v) for v in specs.values()]
            ]).lower()

            # Search for sizes
            for k, v in specs.items():
                if 'size' in k.lower() or 'dimension' in k.lower():
                    match = re.search(r'(\d+(?:\.\d+)?(?:\s?-?\s?\d+)?(?:\s?inch|in|"))', str(v), re.IGNORECASE)
                    if match:
                        raw['attributes']['screen_size'] = match.group(0).strip()
                elif 'pulgadas' in str(v).lower():
                    match = re.search(r'(\d+(?:\.\d+)?)\s*pulgadas?', str(v), re.IGNORECASE)
                    if match:
                        raw['attributes']['screen_size'] = f"{match.group(1).strip()}-inch"

            # Infer compatible devices
            if any(kw in text_blob for kw in ["laptop", "notebook", "portátil"]):
                raw['compatible_devices'].append("laptop")
            if any(kw in text_blob for kw in ["tablet", "ipad"]):
                raw['compatible_devices'].append("tablet")
            if any(kw in text_blob for kw in ["smartphone", "phone", "teléfono", "móvil"]):
                raw['compatible_devices'].append("smartphone")

            # Infer product_type if empty
            if not raw.get('product_type'):
                title_lower = raw.get('title', '').lower()
                for sp, en in cls.get_spanish_to_english().items():
                    if sp in title_lower:
                        raw['product_type'] = en
                        break

            # Ensure details is a ProductDetails object
            raw['details'] = ProductDetails(**raw['details'])

            # Ignore images if present
            raw['images'] = None

            return cls(**raw)
        except Exception as e:
            logger.warning(f"Error creating Product: {e}")
            # Create product with minimal required fields
            return cls(
                title=raw.get('title', 'Unknown Product'),
                id=raw.get('id', str(uuid.uuid4()))
            )

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
    
    def to_text(self) -> str:
        """Improved representation for embedding"""
        price = str(self.price) if isinstance(self.price, (int, float)) else "Price not available"
        rating = str(self.average_rating) if isinstance(self.average_rating, (int, float)) else "No rating available"
        
        parts = [
            self.title,
            f"Category: {self.main_category}",
            self.description or "No description available",
            f"Price: {price}",
            f"Rating: {rating}",
            f"Features: {', '.join(self.details.features)}" if self.details and self.details.features else "No features available",
            f"Type: {self.product_type}" if self.product_type else "No type specified",
            " ".join(self.tags) if self.tags else "No tags",
            " ".join(self.compatible_devices) if self.compatible_devices else "No compatible devices specified"
        ]
        return " ".join(filter(None, parts))

    def to_metadata(self) -> dict:
        """Return all essential metadata for retrieval"""
        import json  # Add this line to ensure json is available
        
        try:
            return {
                "id": self.id,
                "title": self.title or "Untitled Product",
                "main_category": self.main_category or "Uncategorized",
                "categories": json.dumps(self.categories) if self.categories else "[]",
                "price": float(self.price) if self.price is not None else 0.0,
                "average_rating": float(self.average_rating) if self.average_rating else 0.0,
                "description": self.description or "",
                "product_type": self.product_type or "",
                "features": json.dumps(self.details.features) if self.details else "[]"
            }
        except Exception as e:
            logger.error(f"Error converting product to metadata: {e}")
            return {
                "id": self.id,
                "title": self.title or "Untitled Product",
                "main_category": "Uncategorized",
                "categories": "[]",
                "price": 0.0,
                "average_rating": 0.0,
                "description": "",
                "product_type": "",
                "features": "[]"
            }
    
    def to_document(self) -> Document:
        """Convierte el producto a un Document optimizado para almacenamiento en Chroma o LangChain."""
        return Document(
            page_content=self.to_text(),  # Contenido textual principal
            metadata={
                "id": self.id,
                "title": self.title[:100],  # Limita a 100 caracteres
                "price": float(self.price) if self.price is not None else 0.0,
                "rating": float(self.average_rating) if self.average_rating is not None else 0.0,
            }
        )