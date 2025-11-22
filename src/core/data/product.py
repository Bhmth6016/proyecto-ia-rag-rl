from __future__ import annotations
# src/core/data/product.py
import hashlib
import re
from typing import Optional, Dict, List, Any, ClassVar
from pydantic import BaseModel, Field, model_validator
import uuid
import logging
from functools import lru_cache
import json
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants and configuration
# ------------------------------------------------------------------
class AutoProductConfig:
    """Configuración para automatización de productos"""
    
    # Límites y configuraciones
    MAX_TITLE_LENGTH = 200
    MAX_DESCRIPTION_LENGTH = 1000
    DEFAULT_RATING = 0.0
    DEFAULT_PRICE = 0.0
    CACHE_SIZE = 1000

# ------------------------------------------------------------------
# Nested models simplificados
# ------------------------------------------------------------------
class ProductImage(BaseModel):
    """Modelo para imágenes de productos"""
    large: Optional[str] = None
    medium: Optional[str] = None
    small: Optional[str] = None
    
    # Cache para URLs validadas
    _validated_urls: ClassVar[Dict[str, bool]] = {}
    
    @classmethod
    def safe_create(cls, image_data: Optional[Dict]) -> "ProductImage":
        """Crea instancia con validación automática"""
        if not image_data or not isinstance(image_data, dict):
            return cls()
        
        # Validar URLs automáticamente
        validated_data = {}
        for size, url in image_data.items():
            if url and cls._validate_url_automated(url):
                validated_data[size] = url
        
        return cls(**validated_data)
    
    @classmethod
    def _validate_url_automated(cls, url: str) -> bool:
        """Valida URL automáticamente"""
        if url in cls._validated_urls:
            return cls._validated_urls[url]
        
        try:
            # Validación básica de formato
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                cls._validated_urls[url] = False
                return False
            
            # Verificar dominios comunes de imágenes
            valid_domains = {
                'images.amazon.com', 'm.media-amazon.com', 'example.com',
                'cdn.shopify.com', 'i.ebayimg.com', 'target.scene7.com',
                'via.placeholder.com', 'localhost', '127.0.0.1'
            }
            
            if parsed.netloc in valid_domains:
                cls._validated_urls[url] = True
                return True
            
            # Por defecto, aceptar URLs que pasen validación básica
            cls._validated_urls[url] = True
            return True
            
        except Exception:
            cls._validated_urls[url] = False
            return False
    
    def clean_urls_automated(self) -> None:
        """Limpia URLs usando técnicas automatizadas"""
        for field in ['large', 'medium', 'small']:
            url = getattr(self, field)
            if url:
                cleaned_url = self._clean_single_url_advanced(url)
                setattr(self, field, cleaned_url)
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def _clean_single_url_advanced(url_str: str) -> str:
        """Limpia URL con técnicas avanzadas"""
        try:
            # Parsear URL
            parsed = urlparse(url_str)
            
            # Remover parámetros no esenciales
            query_params = {}
            if parsed.query:
                for param in parsed.query.split('&'):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        # Mantener solo parámetros importantes
                        if key in ['id', 'image', 'img', 'photo', 'pic']:
                            query_params[key] = value
            
            # Reconstruir URL limpia
            clean_query = '&'.join(f"{k}={v}" for k, v in query_params.items())
            clean_url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                clean_query,
                parsed.fragment
            ))
            
            return clean_url
            
        except Exception:
            return url_str


class ProductDetails(BaseModel):
    """Modelo para detalles de productos"""
    features: List[str] = Field(default_factory=list)
    specifications: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def safe_create(cls, details_data: Optional[Dict]) -> "ProductDetails":
        """Crea instancia con extracción automática de atributos"""
        if not details_data or not isinstance(details_data, dict):
            return cls()
        
        try:
            # Procesar datos con extracción automática
            processed_data = cls._auto_extract_attributes(details_data)
            return cls(**processed_data)
            
        except Exception as e:
            logger.warning(f"Error creating ProductDetails: {e}")
            return cls()

    @classmethod
    def _auto_extract_attributes(cls, data: Dict) -> Dict:
        """Extrae atributos automáticamente"""
        processed = data.copy()
        
        # Asegurar que features sea una lista
        if 'features' in processed:
            if isinstance(processed['features'], str):
                processed['features'] = [processed['features']]
            elif not isinstance(processed['features'], list):
                processed['features'] = []
        else:
            processed['features'] = []
        
        # Asegurar que specifications sea un diccionario
        if 'specifications' not in processed or not isinstance(processed['specifications'], dict):
            processed['specifications'] = {}
        
        # Extraer campos adicionales si existen
        additional_fields = ['brand', 'model', 'color', 'weight', 'dimensions', 'material']
        for field in additional_fields:
            if field in processed and processed[field]:
                # Agregar a specifications si no existe en features
                if field not in processed['specifications']:
                    processed['specifications'][field] = processed[field]
        
        return processed

    def get_auto_dimensions(self) -> Optional[str]:
        """Extrae dimensiones automáticamente de las especificaciones"""
        dimension_patterns = [
            r'(\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?\s*(?:cm|in|inch|"))',
            r'(\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?\s*(?:cm|in|inch|"))',
        ]
        
        all_specs_text = ' '.join(str(v) for v in self.specifications.values())
        
        for pattern in dimension_patterns:
            match = re.search(pattern, all_specs_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None


# ------------------------------------------------------------------
# Main product entity simplificada
# ------------------------------------------------------------------
class Product(BaseModel):
    """Modelo principal de producto con procesamiento automático"""
    
    # Campos principales
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field("Unknown Product", min_length=1, max_length=AutoProductConfig.MAX_TITLE_LENGTH)
    main_category: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    price: Optional[float] = Field(None, ge=0)
    average_rating: Optional[float] = Field(None, ge=0, le=5)
    rating_count: Optional[int] = Field(None, ge=0)
    images: Optional[ProductImage] = None
    details: ProductDetails = Field(default_factory=ProductDetails)
    product_type: Optional[str] = None
    compatible_devices: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    attributes: Dict[str, str] = Field(default_factory=dict)
    description: Optional[str] = Field(None, max_length=AutoProductConfig.MAX_DESCRIPTION_LENGTH)
    
    # Campos calculados automáticamente
    content_hash: Optional[str] = None

    # --------------------------------------------------
    # Validators simplificados
    # --------------------------------------------------
    @model_validator(mode='before')
    @classmethod
    def auto_process_data(cls, data: Any) -> Any:
        """Procesamiento automático completo de datos"""
        if not isinstance(data, dict):
            return data
        
        processed = data.copy()
        
        # Procesamiento automático
        processed = cls._auto_enrich_data(processed)
        processed = cls._auto_clean_data(processed)
        
        return processed

    @classmethod
    def _auto_enrich_data(cls, data: Dict) -> Dict:
        """Enriquece datos automáticamente"""
        processed = data.copy()
        
        # Generar ID si no existe
        if not processed.get('id'):
            processed['id'] = str(uuid.uuid4())
        
        # Enriquecer título
        if processed.get('title'):
            processed['title'] = cls._auto_clean_title(processed['title'])
        
        # Enriquecer descripción
        processed['description'] = cls._auto_clean_description(processed.get('description'))
        
        # Procesar detalles automáticamente
        details_data = processed.get('details', {})
        processed['details'] = ProductDetails.safe_create(details_data).dict()
        
        # Procesar imágenes automáticamente
        images_data = processed.get('images', {})
        processed['images'] = ProductImage.safe_create(images_data).dict()
        
        # Generar hash de contenido
        processed['content_hash'] = cls._generate_content_hash(processed)
        
        return processed

    @classmethod
    def _auto_clean_data(cls, data: Dict) -> Dict:
        """Limpia datos automáticamente"""
        processed = data.copy()
        
        # Limpiar precio automáticamente
        if 'price' in processed:
            processed['price'] = cls._auto_parse_price(processed['price'])
        
        # Limpiar y normalizar categorías
        if 'categories' in processed:
            processed['categories'] = cls._auto_normalize_categories(processed['categories'])
        
        # Limpiar rating
        if 'average_rating' in processed:
            processed['average_rating'] = cls._auto_clean_rating(processed['average_rating'])
        
        # Limpiar rating_count
        if 'rating_count' in processed:
            processed['rating_count'] = cls._auto_clean_rating_count(processed['rating_count'])
        
        # Asegurar que tags sea una lista
        if 'tags' in processed:
            if isinstance(processed['tags'], str):
                processed['tags'] = [processed['tags']]
            elif not isinstance(processed['tags'], list):
                processed['tags'] = []
        
        # Asegurar que compatible_devices sea una lista
        if 'compatible_devices' in processed:
            if isinstance(processed['compatible_devices'], str):
                processed['compatible_devices'] = [processed['compatible_devices']]
            elif not isinstance(processed['compatible_devices'], list):
                processed['compatible_devices'] = []
        
        # Asegurar que attributes sea un diccionario
        if 'attributes' not in processed or not isinstance(processed['attributes'], dict):
            processed['attributes'] = {}
        
        return processed

    @classmethod
    def _auto_clean_title(cls, title: str) -> str:
        """Limpia título automáticamente"""
        if not title or not isinstance(title, str):
            return "Unknown Product"
        
        # Remover caracteres extraños y normalizar espacios
        cleaned = re.sub(r'[^\w\s\-\.\&\(\)]', ' ', title)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Capitalizar palabras (excepto palabras cortas)
        words = cleaned.split()
        if len(words) > 1:
            short_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            cleaned_words = []
            for i, word in enumerate(words):
                if i == 0 or word.lower() not in short_words:
                    cleaned_words.append(word.capitalize())
                else:
                    cleaned_words.append(word.lower())
            cleaned = ' '.join(cleaned_words)
        else:
            cleaned = cleaned.capitalize()
        
        # Truncar si es necesario
        if len(cleaned) > AutoProductConfig.MAX_TITLE_LENGTH:
            cleaned = cleaned[:AutoProductConfig.MAX_TITLE_LENGTH - 3] + "..."
        
        return cleaned

    @classmethod
    def _auto_clean_description(cls, description: Any) -> str:
        """Limpia descripción automáticamente"""
        if not description:
            return "No description available"
        
        if isinstance(description, list):
            description = ' '.join(str(x) for x in description if x)
        
        if not isinstance(description, str):
            description = str(description)
        
        # Limpiar y truncar
        cleaned = re.sub(r'\s+', ' ', description).strip()
        if len(cleaned) > AutoProductConfig.MAX_DESCRIPTION_LENGTH:
            cleaned = cleaned[:AutoProductConfig.MAX_DESCRIPTION_LENGTH - 3] + "..."
        
        return cleaned

    @classmethod
    def _auto_parse_price(self, price: Any) -> Optional[float]:
        """Parsea un precio desde múltiples formatos comunes."""
        
        if price is None:
            return None

        # Si ya es numérico, retornarlo como float
        if isinstance(price, (int, float)):
            return float(price)

        if isinstance(price, str):
            # Lista unificada de patrones (originales + nuevos)
            patterns = [
                r'[\$€£]?\s*(\d+(?:[.,]\d{1,2})?)',            # $123.45  | €123,45 | £ 59
                r'(\d+(?:[.,]\d{1,2})?)\s*USD',                # 123.45 USD
                r'price[:\s]*[\$€£]?\s*(\d+(?:[.,]\d{1,2})?)', # Price: $123.45
                r'(\d+(?:[.,]\d{1,2})?)\s*dollars',            # 59.99 dollars
                
                # --- Nuevos patrones ---
                r'\$(\d+(?:\.\d{2})?)',                        # $58.00
                r'(\d+(?:\.\d{2})?)\s*(?:USD|dollars)',        # 58.00 USD
                r'price[\s:]*\$?(\d+(?:\.\d{2})?)',            # Price: 58.00
            ]

            # Intentar todos los patrones
            for pattern in patterns:
                match = re.search(pattern, price, re.IGNORECASE)
                if match:
                    try:
                        price_str = match.group(1).replace(',', '.')  # Normalizar
                        return float(price_str)
                    except (ValueError, TypeError):
                        continue

        return None

    @classmethod
    def _auto_normalize_categories(cls, categories: Any) -> List[str]:
        """Normaliza categorías automáticamente"""
        if not categories:
            return []
        
        if isinstance(categories, str):
            categories = [categories]
        
        if not isinstance(categories, list):
            return []
        
        normalized = set()
        for category in categories:
            if isinstance(category, str) and category.strip():
                # Capitalización consistente
                clean_category = category.strip().title()
                normalized.add(clean_category)
        
        return sorted(normalized)

    @classmethod
    def _auto_clean_rating(cls, rating: Any) -> Optional[float]:
        """Limpia rating automáticamente"""
        if rating is None:
            return None
        
        try:
            if isinstance(rating, str):
                # Extraer número de string
                match = re.search(r'(\d+(?:\.\d+)?)', rating)
                if match:
                    rating = float(match.group(1))
                else:
                    return None
            
            rating_float = float(rating)
            if 0 <= rating_float <= 5:
                return round(rating_float, 1)
        except (ValueError, TypeError):
            pass
        
        return None

    @classmethod
    def _auto_clean_rating_count(cls, rating_count: Any) -> Optional[int]:
        """Limpia rating_count automáticamente"""
        if rating_count is None:
            return None
        
        try:
            if isinstance(rating_count, str):
                # Extraer número de string
                match = re.search(r'(\d+)', rating_count)
                if match:
                    rating_count = int(match.group(1))
                else:
                    return None
            
            count_int = int(rating_count)
            if count_int >= 0:
                return count_int
        except (ValueError, TypeError):
            pass
        
        return None

    @classmethod
    def _generate_content_hash(cls, data: Dict) -> str:
        """Genera hash del contenido para detección de duplicados"""
        content_parts = [
            data.get('title', ''),
            data.get('description', ''),
            str(data.get('price', '')),
            data.get('main_category', ''),
            ' '.join(data.get('categories', [])),
            data.get('product_type', '')
        ]
        content_str = '|'.join(content_parts)
        return hashlib.md5(content_str.encode()).hexdigest()

    # --------------------------------------------------
    # Constructors and methods
    # --------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Dict) -> "Product":
        """Constructor automatizado desde diccionario"""
        try:
            # Usar el procesamiento automático del model_validator
            return cls(**raw)
            
        except Exception as e:
            logger.warning(f"Error creating Product from dict: {e}")
            # Crear producto mínimo con valores por defecto
            return cls(
                title=raw.get('title', 'Unknown Product'),
                id=raw.get('id', str(uuid.uuid4()))
            )

    def clean_image_urls(self) -> None:
        """Limpia URLs de imágenes automáticamente"""
        if self.images:
            self.images.clean_urls_automated()
        else:
            # Crear imagen por defecto si no existe
            self.images = ProductImage(
                large="https://via.placeholder.com/500",
                medium="https://via.placeholder.com/300", 
                small="https://via.placeholder.com/150"
            )

    def to_text(self) -> str:
        """Representación optimizada para embedding"""
        parts = [
            self.title,
            f"Category: {self.main_category or 'Uncategorized'}",
            self.description or "No description available",
            f"Price: {self._format_price()}",
            f"Rating: {self._format_rating()}",
            f"Type: {self.product_type or 'No type specified'}",
            f"Tags: {self._format_tags()}"
        ]
        
        # Agregar características si existen
        if self.details and self.details.features:
            features_str = ", ".join(self.details.features[:5])
            parts.append(f"Features: {features_str}")
        
        return " | ".join(filter(None, parts))

    def _format_price(self) -> str:
        if isinstance(self.price, (int, float)) and self.price > 0:
            return f"${self.price:.2f}"
        return "Price not available"

    def _format_rating(self) -> str:
        if isinstance(self.average_rating, (int, float)) and self.average_rating > 0:
            count_str = f"({self.rating_count} reviews)" if self.rating_count else ""
            return f"{self.average_rating:.1f}/5 {count_str}".strip()
        return "No rating available"

    def _format_tags(self) -> str:
        return ", ".join(self.tags[:8]) if self.tags else "No tags"

    def to_metadata(self) -> dict:
        """Metadata enriquecida con información automática"""
        try:
            metadata = {
                "id": self.id,
                "title": self.title[:100],
                "main_category": self.main_category or "Uncategorized",
                "categories": json.dumps(self.categories, ensure_ascii=False) if self.categories else "[]",
                "price": float(self.price) if self.price is not None else AutoProductConfig.DEFAULT_PRICE,
                "average_rating": float(self.average_rating) if self.average_rating else AutoProductConfig.DEFAULT_RATING,
                "rating_count": self.rating_count or 0,
                "description": (self.description or "")[:200],
                "product_type": self.product_type or "",
                "content_hash": self.content_hash or "",
                "features": json.dumps(self.details.features[:10], ensure_ascii=False) if self.details.features else "[]",
                "tags": json.dumps(self.tags[:5], ensure_ascii=False) if self.tags else "[]",
                "compatible_devices": json.dumps(self.compatible_devices[:5], ensure_ascii=False) if self.compatible_devices else "[]"
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error converting to metadata: {e}")
            return self._get_fallback_metadata()

    def _get_fallback_metadata(self) -> dict:
        """Metadata de fallback en caso de error"""
        return {
            "id": self.id,
            "title": self.title[:100] if self.title else "Untitled Product",
            "main_category": "Uncategorized",
            "categories": "[]",
            "price": AutoProductConfig.DEFAULT_PRICE,
            "average_rating": AutoProductConfig.DEFAULT_RATING,
            "rating_count": 0,
            "description": "",
            "product_type": "",
            "content_hash": self.content_hash or "",
            "features": "[]",
            "tags": "[]",
            "compatible_devices": "[]"
        }

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """Incluye procesamiento automático en el dump"""
        self.clean_image_urls()
        return super().model_dump(*args, **kwargs)

    def get_summary(self) -> Dict[str, Any]:
        """Resumen del producto para visualización"""
        return {
            "id": self.id,
            "title": self.title,
            "price": self.price,
            "category": self.main_category,
            "rating": self.average_rating,
            "type": self.product_type,
            "has_description": bool(self.description and self.description != "No description available"),
            "has_images": bool(self.images and (self.images.large or self.images.medium or self.images.small)),
            "feature_count": len(self.details.features) if self.details else 0,
            "tag_count": len(self.tags)
        }

    def __str__(self) -> str:
        return f"Product(title='{self.title}', price={self.price}, category='{self.main_category}')"

    def __repr__(self) -> str:
        return f"Product(id='{self.id}', title='{self.title}')"


# Aliases para compatibilidad
ProductImage = ProductImage
ProductDetails = ProductDetails