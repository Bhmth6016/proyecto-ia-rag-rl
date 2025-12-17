from __future__ import annotations
# src/core/data/product.py
import hashlib
import re
import json
import uuid
import time
import threading
from typing import Optional, Dict, List, Any, ClassVar, Union
from urllib.parse import urlparse, urlunparse
from functools import lru_cache
import logging

from pydantic import BaseModel, Field, model_validator
import numpy as np

from src.core.config import settings, get_settings

logger = logging.getLogger(__name__)

class ProductImage(BaseModel):
    large: Optional[str] = None
    medium: Optional[str] = None
    small: Optional[str] = None
    
    _validated_urls: ClassVar[Dict[str, bool]] = {}
    
    @classmethod
    def safe_create(cls, image_data: Optional[Dict]) -> "ProductImage":
        if not image_data or not isinstance(image_data, dict):
            return cls()
        
        validated_data = {}
        for size, url in image_data.items():
            if url and cls._validate_url_automated(url):
                validated_data[size] = url
        
        return cls(**validated_data)
    
    @classmethod
    def _validate_url_automated(cls, url: str) -> bool:
        if url in cls._validated_urls:
            return cls._validated_urls[url]
        
        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                cls._validated_urls[url] = False
                return False
            
            valid_domains = {
                'images.amazon.com', 'm.media-amazon.com', 'example.com',
                'cdn.shopify.com', 'i.ebayimg.com', 'target.scene7.com',
                'via.placeholder.com', 'localhost', '127.0.0.1'
            }
            
            if parsed.netloc in valid_domains:
                cls._validated_urls[url] = True
                return True
            
            cls._validated_urls[url] = True
            return True
            
        except Exception:
            cls._validated_urls[url] = False
            return False
    
    def clean_urls_automated(self) -> None:
        for field in ['large', 'medium', 'small']:
            url = getattr(self, field)
            if url:
                cleaned_url = self._clean_single_url_advanced(url)
                setattr(self, field, cleaned_url)
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def _clean_single_url_advanced(url_str: str) -> str:
        try:
            parsed = urlparse(url_str)
            
            query_params = {}
            if parsed.query:
                for param in parsed.query.split('&'):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        if key in ['id', 'image', 'img', 'photo', 'pic']:
                            query_params[key] = value
            
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
    features: List[str] = Field(default_factory=list)
    specifications: Dict[str, Any] = Field(default_factory=dict)
    nlp_processed: bool = Field(default=False, description="Indica si el producto fue procesado por NLP")
    has_ner: bool = Field(default=False, description="Tiene entidades NER extraídas")
    has_zero_shot: bool = Field(default=False, description="Tiene clasificación Zero-Shot")
    ner_entities: Optional[Dict[str, Any]] = Field(default=None, description="Entidades NER extraídas")
    zero_shot_classification: Optional[Dict[str, float]] = Field(default=None, description="Clasificación Zero-Shot")
    
    @classmethod
    def safe_create(cls, details_data: Optional[Dict]) -> "ProductDetails":
        if not details_data or not isinstance(details_data, dict):
            return cls()
        
        try:
            processed_data = cls._auto_extract_attributes(details_data)
            return cls(**processed_data)
            
        except Exception as e:
            logger.warning(f"Error creating ProductDetails: {e}")
            return cls()

    @classmethod
    def _auto_extract_attributes(cls, data: Dict) -> Dict:
        processed = data.copy()
        
        if 'features' in processed:
            if isinstance(processed['features'], str):
                processed['features'] = [processed['features']]
            elif not isinstance(processed['features'], list):
                processed['features'] = []
        else:
            processed['features'] = []
        
        if 'specifications' not in processed or not isinstance(processed['specifications'], dict):
            processed['specifications'] = {}
        
        additional_fields = ['brand', 'model', 'color', 'weight', 'dimensions', 'material']
        for field in additional_fields:
            if field in processed and processed[field]:
                if field not in processed['specifications']:
                    processed['specifications'][field] = processed[field]
        
        return processed

    def get_auto_dimensions(self) -> Optional[str]:
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


class MLProductProcessor:
    
    _embedding_model = None
    _model_lock = threading.Lock()
    
    @classmethod
    def _get_embedding_model(cls, model_name: Optional[str] = None):
        # ✅ CORRECCIÓN: Importar dentro del método para evitar problemas
        if model_name is None:
            from src.core.config import get_settings
            settings = get_settings()
            model_name = settings.ML_EMBEDDING_MODEL
        
        if cls._embedding_model is None:
            with cls._model_lock:
                if cls._embedding_model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                        logger.info(f"🔧 Cargando modelo de embeddings para ML: {model_name}")
                        cls._embedding_model = SentenceTransformer(model_name)
                        logger.info(f"✅ Modelo de embeddings cargado")
                    except ImportError:
                        logger.warning("⚠️ SentenceTransformer no disponible, embeddings no funcionarán")
                        cls._embedding_model = None
                    except Exception as e:
                        logger.error(f"❌ Error cargando modelo: {e}")
                        cls._embedding_model = None
        
        return cls._embedding_model
    
    @classmethod
    def enrich_product(cls, data: Dict) -> Dict:
        if not data:
            return data
        
        # ✅ CORRECCIÓN: Importar dentro del método
        from src.core.config import get_settings
        settings = get_settings()
        
        if not settings.ML_ENABLED:
            data['ml_processed'] = False
            return data
        
        try:
            enriched = data.copy()
            text = f"{enriched.get('title', '')} {enriched.get('description', '')}".strip()
            
            # Solo generar embedding si la característica está habilitada
            if 'embedding' in settings.ML_FEATURES and text:
                model = cls._get_embedding_model()
                if model:
                    embedding = model.encode(text[:1000], normalize_embeddings=True)
                    
                    # ✅ CORRECCIÓN: Manejar diferentes tipos de embedding (Tensor, numpy array, lista)
                    enriched['embedding'] = cls._convert_embedding_to_list(embedding)
                    enriched['embedding_model'] = settings.ML_EMBEDDING_MODEL
            
            # ✅ CORRECCIÓN: Usar _predict_category (no _predict_category_general)
            if 'category' in settings.ML_FEATURES and text:
                category = cls._predict_category(text, settings.ML_CATEGORIES)
                if category:
                    enriched['predicted_category'] = category
                    enriched.setdefault('main_category', category)

            # ✅ CORRECCIÓN: Usar _extract_entities (no _extract_entities_general)
            if 'entities' in settings.ML_FEATURES and text:
                entities = cls._extract_entities(text)
                if entities:
                    enriched['extracted_entities'] = entities
            
            # ✅ CORRECCIÓN: Restaurar generación de tags si está habilitado
            if 'tags' in settings.ML_FEATURES and text:
                tags = cls._generate_tags(text)
                if tags:
                    enriched['ml_tags'] = tags

            enriched['ml_processed'] = True
            return enriched
        
        except Exception as e:
            logger.warning(f"ML processing failed: {e}")
            data['ml_processed'] = False
            return data
    
    @staticmethod
    def _convert_embedding_to_list(embedding) -> List[float]:
        """Convierte embeddings de cualquier tipo a lista de Python."""
        try:
            # Si es un Tensor de PyTorch
            if hasattr(embedding, 'cpu'):
                embedding = embedding.cpu().numpy()
            elif hasattr(embedding, 'detach'):
                embedding = embedding.detach().cpu().numpy()
            
            # Si es un numpy array
            if isinstance(embedding, np.ndarray):
                return embedding.tolist()
            # Si ya es una lista
            elif isinstance(embedding, list):
                return embedding
            # Si es iterable pero no lista
            elif hasattr(embedding, '__iter__'):
                return list(embedding)
            # Último recurso
            else:
                return []
        except Exception as e:
            logger.warning(f"Error convirtiendo embedding a lista: {e}")
            return []
    
    @staticmethod
    def _predict_category(text: str, categories: List[str]) -> Optional[str]:
        """Predicción de categorías para e-commerce general - FUSIÓN MEJORADA"""
        if not text:
            return None
        
        text_lower = text.lower()

        # ✅ FUSIÓN: Tomar diccionario generalizado de NUEVA versión
        keyword_expansion = {
            'Electronics': [
                'laptop', 'computer', 'pc', 'macbook', 'notebook', 'desktop',
                'tablet', 'smartphone', 'phone', 'mobile', 'monitor', 'keyboard',
                'mouse', 'printer', 'scanner', 'camera', 'headphones', 'earphones',
                'speaker', 'tv', 'television', 'electronic', 'device', 'gadget',
                'usb', 'hdmi', 'cable', 'charger', 'battery', 'router', 'modem',
                'smartwatch', 'fitness tracker', 'drone', 'projector'
            ],
            'Clothing': [
                'shirt', 't-shirt', 'pants', 'jeans', 'dress', 'jacket', 'hoodie',
                'sweater', 'sweatshirt', 'shorts', 'skirt', 'blouse', 'coat',
                'underwear', 'socks', 'shoes', 'sneakers', 'boots', 'sandals',
                'hat', 'cap', 'gloves', 'scarf', 'belt', 'tie', 'suit', 'uniform'
            ],
            'Home & Kitchen': [
                'kitchen', 'cookware', 'appliance', 'furniture', 'sofa', 'bed',
                'chair', 'table', 'desk', 'lamp', 'light', 'rug', 'carpet',
                'curtain', 'blanket', 'pillow', 'mattress', 'cabinet', 'shelf',
                'refrigerator', 'oven', 'microwave', 'blender', 'toaster', 'dishwasher',
                'vacuum', 'mop', 'broom', 'detergent', 'cleaner'
            ],
            'Books': [
                'book', 'novel', 'author', 'edition', 'hardcover', 'paperback',
                'kindle', 'ebook', 'textbook', 'magazine', 'comic', 'biography',
                'fiction', 'non-fiction', 'science', 'history', 'cookbook', 'manual'
            ],
            'Sports & Outdoors': [
                'fitness', 'exercise', 'gym', 'yoga', 'outdoor', 'camping',
                'hiking', 'running', 'training', 'bike', 'bicycle', 'ball',
                'soccer', 'basketball', 'tennis', 'golf', 'swimming', 'fishing',
                'tent', 'sleeping bag', 'backpack', 'hiking boots', 'kayak', 'paddle'
            ],
            'Beauty': [
                'makeup', 'cosmetic', 'skincare', 'perfume', 'serum', 'lotion',
                'shampoo', 'conditioner', 'hair', 'nail', 'lipstick', 'mascara',
                'brush', 'mirror', 'cream', 'oil', 'soap', 'deodorant',
                'razor', 'shaver', 'trimmer', 'epilator', 'massager'
            ],
            'Toys & Games': [
                'toy', 'lego', 'puzzle', 'doll', 'kids', 'children', 'toddler',
                'action figure', 'board game', 'video game', 'game', 'console',
                'playstation', 'xbox', 'nintendo', 'switch', 'ps5', 'controller',
                'card game', 'dice', 'chess', 'puzzle', 'blocks'
            ],
            'Automotive': [
                'car', 'auto', 'vehicle', 'engine', 'tire', 'motor', 'battery',
                'oil', 'filter', 'brake', 'light', 'tool', 'accessory', 'parts',
                'wiper', 'mirror', 'seat cover', 'steering wheel', 'antenna'
            ],
            'Office Products': [
                'office', 'stationery', 'paper', 'pen', 'pencil', 'notebook',
                'printer', 'scanner', 'desk', 'chair', 'lamp', 'folder', 'binder',
                'stapler', 'scissors', 'tape', 'envelope', 'clipboard', 'calendar'
            ],
            'Health': [
                'vitamin', 'supplement', 'medicine', 'first aid', 'thermometer',
                'bandage', 'mask', 'sanitizer', 'pill', 'tablet', 'syrup',
                'blood pressure', 'glucometer', 'inhaler', 'wheelchair', 'crutch'
            ],
            'Video Games': [
                'video game', 'game', 'gaming', 'play', 'player', 'console', 'controller',
                'nintendo', 'playstation', 'xbox', 'switch', 'ps4', 'ps5',
                'retro', 'arcade', 'emulator', 'rom', 'cartridge', 'disc',
                'steam', 'epic games', 'controller', 'joystick', 'headset'
            ]
        }

        found_categories = {}

        for category, keywords in keyword_expansion.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_categories[category] = found_categories.get(category, 0) + 1

        if found_categories:
            best_category = max(found_categories.items(), key=lambda x: x[1])[0]
            return best_category
        
        return None

    @classmethod
    def get_embedding_model_singleton(cls, model_name: Optional[str] = None):
        if not hasattr(cls, '_global_embedding_model'):
            cls._global_embedding_model = None
        
        if cls._global_embedding_model is None:
            with cls._model_lock:
                if cls._global_embedding_model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                        from src.core.config import get_settings  # ✅ Importación local
                        settings = get_settings()
                        model_name = model_name or settings.ML_EMBEDDING_MODEL
                        logger.info(f"🔧 [SINGLETON] Cargando modelo de embeddings: {model_name}")
                        cls._global_embedding_model = SentenceTransformer(model_name)
                        logger.info("✅ [SINGLETON] Modelo cargado")
                    except Exception as e:
                        logger.error(f"❌ [SINGLETON] Error cargando modelo: {e}")
                        cls._global_embedding_model = None
        
        return cls._global_embedding_model

    @classmethod
    def enrich_product_with_embeddings(cls, data: Dict, force_recompute: bool = False) -> Dict:
        if not data:
            return data
        
        from src.core.config import get_settings
        settings = get_settings()
        
        if not settings.ML_ENABLED or 'embedding' not in settings.ML_FEATURES:
            data['ml_processed'] = False
            return data
        
        try:
            enriched = data.copy()
            text = f"{enriched.get('title', '')} {enriched.get('description', '')}".strip()
            
            if not text:
                return enriched
            
            existing_embedding = enriched.get('embedding')
            if (not force_recompute and existing_embedding and 
                isinstance(existing_embedding, list) and 
                len(existing_embedding) > 10):  
                logger.debug("✅ Usando embedding existente")
                enriched['embedding_model'] = enriched.get('embedding_model', 'existing')
                return enriched
            
            model = cls.get_embedding_model_singleton()
            if model:
                text_limited = text[:1000]
                embedding = model.encode(text_limited, normalize_embeddings=True)
                
                # ✅ CORRECCIÓN: Usar el método helper para convertir embedding
                enriched['embedding'] = cls._convert_embedding_to_list(embedding)
                enriched['embedding_model'] = settings.ML_EMBEDDING_MODEL
                enriched['embedding_timestamp'] = time.time()
                
                logger.debug(f"🔧 Embedding generado: {len(enriched['embedding'])} dimensiones")
            
            return enriched
            
        except Exception as e:
            logger.warning(f"ML embedding failed: {e}")
            return data
    
    @staticmethod
    def _extract_entities(text: str) -> Dict[str, List[str]]:
        """Extracción de entidades para e-commerce - FUSIÓN MEJORADA"""
        # ✅ FUSIÓN: Mantener estructura ORIGINAL para compatibilidad
        entities = {
            'ORG': [],      # Marcas y organizaciones
            'PRODUCT': [],  # Nombres de productos
            'COLOR': [],    # Colores (agregado de NUEVA versión)
            'SIZE': []      # Tamaños (agregado de NUEVA versión)
        }
        
        # ✅ FUSIÓN: Combinar ambos enfoques
        
        # 1. Detección de nombres propios (de ORIGINAL)
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        for word in words:
            if len(word) > 3 and word.lower() not in ['The', 'And', 'For', 'With', 'This']:
                entities['PRODUCT'].append(word)
        
        # 2. Detección de marcas específicas (de NUEVA versión)
        text_lower = text.lower()
        brands = ['apple', 'samsung', 'sony', 'lg', 'nike', 'adidas', 'dell', 'hp', 'lenovo']
        for brand in brands:
            if brand in text_lower:
                entities['ORG'].append(brand.title())
        
        # 3. Detección de colores (de NUEVA versión)
        colors = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'pink', 'purple', 'gray']
        for color in colors:
            if color in text_lower:
                entities['COLOR'].append(color.title())
        
        # 4. Detección de tamaños (de NUEVA versión)
        sizes = ['small', 'medium', 'large', 'xl', 'xxl', 'xs', 's', 'm', 'l']
        for size in sizes:
            if f' {size} ' in f' {text_lower} ' or text_lower.endswith(f' {size}'):
                entities['SIZE'].append(size.upper())
        
        # Filtrar diccionarios vacíos
        return {k: v for k, v in entities.items() if v}
    
    @staticmethod
    def _generate_tags(text: str) -> List[str]:
        """Generación de tags - MANTENER de ORIGINAL"""
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        
        # Palabras de stop
        stop_words = {
            'this', 'that', 'with', 'from', 'have', 'they', 'what',
            'were', 'when', 'will', 'your', 'there', 'their', 'about'
        }
        
        # Contar frecuencia
        word_counts = {}
        for word in words:
            if word not in stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Top 5 palabras más frecuentes
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:5]]


class ProductAttributeHelper:
    """Helper para manejar atributos de productos de forma segura."""
    
    @staticmethod
    def safe_get_float(obj, attr_name: str, default: float = 0.0) -> float:
        """Obtiene un atributo float de forma segura."""
        try:
            value = getattr(obj, attr_name, default)
            if value is None:
                return default
            return float(value)
        except (ValueError, TypeError, AttributeError):
            return default
    
    @staticmethod
    def safe_get_int(obj, attr_name: str, default: int = 0) -> int:
        """Obtiene un atributo int de forma segura."""
        try:
            value = getattr(obj, attr_name, default)
            if value is None:
                return default
            return int(value)
        except (ValueError, TypeError, AttributeError):
            return default
    
    @staticmethod
    def safe_get_str(obj, attr_name: str, default: str = "") -> str:
        """Obtiene un atributo string de forma segura."""
        try:
            value = getattr(obj, attr_name, default)
            if value is None:
                return default
            return str(value)
        except (AttributeError):
            return default


# ------------------------------------------------------------------
# Main Product entity - MEJORADA PARA E-COMMERCE GENERAL
# ------------------------------------------------------------------
class Product(BaseModel):
    """
    Modelo principal de producto para e-commerce general.
    """
    
    # 🔥 CONSTANTES LOCALES (no configuración global)
    _MAX_TITLE_LENGTH: ClassVar[int] = 200
    _MAX_DESCRIPTION_LENGTH: ClassVar[int] = 1000
    _DEFAULT_RATING: ClassVar[float] = 0.0
    _DEFAULT_PRICE: ClassVar[float] = 0.0
    
    # Campos principales
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field("Unknown Product", min_length=1, max_length=_MAX_TITLE_LENGTH)
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
    description: Optional[str] = Field(None, max_length=_MAX_DESCRIPTION_LENGTH)
    
    # Campos calculados automáticamente
    content_hash: Optional[str] = None
    
    # Campos ML (opcionales)
    predicted_category: Optional[str] = None
    extracted_entities: Dict[str, List[str]] = Field(default_factory=dict)
    ml_tags: List[str] = Field(default_factory=list)
    ml_processed: bool = Field(default=False)
    
    # Campos embeddings
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    
    # Campos NLP (para compatibilidad)
    nlp_processed: bool = Field(default=False)
    has_ner: bool = Field(default=False)
    has_zero_shot: bool = Field(default=False)
    ner_entities: Optional[Dict[str, Any]] = Field(default=None)
    zero_shot_classification: Optional[Dict[str, float]] = Field(default=None)
    
    # ✅ CORRECCIÓN: Configurar valores por defecto para campos opcionales
    class Config:
        arbitrary_types_allowed = True
    
    # --------------------------------------------------
    # Validators simplificados
    # --------------------------------------------------
    @model_validator(mode='before')
    @classmethod
    def auto_process_data(cls, data: Any) -> Any:
        """Procesamiento automático completo de datos"""
        if not isinstance(data, dict):
            return data
        
        # Obtener configuración actual
        current_settings = get_settings()
        
        # 🔥 CORRECCIÓN PROBLEMA 4: Limpiar datos primero
        processed = cls._clean_raw_data(data)
        
        # Procesamiento automático base
        processed = cls._auto_enrich_data(processed)
        processed = cls._auto_clean_data(processed)
        
        # 🔥 IMPORTANTE: Procesamiento ML si está habilitado en settings
        if current_settings.ML_ENABLED:
            processed = MLProductProcessor.enrich_product(processed)
        
        return processed

    @classmethod
    def _clean_raw_data(cls, raw: Dict) -> Dict:
        """Limpia datos crudos de forma robusta"""
        cleaned = {}
        
        # 🔥 Mapeo de campos con limpieza específica
        field_cleaners = {
            'title': lambda x: str(x).strip()[:200] if x else 'Unknown Product',
            'description': lambda x: str(x).strip()[:1000] if x else '',
            'main_category': lambda x: str(x).strip() if x else 'General',
            'categories': cls._clean_categories,
            'price': cls._clean_price,
            'brand': lambda x: str(x).strip() if x else '',
            'product_type': lambda x: str(x).strip() if x else '',
            'average_rating': lambda x: float(x) if x is not None else None,
            'rating_count': lambda x: int(x) if x is not None else None
        }
        
        for field, cleaner in field_cleaners.items():
            if field in raw:
                try:
                    cleaned[field] = cleaner(raw[field])
                except Exception as e:
                    logger.debug(f"Error limpiando campo {field}: {e}")
                    cleaned[field] = field_cleaners[field](None)  # Valor por defecto
        
        # Campos adicionales con limpieza segura
        cleaned['id'] = raw.get('id', str(uuid.uuid4()))
        
        # Copiar otros campos sin procesar (serán procesados después)
        for key, value in raw.items():
            if key not in cleaned:
                cleaned[key] = value
        
        return cleaned
    
    @classmethod
    def _clean_categories(cls, categories: Any) -> List[str]:
        """Limpia categorías"""
        if not categories:
            return []
        
        if isinstance(categories, str):
            # Separar por comas, puntos, etc.
            split_categories = re.split(r'[,;|]', categories)
            return [c.strip() for c in split_categories if c.strip()]
        
        if isinstance(categories, list):
            cleaned = []
            for cat in categories:
                if cat and isinstance(cat, str):
                    cleaned.append(cat.strip())
            return cleaned
        
        return []
    
    @classmethod
    def _clean_price(cls, price: Any) -> Optional[float]:
        """Limpia precio de forma robusta"""
        return cls._auto_parse_price(price)

    @classmethod
    def _auto_enrich_data(cls, data: Dict) -> Dict:
        """Enriquece datos automáticamente con mejora de categorización."""
        processed = data.copy()
        
        # ============================
        # ✔ Código base
        # ============================
        if not processed.get('id'):
            processed['id'] = str(uuid.uuid4())
        
        if processed.get('title'):
            processed['title'] = cls._auto_clean_title(processed['title'])
            
        processed['description'] = cls._auto_clean_description(processed.get('description'))
        
        details_data = processed.get('details', {})
        processed['details'] = ProductDetails.safe_create(details_data).dict()

        images_data = processed.get('images', {})
        processed['images'] = ProductImage.safe_create(images_data).dict()

        processed['content_hash'] = cls._generate_content_hash(processed)

        # =======================================================
        # 🔥 A) NUEVO — Mejorar categorización automática
        # =======================================================
        if not processed.get("main_category") or processed.get("main_category") in [None, "", "General"]:

            title = processed.get("title", "")
            category = cls._extract_category_from_title(title)
            if category:
                processed["main_category"] = category

            # Si no encontró en el título, probar en descripción
            if not processed.get("main_category") or processed.get("main_category") == "General":
                description = processed.get("description", "")
                category = cls._extract_category_from_description(description)
                if category:
                    processed["main_category"] = category
        
        # Si aún queda como General → se manejará después en la etapa ML
        return processed
    
    @staticmethod
    def _extract_category_from_title(title: str) -> Optional[str]:
        """Extrae categoría del título usando palabras clave generales para e-commerce - MEJORADO"""
        if not title:
            return None
        
        title_lower = title.lower()
        
        # 🔥 DICCIONARIO MEJORADO CON SINÓNIMOS EN ESPAÑOL
        category_keywords = {
            'Electronics': [
                'laptop', 'computer', 'pc', 'macbook', 'notebook', 'desktop',
                'tablet', 'smartphone', 'phone', 'mobile', 'monitor', 'keyboard',
                'mouse', 'printer', 'scanner', 'camera', 'headphones', 'earphones',
                'speaker', 'tv', 'television', 'electronic', 'device', 'gadget',
                'usb', 'hdmi', 'cable', 'charger', 'battery', 'router', 'modem',
                'smartwatch', 'fitness tracker', 'drone', 'projector', 'auricular',
                'audífono', 'teclado', 'ratón', 'impresora', 'escáner'
            ],
            'Clothing': [
                'shirt', 't-shirt', 'pants', 'jeans', 'dress', 'jacket', 'hoodie',
                'sweater', 'sweatshirt', 'shorts', 'skirt', 'blouse', 'coat',
                'underwear', 'socks', 'shoes', 'sneakers', 'boots', 'sandals',
                'hat', 'cap', 'gloves', 'scarf', 'belt', 'tie', 'suit', 'uniform',
                'camisa', 'pantalón', 'vestido', 'chaqueta', 'sudader', 'falda',
                'blusa', 'abrigo', 'calcetines', 'zapatos', 'zapatillas', 'botas',
                'sombrero', 'gorra', 'guantes', 'bufanda', 'cinturón', 'corbata'
            ],
            'Home & Kitchen': [
                'kitchen', 'cookware', 'appliance', 'furniture', 'sofa', 'bed',
                'chair', 'table', 'desk', 'lamp', 'light', 'rug', 'carpet',
                'curtain', 'blanket', 'pillow', 'mattress', 'cabinet', 'shelf',
                'refrigerator', 'oven', 'microwave', 'blender', 'toaster',
                'dishwasher', 'vacuum', 'mop', 'broom', 'detergent', 'cleaner',
                'cocina', 'mueble', 'sofá', 'cama', 'silla', 'mesa', 'escritorio',
                'lámpara', 'alfombra', 'cortina', 'manta', 'almohada', 'colchón',
                'armario', 'estante', 'nevera', 'horno', 'microondas', 'batidora',
                'tostadora', 'lavavajillas', 'aspiradora', 'fregona', 'escoba'
            ],
            'Books': [
                'book', 'novel', 'author', 'edition', 'hardcover', 'paperback',
                'kindle', 'ebook', 'textbook', 'magazine', 'comic', 'biography',
                'fiction', 'non-fiction', 'science', 'history', 'cookbook', 'manual',
                'libro', 'novela', 'autor', 'edición', 'tapa dura', 'tapa blanda',
                'revista', 'cómic', 'biografía', 'ficción', 'no ficción', 'ciencia',
                'historia', 'libro de cocina', 'manual'
            ],
            'Sports & Outdoors': [
                'fitness', 'exercise', 'gym', 'yoga', 'outdoor', 'camping',
                'hiking', 'running', 'training', 'bike', 'bicycle', 'ball',
                'soccer', 'basketball', 'tennis', 'golf', 'swimming', 'fishing',
                'tent', 'sleeping bag', 'backpack', 'hiking boots', 'kayak',
                'deporte', 'ejercicio', 'gimnasio', 'camping', 'senderismo',
                'correr', 'entrenamiento', 'bicicleta', 'pelota', 'fútbol',
                'baloncesto', 'tenis', 'golf', 'natación', 'pesca', 'tienda',
                'saco de dormir', 'mochila', 'botas de senderismo'
            ],
            'Beauty': [
                'makeup', 'cosmetic', 'skincare', 'perfume', 'serum', 'lotion',
                'shampoo', 'conditioner', 'hair', 'nail', 'lipstick', 'mascara',
                'brush', 'mirror', 'cream', 'oil', 'soap', 'deodorant',
                'razor', 'shaver', 'trimmer', 'epilator', 'massager',
                'maquillaje', 'cosmético', 'cuidado de la piel', 'perfume',
                'champú', 'acondicionador', 'pelo', 'uña', 'labial', 'rimel',
                'cepillo', 'espejo', 'crema', 'aceite', 'jabón', 'desodorante',
                'maquinilla', 'afeitadora', 'recortadora'
            ],
            'Toys & Games': [
                'toy', 'lego', 'puzzle', 'doll', 'kids', 'children', 'toddler',
                'action figure', 'board game', 'video game', 'game', 'console',
                'playstation', 'xbox', 'nintendo', 'switch', 'ps5', 'controller',
                'card game', 'dice', 'chess', 'puzzle', 'blocks',
                'juguete', 'rompecabezas', 'muñeca', 'niños', 'figura de acción',
                'juego de mesa', 'videojuego', 'consola', 'mando', 'cartas',
                'dados', 'ajedrez', 'bloques'
            ],
            'Automotive': [
                'car', 'auto', 'vehicle', 'engine', 'tire', 'motor', 'battery',
                'oil', 'filter', 'brake', 'light', 'tool', 'accessory', 'parts',
                'wiper', 'mirror', 'seat cover', 'steering wheel', 'antenna',
                'coche', 'automóvil', 'vehículo', 'motor', 'neumático', 'batería',
                'aceite', 'filtro', 'freno', 'luz', 'herramienta', 'accesorio',
                'pieza', 'limpiaparabrisas', 'espejo', 'funda de asiento', 'volante'
            ],
            'Office Products': [
                'office', 'stationery', 'paper', 'pen', 'pencil', 'notebook',
                'printer', 'scanner', 'desk', 'chair', 'lamp', 'folder', 'binder',
                'stapler', 'scissors', 'tapes', 'envelope', 'clipboard', 'calendar',
                'oficina', 'papelería', 'papel', 'bolígrafo', 'lápiz', 'cuaderno',
                'impresora', 'escáner', 'escritorio', 'silla', 'carpeta', 'archivador',
                'grapadora', 'tijeras', 'cinta', 'sobre', 'portapapeles', 'calendario'
            ],
            'Health': [
                'vitamin', 'supplement', 'medicine', 'first aid', 'thermometer',
                'bandage', 'mask', 'sanitizer', 'pill', 'tablet', 'syrup',
                'blood pressure', 'glucometer', 'inhaler', 'wheelchair', 'crutch',
                'vitamina', 'suplemento', 'medicina', 'primeros auxilios', 'termómetro',
                'vendaje', 'mascarilla', 'desinfectante', 'píldora', 'comprimido', 'jarabe',
                'presión arterial', 'glucómetro', 'inhalador', 'silla de ruedas', 'muleta'
            ],
            'Video Games': [
                'video game', 'game', 'gaming', 'play', 'player', 'console', 'controller',
                'nintendo', 'playstation', 'xbox', 'switch', 'ps4', 'ps5',
                'retro', 'arcade', 'emulator', 'rom', 'cartridge', 'disc',
                'steam', 'epic games', 'controller', 'joystick', 'headset',
                'videojuego', 'juego', 'consola', 'mando', 'mandos', 'retro',
                'arcade', 'emulador', 'cartucho', 'disco', 'auriculares'
            ]
        }
        
        # Contador de coincidencias
        category_scores = {}
        
        for category, keywords in category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in title_lower:
                    score += 1
            
            if score > 0:
                category_scores[category] = score
        
        # Si hay categorías con puntuación, devolver la mejor
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])[0]
            
            # Evitar que videojuegos capture todo (requerir más evidencia)
            if best_category == 'Video Games' and category_scores['Video Games'] < 2:
                # Buscar segunda mejor categoría
                category_scores.pop('Video Games', None)
                if category_scores:
                    best_category = max(category_scores.items(), key=lambda x: x[1])[0]
            
            return best_category
        
        return None
    
    @staticmethod
    def _extract_category_from_description(description: str) -> Optional[str]:
        if not description:
            return None
        
        desc_lower = description.lower()
        category_keywords = {
            'Video Games': ['nintendo','playstation','xbox','switch','ps5','videogame','console'],
            'Electronics': ['iphone','samsung','android','tablet','laptop','pc','macbook'],
            'Books': ['book','novel','author','paperback','kindle','fiction'],
            'Clothing': ['shirt','jeans','dress','hoodie','apparel'],
            'Home & Kitchen': ['kitchen','cookware','appliance','furniture'],
            'Sports & Outdoors': ['fitness','gym','camping','running','training'],
            'Beauty': ['makeup','cosmetic','skincare','serum','hair'],
            'Toys & Games': ['toy','lego','board game','kids','children'],
            'Automotive': ['car','vehicle','engine','battery'],
            'Office Products': ['office','stationery','desk','supplies'],
            'Health': ['vitamin','supplement','medicine','first aid','thermometer']
        }
    
        scores = {}
        for category, words in category_keywords.items():
            score = sum(1 for kw in words if kw in desc_lower)
            if score > 0:
                scores[category] = score

        if scores:
            best_category = max(scores.items(), key=lambda x: x[1])[0]
            return best_category
        
        return None

    @classmethod
    def _auto_clean_data(cls, data: Dict) -> Dict:
        """Limpia datos automáticamente"""
        processed = data.copy()
        
        # Limpiar precio automáticamente
        if 'price' in processed and processed['price'] is not None:
            try:
                processed['price'] = float(processed['price'])
            except (ValueError, TypeError):
                processed['price'] = None
        
        # Limpiar y normalizar categorías
        if 'categories' in processed:
            processed['categories'] = cls._auto_normalize_categories(processed['categories'])
        
        # Limpiar rating
        if 'average_rating' in processed:
            processed['average_rating'] = cls._auto_clean_rating(processed['average_rating'])
        
        # Limpiar rating_count
        if 'rating_count' in processed:
            processed['rating_count'] = cls._auto_clean_rating_count(processed['rating_count'])
        
        # 🔥 Asegurar que todos los campos de lista estén inicializados correctamente
        list_fields = ['tags', 'compatible_devices', 'ml_tags']
        for field in list_fields:
            if field in processed:
                if isinstance(processed[field], str):
                    processed[field] = [processed[field]]
                elif not isinstance(processed[field], list):
                    processed[field] = []
                # Filtrar valores None de las listas
                processed[field] = [item for item in processed[field] if item is not None]
            else:
                processed[field] = []
        
        # Asegurar que attributes sea un diccionario
        if 'attributes' not in processed or not isinstance(processed['attributes'], dict):
            processed['attributes'] = {}
        
        # Asegurar que main_category tenga valor por defecto
        if not processed.get('main_category'):
            processed['main_category'] = 'General'
        
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
        if len(cleaned) > cls._MAX_TITLE_LENGTH:
            cleaned = cleaned[:cls._MAX_TITLE_LENGTH - 3] + "..."
        
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
        if len(cleaned) > cls._MAX_DESCRIPTION_LENGTH:
            cleaned = cleaned[:cls._MAX_DESCRIPTION_LENGTH - 3] + "..."
        
        return cleaned
    
    @classmethod
    def _auto_parse_price(cls, price: Any) -> Optional[float]:
        """Extractor de precios altamente robusto"""
        if isinstance(price, list):
            for item in price:
                if isinstance(item, (str, int, float)):
                    result = cls._auto_parse_price(item)
                    if result is not None:
                        return result
            return None

        if price is None:
            return None

        if isinstance(price, (int, float)):
            return float(price)

        if not isinstance(price, str):
            return None

        text = price.strip()

        invalid_keywords = [
            "unavailable", "see price", "contact", "free",
            "n/a", "not available", "out of stock", "varies"
        ]
        if any(kw in text.lower() for kw in invalid_keywords):
            return None

        text = text.replace("USD", "$").replace("usd", "$")
        text = text.replace("€", "€ ").replace("£", "£ ")

        range_pattern = (
            r'([\$€£]?\s*\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d+)?).{0,5}[-~].{0,5}'
            r'([\$€£]?\s*\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d+)?)'
        )
        match_range = re.search(range_pattern, text)
        if match_range:
            low = cls._normalize_number(match_range.group(1))
            high = cls._normalize_number(match_range.group(2))
            if low and high:
                return min(low, high)

        single_patterns = [
            r'[\$€£]\s*\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d+)?',
            r'\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d+)?\s*[\$€£]',
            r'\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d+)?'
        ]

        for pattern in single_patterns:
            match = re.search(pattern, text)
            if match:
                number = cls._normalize_number(match.group(0))
                if number is not None:
                    return number

        return None

    @staticmethod
    def _normalize_number(value: str) -> Optional[float]:
        """Convierte strings de números internacionales a float."""
        if not value:
            return None

        value = value.replace("$", "").replace("€", "").replace("£", "")
        value = value.strip()

        if value.count(".") > 1 or ("," in value and value.rfind(",") > value.rfind(".")):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")

        try:
            return float(value)
        except:
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
        """Genera hash del contenido para detección de duplicados."""
        content_parts = [
            data.get('title', ''),
            data.get('description', ''),
            str(data.get('price', '')),
            data.get('main_category', ''),
            ' '.join(data.get('categories', [])),
            data.get('product_type', '')
        ]
        
        # Asegurar que todos sean strings
        safe_parts = []
        for part in content_parts:
            if part is None:
                part = ''
            elif isinstance(part, list):
                # Filtrar elementos None de la lista
                part = ' '.join(str(p) for p in part if p is not None)
            elif not isinstance(part, str):
                part = str(part)
            safe_parts.append(part)
        
        content_str = '|'.join(safe_parts)
        return hashlib.md5(content_str.encode()).hexdigest()

    # --------------------------------------------------
    # Métodos de instancia
    # --------------------------------------------------
    
    @property
    def product_id(self) -> str:
        """ID universal del producto"""
        for key in ["asin", "id", "productId", "product_type", "code"]:
            if hasattr(self, key):
                value = getattr(self, key)
                if value:
                    return str(value)
        return self.title or "unknown"
    
    @classmethod
    def from_dict(cls, raw: Dict, **kwargs) -> "Product":
        """Constructor automatizado desde diccionario."""
        try:
            return cls(**raw)
            
        except Exception as e:
            logger.warning(f"Error creating Product from dict: {e}")
            # Crear producto mínimo con valores seguros
            return cls._create_minimal_product(raw)

    @classmethod
    def _create_minimal_product(cls, raw: Dict) -> "Product":
        """Crea producto mínimo cuando falla la creación normal."""
        try:
            # Extraer datos mínimos
            title = raw.get('title')
            if not title or not isinstance(title, str):
                title = 'Unknown Product'
            
            # ✅ CORRECCIÓN: Proporcionar valores por defecto para todos los campos requeridos
            return cls(
                title=title[:200],
                id=raw.get('id', str(uuid.uuid4())),
                description=raw.get('description', '')[:5000] if raw.get('description') else '',
                price=cls._auto_parse_price(raw.get('price')),
                main_category=raw.get('main_category', 'General'),
                ml_processed=False,
                # ✅ Proporcionar valores por defecto para campos requeridos
                average_rating=None,
                rating_count=None
            )
        except Exception as e:
            logger.error(f"Error creating minimal product: {e}")
            # ✅ CORRECCIÓN: Producto de error con valores por defecto
            return cls(
                title='Error Product',
                price=None,
                average_rating=None,
                rating_count=None,
                description=''
            )
    
    @classmethod
    def batch_create(cls, raw_list: List[Dict]) -> List["Product"]:
        """Crea múltiples productos con procesamiento optimizado."""
        products = []
        for data in raw_list:
            try:
                product = cls.from_dict(data)
                products.append(product)
            except Exception as e:
                logger.warning(f"Error creating product: {e}")
                # ✅ CORRECCIÓN: Crear producto mínimo con valores por defecto
                product = cls(
                    title=data.get('title', 'Unknown Product'),
                    id=data.get('id', str(uuid.uuid4())),
                    price=cls._auto_parse_price(data.get('price')),
                    average_rating=None,
                    rating_count=None,
                    description=data.get('description', '')
                )
                products.append(product)
        
        return products
    
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
        
        # ✅ CORRECCIÓN: Buscar 'ORG' (no 'BRAND')
        if self.extracted_entities and 'ORG' in self.extracted_entities:
            orgs = ", ".join(self.extracted_entities['ORG'][:3])
            if orgs:
                parts.append(f"Brands: {orgs}")
        
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
        all_tags = self.tags + self.ml_tags
        return ", ".join(all_tags[:8]) if all_tags else "No tags"
    
    def to_metadata(self) -> dict:
        """Metadata enriquecida para almacenamiento vectorial."""
        try:
            metadata = {
                "id": self.id,
                "title": self.title[:100] if self.title else "Untitled Product",
                "main_category": self.main_category or "General",
                "categories": json.dumps(self.categories, ensure_ascii=False) if self.categories else "[]",
                "price": float(self.price) if self.price is not None else self._DEFAULT_PRICE,
                "average_rating": float(self.average_rating) if self.average_rating else self._DEFAULT_RATING,
                "rating_count": self.rating_count or 0,
                "description": (self.description or "")[:200],
                "product_type": self.product_type or "",
                "content_hash": self.content_hash or "",
                "features": json.dumps(self.details.features[:10], ensure_ascii=False) if self.details.features else "[]",
                "tags": json.dumps(self.tags[:5], ensure_ascii=False) if self.tags else "[]",
                "ml_tags": json.dumps(self.ml_tags[:5], ensure_ascii=False) if self.ml_tags else "[]",
                "compatible_devices": json.dumps(self.compatible_devices[:5], ensure_ascii=False) if self.compatible_devices else "[]",
                "ml_processed": self.ml_processed,
                "has_embedding": bool(self.embedding),
                "nlp_processed": self.nlp_processed,
                "has_ner": self.has_ner,
                "has_zero_shot": self.has_zero_shot,
            }

            # Campos opcionales
            if self.ner_entities:
                try:
                    metadata['ner_entities'] = json.dumps(self.ner_entities, ensure_ascii=False, default=str)
                except: pass
            
            if self.zero_shot_classification:
                try:
                    metadata['zero_shot_classification'] = json.dumps(self.zero_shot_classification, ensure_ascii=False)
                except: pass

            if self.embedding and self.embedding_model:
                try:
                    metadata["embedding"] = json.dumps(self.embedding, ensure_ascii=False)
                    metadata["embedding_model"] = self.embedding_model
                    metadata["has_embedding"] = True
                except:
                    metadata["has_embedding"] = False

            if self.predicted_category:
                metadata["predicted_category"] = self.predicted_category
                if not metadata["main_category"] or metadata["main_category"]=="General":
                    metadata["main_category"]=self.predicted_category

            if self.extracted_entities:
                try:
                    metadata["extracted_entities"]=json.dumps({k:v[:3] for k,v in self.extracted_entities.items() if v},ensure_ascii=False)
                except: pass

            logger.debug(f"[to_metadata] Producto {self.id}: categoría='{metadata['main_category']}', embedding={metadata.get('has_embedding')}")
            return metadata

        except Exception as e:
            logger.error(f"❌ Error converting to metadata: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._get_fallback_metadata()
    
    def _get_fallback_metadata(self) -> dict:
        return {
            "id": self.id,
            "title": self.title[:100] if self.title else "Untitled Product",
            "main_category": "Uncategorized",
            "categories": "[]",
            "price": self._DEFAULT_PRICE,
            "average_rating": self._DEFAULT_RATING,
            "rating_count": 0,
            "description": "",
            "product_type": "",
            "content_hash": self.content_hash or "",
            "features": "[]",
            "tags": "[]",
            "ml_tags": "[]",
            "compatible_devices": "[]",
            "ml_processed": False,
            "nlp_processed": False,
            "has_ner": False,
            "has_zero_shot": False,
        }
    
    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        self.clean_image_urls()
        return super().model_dump(*args, **kwargs)
    
    def get_summary(self) -> Dict[str, Any]:
        settings = get_settings()
        
        summary = {
            "id": self.id,
            "title": self.title,
            "price": self.price,
            "category": self.main_category,
            "predicted_category": self.predicted_category,
            "rating": self.average_rating,
            "type": self.product_type,
            "has_description": bool(self.description and self.description != "No description available"),
            "has_images": bool(self.images and (self.images.large or self.images.medium or self.images.small)),
            "feature_count": len(self.details.features) if self.details else 0,
            "tag_count": len(self.tags),
            "ml_tag_count": len(self.ml_tags),
            "ml_enabled": settings.ML_ENABLED,
            "ml_processed": self.ml_processed,
            "entity_count": sum(len(v) for v in self.extracted_entities.values())
        }
        
        if self.embedding:
            summary['has_embedding'] = True
            summary['embedding_dim'] = len(self.embedding)
            summary['embedding_model'] = self.embedding_model
        else:
            summary['has_embedding'] = False
        
        return summary
    
    def __str__(self) -> str:
        ml_info = f", ML: {self.ml_processed}"
        embedding_info = f", HasEmbedding: {bool(self.embedding)}"
        return f"Product(title='{self.title}', price={self.price}, category='{self.main_category}'{ml_info}{embedding_info})"
    
    def __repr__(self) -> str:
        return f"Product(id='{self.id}', title='{self.title}', ml_processed={self.ml_processed})"


def create_product(raw_data: Dict[str, Any]) -> Product:
    return Product.from_dict(raw_data)


def batch_create_products(raw_data_list: List[Dict[str, Any]]) -> List[Product]:
    return Product.batch_create(raw_data_list)


def get_product_metrics() -> Dict[str, Any]:
    settings = get_settings()
    
    return {
        "product_model": {
            "ml_enabled": settings.ML_ENABLED,
            "ml_features": list(settings.ML_FEATURES) if settings.ML_ENABLED else [],
            "embedding_model": settings.ML_EMBEDDING_MODEL if settings.ML_ENABLED else None,
            "max_title_length": Product._MAX_TITLE_LENGTH,
            "max_description_length": Product._MAX_DESCRIPTION_LENGTH,
            "is_ecommerce_general": True  # ✅ Indicador de e-commerce general
        }
    }


__all__ = [
    'Product',
    'ProductImage', 
    'ProductDetails',
    'MLProductProcessor',
    'create_product',
    'batch_create_products',
    'get_product_metrics'
]