from __future__ import annotations
# src/core/data/product.py
import hashlib
import re
from typing import Optional, Dict, List, Any, ClassVar
from pydantic import BaseModel, Field, validator, model_validator
from langchain_core.documents import Document
import uuid
import logging
from functools import lru_cache
import json
from urllib.parse import urlparse, urlunparse
import requests
from PIL import Image
import io

# ML/AI imports
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# ------------------------------------------------------------------
# Constants and configuration
# ------------------------------------------------------------------
class AutoProductConfig:
    """Configuración para automatización de productos"""
    
    # Modelos ML
    SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
    NER_MODEL_NAME = "en_core_web_sm"
    CLASSIFICATION_MODEL = "facebook/bart-large-mnli"
    
    # Límites y configuraciones
    MAX_TITLE_LENGTH = 200
    MAX_DESCRIPTION_LENGTH = 1000
    DEFAULT_RATING = 0.0
    DEFAULT_PRICE = 0.0
    CACHE_SIZE = 1000
    MIN_CONFIDENCE = 0.6
    
    # URLs para servicios automáticos
    PRICE_VALIDATION_API = "https://api.currencyapi.com/v3/latest"
    IMAGE_VALIDATION_TIMEOUT = 5

# ------------------------------------------------------------------
# Automated Services
# ------------------------------------------------------------------
class AutomatedProductServices:
    """Servicios automatizados para procesamiento de productos"""
    
    _instance = None
    _models_initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize_models(self):
        """Inicializa modelos ML bajo demanda"""
        if self._models_initialized:
            return
            
        try:
            logger.info("Initializing automated product services...")
            
            # Modelo de embeddings
            self.embedding_model = SentenceTransformer(AutoProductConfig.SENTENCE_MODEL_NAME)
            
            # Modelo de NER
            try:
                self.nlp = spacy.load(AutoProductConfig.NER_MODEL_NAME)
            except OSError:
                logger.warning(f"SpaCy model not found, installing...")
                import os
                os.system(f"python -m spacy download {AutoProductConfig.NER_MODEL_NAME}")
                self.nlp = spacy.load(AutoProductConfig.NER_MODEL_NAME)
            
            # Modelo de clasificación zero-shot
            self.classifier = pipeline(
                "zero-shot-classification",
                model=AutoProductConfig.CLASSIFICATION_MODEL,
                device=-1  # Usar CPU por defecto
            )
            
            # Vectorizador TF-IDF
            self.tfidf = TfidfVectorizer(
                max_features=500,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Categorías base para clasificación
            self.base_categories = [
                "electronics", "clothing", "home appliances", "books", 
                "sports equipment", "beauty products", "toys", "furniture",
                "automotive parts", "office supplies", "food and beverages",
                "health and wellness", "jewelry", "tools and hardware"
            ]
            
            self._models_initialized = True
            logger.info("Automated product services initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing automated services: {e}")
            self._models_initialized = False
    
    @property
    def ready(self) -> bool:
        return self._models_initialized

# ------------------------------------------------------------------
# Nested models automatizados
# ------------------------------------------------------------------
class AutomatedProductImage(BaseModel):
    large: Optional[str] = None
    medium: Optional[str] = None
    small: Optional[str] = None
    
    # Cache para URLs validadas
    _validated_urls: ClassVar[Dict[str, bool]] = {}
    
    @classmethod
    def safe_create(cls, image_data: Optional[Dict]) -> "AutomatedProductImage":
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
                'cdn.shopify.com', 'i.ebayimg.com', 'target.scene7.com'
            }
            
            if parsed.netloc in valid_domains:
                cls._validated_urls[url] = True
                return True
            
            # Verificación opcional más estricta (comentada por performance)
            # response = requests.head(url, timeout=2, allow_redirects=True)
            # cls._validated_urls[url] = response.status_code == 200
            # return cls._validated_urls[url]
            
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


class AutomatedProductDetails(BaseModel):
    brand: Optional[str] = Field(None, alias="Brand")
    model: Optional[str] = Field(None, alias="Model")
    features: List[str] = Field(default_factory=list)
    specifications: Dict[str, Any] = Field(default_factory=dict)
    color: Optional[str] = None
    weight: Optional[str] = None
    wireless: Optional[bool] = None
    bluetooth: Optional[bool] = None
    dimensions: Optional[str] = None
    material: Optional[str] = None

    @classmethod
    def safe_create(cls, details_data: Optional[Dict]) -> "AutomatedProductDetails":
        """Crea instancia con extracción automática de atributos"""
        if not details_data or not isinstance(details_data, dict):
            return cls()
        
        try:
            # Procesar datos con extracción automática
            processed_data = cls._auto_extract_attributes(details_data)
            return cls(**processed_data)
            
        except Exception as e:
            logger.warning(f"Error creating AutomatedProductDetails: {e}")
            return cls()

    @classmethod
    def _auto_extract_attributes(cls, data: Dict) -> Dict:
        """Extrae atributos automáticamente usando NLP"""
        services = AutomatedProductServices()
        if not services.ready:
            return data
        
        processed = data.copy()
        all_text = cls._extract_all_text(data)
        
        if not all_text.strip():
            return processed
        
        # Extraer atributos usando NER
        extracted_attrs = cls._extract_attributes_with_ner(all_text)
        processed.update(extracted_attrs)
        
        # Extraer características automáticamente
        if not processed.get('features'):
            processed['features'] = cls._extract_features_automated(all_text)
        
        # Normalizar especificaciones
        processed['specifications'] = cls._normalize_specs_automated(
            processed.get('specifications', {})
        )
        
        return processed

    @classmethod
    def _extract_all_text(cls, data: Dict) -> str:
        """Extrae todo el texto relevante para análisis"""
        text_parts = []
        
        # Agregar campos de texto
        for field in ['brand', 'model', 'color', 'weight']:
            if data.get(field):
                text_parts.append(str(data[field]))
        
        # Agregar características
        if data.get('features'):
            text_parts.extend(data['features'])
        
        # Agregar especificaciones
        if data.get('specifications'):
            for key, value in data['specifications'].items():
                text_parts.append(f"{key} {value}")
        
        return ' '.join(text_parts)

    @classmethod
    def _extract_attributes_with_ner(cls, text: str) -> Dict[str, Any]:
        """Extrae atributos usando Named Entity Recognition"""
        services = AutomatedProductServices()
        extracted = {}
        
        try:
            doc = services.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ == "ORG" and not extracted.get('brand'):
                    extracted['brand'] = ent.text
                elif ent.label_ == "PRODUCT" and not extracted.get('model'):
                    extracted['model'] = ent.text
                elif ent.label_ == "COLOR" and not extracted.get('color'):
                    extracted['color'] = ent.text
                elif ent.label_ == "QUANTITY" and "kg" in ent.text.lower():
                    extracted['weight'] = ent.text
                elif ent.label_ == "GPE" and not extracted.get('material'):
                    # Algunos materiales pueden ser detectados como GPE
                    material_keywords = ['steel', 'aluminum', 'plastic', 'wood', 'leather']
                    if any(kw in ent.text.lower() for kw in material_keywords):
                        extracted['material'] = ent.text
            
            # Detección de características booleanas
            text_lower = text.lower()
            if any(word in text_lower for word in ['wireless', 'wifi', 'bluetooth']):
                extracted['wireless'] = True
            if 'bluetooth' in text_lower:
                extracted['bluetooth'] = True
                
        except Exception as e:
            logger.debug(f"Error in NER attribute extraction: {e}")
        
        return extracted

    @classmethod
    def _extract_features_automated(cls, text: str) -> List[str]:
        """Extrae características automáticamente del texto"""
        try:
            # Tokenizar y limpiar texto
            words = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
            
            # Tagging de partes del discurso
            tagged = pos_tag(filtered_words)
            
            # Extraer sustantivos y adjetivos como características potenciales
            features = []
            for word, pos in tagged:
                if pos.startswith('NN') and len(word) > 2:  # Sustantivos
                    features.append(word.title())
                elif pos.startswith('JJ') and len(word) > 3:  # Adjetivos
                    features.append(word.title())
            
            # Devolver características únicas
            return list(set(features))[:10]  # Limitar a 10 características
            
        except Exception as e:
            logger.debug(f"Error in automated feature extraction: {e}")
            return []

    @classmethod
    def _normalize_specs_automated(cls, specs: Dict[str, Any]) -> Dict[str, str]:
        """Normaliza especificaciones automáticamente"""
        normalized = {}
        
        for key, value in specs.items():
            if value is None:
                continue
                
            key_normalized = key.strip().lower()
            
            # Normalizar Best Sellers Rank
            if key_normalized == "best sellers rank" and isinstance(value, dict):
                normalized[key] = ", ".join(f"{k}: {v}" for k, v in value.items())
            elif isinstance(value, (dict, list)):
                normalized[key] = str(value)
            else:
                normalized[key] = str(value).strip()
        
        return normalized

    def get_auto_dimensions(self) -> Optional[str]:
        """Extrae dimensiones automáticamente de las especificaciones"""
        dimension_patterns = [
            r'(\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?\s*(?:cm|in|inch|"))',
            r'(\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?\s*(?:cm|in|inch|"))',
            r'(\d+(?:\.\d+)?\s*cm\s*[x×]\s*\d+(?:\.\d+)?\s*cm\s*[x×]\s*\d+(?:\.\d+)?\s*cm)',
            r'(\d+(?:\.\d+)?\s*"\s*[x×]\s*\d+(?:\.\d+)?\s*"\s*[x×]\s*\d+(?:\.\d+)?\s*")'
        ]
        
        all_specs_text = ' '.join(str(v) for v in self.specifications.values())
        
        for pattern in dimension_patterns:
            match = re.search(pattern, all_specs_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None


# ------------------------------------------------------------------
# Main product entity automatizada
# ------------------------------------------------------------------
class AutomatedProduct(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field("Unknown Product", min_length=1, max_length=AutoProductConfig.MAX_TITLE_LENGTH)
    main_category: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    price: Optional[float] = Field(None, ge=0)
    average_rating: Optional[float] = Field(None, ge=0, le=5)
    rating_count: Optional[int] = Field(None, alias="rating_number", ge=0)
    images: Optional[AutomatedProductImage] = None
    details: AutomatedProductDetails = Field(default_factory=AutomatedProductDetails)
    product_type: Optional[str] = None
    compatible_devices: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    attributes: Dict[str, str] = Field(default_factory=dict)
    description: Optional[str] = Field(None, max_length=AutoProductConfig.MAX_DESCRIPTION_LENGTH)
    
    # Campos calculados automáticamente
    auto_category_confidence: Optional[float] = Field(None, ge=0, le=1)
    auto_tags_confidence: Optional[float] = Field(None, ge=0, le=1)
    content_hash: Optional[str] = None

    # --------------------------------------------------
    # Validators automatizados
    # --------------------------------------------------
    @model_validator(mode='before')
    @classmethod
    def auto_process_data(cls, data: Any) -> Any:
        """Procesamiento automático completo de datos"""
        if not isinstance(data, dict):
            return data
        
        processed = data.copy()
        
        # Inicializar servicios automáticos
        services = AutomatedProductServices()
        services.initialize_models()
        
        # Procesamiento automático
        processed = cls._auto_enrich_data(processed)
        processed = cls._auto_clean_data(processed)
        processed = cls._auto_classify_data(processed)
        
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
        processed['details'] = AutomatedProductDetails.safe_create(details_data).dict()
        
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
        
        return processed

    @classmethod
    def _auto_classify_data(cls, data: Dict) -> Dict:
        """Clasifica datos automáticamente usando ML"""
        processed = data.copy()
        services = AutomatedProductServices()
        
        if not services.ready:
            return processed
        
        # Texto para clasificación
        classification_text = cls._build_classification_text(data)
        
        if not classification_text.strip():
            return processed
        
        # Clasificación automática de categoría
        if not processed.get('product_type'):
            category_result = cls._auto_classify_category(classification_text)
            if category_result:
                processed['product_type'] = category_result['category']
                processed['auto_category_confidence'] = category_result['confidence']
        
        # Generación automática de tags
        if not processed.get('tags'):
            auto_tags = cls._auto_generate_tags(classification_text)
            if auto_tags:
                processed['tags'] = auto_tags
                processed['auto_tags_confidence'] = min(1.0, len(auto_tags) / 10.0)
        
        # Inferencia automática de dispositivos compatibles
        if not processed.get('compatible_devices'):
            processed['compatible_devices'] = cls._auto_infer_devices(classification_text)
        
        # Generar hash de contenido
        processed['content_hash'] = cls._generate_content_hash(data)
        
        return processed

    @classmethod
    def _build_classification_text(cls, data: Dict) -> str:
        """Construye texto para clasificación ML"""
        text_parts = [
            data.get('title', ''),
            data.get('description', ''),
            data.get('main_category', ''),
            ' '.join(data.get('categories', [])),
        ]
        
        # Agregar detalles si existen
        details = data.get('details', {})
        if details:
            text_parts.extend([
                ' '.join(details.get('features', [])),
                ' '.join(f"{k} {v}" for k, v in details.get('specifications', {}).items())
            ])
        
        return ' '.join(filter(None, text_parts))

    @classmethod
    def _auto_classify_category(cls, text: str) -> Optional[Dict[str, Any]]:
        """Clasifica categoría automáticamente usando zero-shot learning"""
        services = AutomatedProductServices()
        
        try:
            result = services.classifier(
                text,
                candidate_labels=services.base_categories,
                multi_label=False
            )
            
            # Devolver si la confianza es suficientemente alta
            if result['scores'][0] >= AutoProductConfig.MIN_CONFIDENCE:
                return {
                    'category': result['labels'][0],
                    'confidence': result['scores'][0]
                }
                
        except Exception as e:
            logger.debug(f"Error in auto category classification: {e}")
        
        return None

    @classmethod
    def _auto_generate_tags(cls, text: str) -> List[str]:
        """Genera tags automáticamente usando TF-IDF y NER"""
        services = AutomatedProductServices()
        tags = set()
        
        try:
            # Extraer keywords con TF-IDF
            tfidf_matrix = services.tfidf.fit_transform([text])
            feature_names = services.tfidf.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Agregar top keywords como tags
            top_indices = scores.argsort()[-5:][::-1]
            for idx in top_indices:
                if scores[idx] > 0.1:  # Threshold mínimo
                    tags.add(feature_names[idx].title())
            
            # Extraer entidades como tags
            doc = services.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PRODUCT", "ORG", "GPE"] and len(ent.text) > 2:
                    tags.add(ent.text.title())
            
        except Exception as e:
            logger.debug(f"Error in auto tag generation: {e}")
        
        return list(tags)[:8]  # Limitar a 8 tags

    @classmethod
    def _auto_infer_devices(cls, text: str) -> List[str]:
        """Infere dispositivos compatibles automáticamente"""
        device_keywords = {
            "Laptop": ["laptop", "notebook", "macbook", "computer"],
            "Tablet": ["tablet", "ipad", "surface pro"],
            "Smartphone": ["smartphone", "iphone", "android", "phone", "mobile"],
            "Desktop": ["desktop", "pc", "workstation"],
            "Smart TV": ["smart tv", "television", "tv", "android tv"]
        }
        
        text_lower = text.lower()
        devices = set()
        
        for device, keywords in device_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                devices.add(device)
        
        return sorted(devices)

    @classmethod
    def _auto_clean_title(cls, title: str) -> str:
        """Limpia título automáticamente"""
        if not title or not isinstance(title, str):
            return "Unknown Product"
        
        # Remover caracteres extraños y normalizar espacios
        cleaned = re.sub(r'[^\w\s\-\.]', ' ', title)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Capitalización inteligente
        words = cleaned.split()
        if words:
            # Mantener acrónimos en mayúsculas
            cleaned_words = []
            for word in words:
                if word.isupper() or (len(word) <= 3 and word.isalpha()):
                    cleaned_words.append(word)
                else:
                    cleaned_words.append(word.capitalize())
            cleaned = ' '.join(cleaned_words)
        
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
    def _auto_parse_price(cls, price: Any) -> Optional[float]:
        """Parsea precio automáticamente"""
        if price is None:
            return None
        
        if isinstance(price, (int, float)):
            return float(price)
        
        if isinstance(price, str):
            # Múltiples estrategias de parsing
            patterns = [
                r'[\$€£]?(\d+(?:\.\d{2})?)',  # $123.45
                r'(\d+(?:\.\d{2})?)\s*USD',   # 123.45 USD
                r'price:\s*[\$€£]?(\d+(?:\.\d{2})?)',  # Price: $123.45
            ]
            
            for pattern in patterns:
                match = re.search(pattern, price)
                if match:
                    try:
                        return float(match.group(1))
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
                normalized.add(category.strip().title())
        
        return sorted(normalized)

    @classmethod
    def _auto_clean_rating(cls, rating: Any) -> Optional[float]:
        """Limpia rating automáticamente"""
        if rating is None:
            return None
        
        try:
            rating_float = float(rating)
            if 0 <= rating_float <= 5:
                return round(rating_float, 1)
        except (ValueError, TypeError):
            pass
        
        return None

    @classmethod
    def _generate_content_hash(cls, data: Dict) -> str:
        """Genera hash del contenido para detección de duplicados"""
        content_str = ''.join([
            data.get('title', ''),
            data.get('description', ''),
            str(data.get('price', '')),
            data.get('main_category', '')
        ])
        return hashlib.md5(content_str.encode()).hexdigest()

    # --------------------------------------------------
    # Constructors and methods
    # --------------------------------------------------

    @classmethod
    def from_dict(cls, raw: Dict) -> "AutomatedProduct":
        """Constructor automatizado desde diccionario"""
        try:
            # Usar el procesamiento automático del model_validator
            return cls(**raw)
            
        except Exception as e:
            logger.warning(f"Error creating AutomatedProduct from dict: {e}")
            # Crear producto mínimo con valores por defecto
            return cls(
                title=raw.get('title', 'Unknown Product'),
                id=raw.get('id', str(uuid.uuid4()))
            )

    def clean_image_urls(self) -> None:
        """Limpia URLs de imágenes automáticamente"""
        if self.images:
            self.images.clean_urls_automated()

    def to_text(self) -> str:
        """Representación optimizada para embedding"""
        parts = [
            self.title,
            f"Category: {self.main_category or 'Uncategorized'}",
            self.description or "No description available",
            f"Price: {self._format_price()}",
            f"Rating: {self._format_rating()}",
            f"Type: {self.product_type or 'No type specified'}",
            f"Tags: {self._format_tags()}",
            f"Confidence: {self._format_confidence()}"
        ]
        
        return " ".join(filter(None, parts))

    def _format_confidence(self) -> str:
        """Formatea niveles de confianza automáticos"""
        confidences = []
        if self.auto_category_confidence:
            confidences.append(f"Category: {self.auto_category_confidence:.2f}")
        if self.auto_tags_confidence:
            confidences.append(f"Tags: {self.auto_tags_confidence:.2f}")
        
        return f"Auto-confidence: [{', '.join(confidences)}]" if confidences else ""

    def _format_price(self) -> str:
        if isinstance(self.price, (int, float)):
            return f"${self.price:.2f}"
        return "Price not available"

    def _format_rating(self) -> str:
        if isinstance(self.average_rating, (int, float)):
            count = f"({self.rating_count} reviews)" if self.rating_count else ""
            return f"{self.average_rating:.1f}/5 {count}".strip()
        return "No rating available"

    def _format_tags(self) -> str:
        return ", ".join(self.tags[:10]) if self.tags else "No tags"

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
                "description": (self.description or "")[:200],
                "product_type": self.product_type or "",
                "auto_category_confidence": self.auto_category_confidence or 0.0,
                "auto_tags_confidence": self.auto_tags_confidence or 0.0,
                "content_hash": self.content_hash or "",
                "features": json.dumps(self.details.features[:15], ensure_ascii=False) if self.details else "[]",
                "tags": json.dumps(self.tags[:8], ensure_ascii=False) if self.tags else "[]"
            }
            
            # Agregar atributos automáticos si existen
            if self.details:
                if self.details.brand:
                    metadata["brand"] = self.details.brand
                if self.details.model:
                    metadata["model"] = self.details.model
                if self.details.color:
                    metadata["color"] = self.details.color
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error converting to metadata: {e}")
            return self._get_fallback_metadata()

    def _get_fallback_metadata(self) -> dict:
        return {
            "id": self.id,
            "title": self.title[:100] if self.title else "Untitled Product",
            "main_category": "Uncategorized",
            "categories": "[]",
            "price": AutoProductConfig.DEFAULT_PRICE,
            "average_rating": AutoProductConfig.DEFAULT_RATING,
            "description": "",
            "product_type": "",
            "auto_category_confidence": 0.0,
            "auto_tags_confidence": 0.0,
            "content_hash": self.content_hash or "",
            "features": "[]",
            "tags": "[]"
        }

    def to_document(self) -> Document:
        """Documento enriquecido con metadatos automáticos"""
        return Document(
            page_content=self.to_text(),
            metadata={
                "id": self.id,
                "title": self.title[:100],
                "price": float(self.price) if self.price is not None else AutoProductConfig.DEFAULT_PRICE,
                "rating": float(self.average_rating) if self.average_rating is not None else AutoProductConfig.DEFAULT_RATING,
                "category": self.main_category or "Uncategorized",
                "product_type": self.product_type or "Unknown",
                "auto_confidence": self.auto_category_confidence or 0.0,
                "content_hash": self.content_hash or ""
            }
        )

    def get_automation_stats(self) -> Dict[str, Any]:
        """Estadísticas del proceso automático"""
        return {
            "category_confidence": self.auto_category_confidence or 0.0,
            "tags_confidence": self.auto_tags_confidence or 0.0,
            "auto_generated_tags": len(self.tags) if self.tags else 0,
            "auto_detected_features": len(self.details.features) if self.details else 0,
            "content_hash": self.content_hash or ""
        }

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """Incluye procesamiento automático en el dump"""
        self.clean_image_urls()
        return super().model_dump(*args, **kwargs)


# Alias para compatibilidad
Product = AutomatedProduct
ProductImage = AutomatedProductImage
ProductDetails = AutomatedProductDetails