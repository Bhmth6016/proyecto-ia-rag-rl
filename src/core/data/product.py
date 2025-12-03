from __future__ import annotations
# src/core/data/product.py
import hashlib
import re
from typing import Optional, Dict, List, Any, ClassVar, Union
from pydantic import BaseModel, Field, model_validator
import uuid
import logging
from functools import lru_cache
import json
from urllib.parse import urlparse, urlunparse
import numpy as np

# Importamos el preprocesador ML como dependencia opcional
try:
    from .ml_processor import ProductDataPreprocessor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    ProductDataPreprocessor = None
    logging.warning("ML dependencies not available. ML features disabled.")

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
    
    # Configuraciones ML (opcionales)
    ML_ENABLED = ML_AVAILABLE
    DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
    DEFAULT_CATEGORIES = [
        "Electronics", "Home & Kitchen", "Clothing & Accessories", 
        "Sports & Outdoors", "Books", "Health & Beauty", 
        "Toys & Games", "Automotive", "Office Supplies", "Food & Beverages"
    ]

# ------------------------------------------------------------------
# ML Processor wrapper (compatible con sistema existente)
# ------------------------------------------------------------------
class MLProductEnricher:
    """Wrapper para enriquecimiento ML que se integra con el sistema Product existente"""
    
    _instance = None
    _preprocessor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_preprocessor(cls, config: Dict[str, Any] = None) -> Optional[Any]:
        """Obtiene el preprocesador ML (singleton con lazy loading)"""
        if not AutoProductConfig.ML_ENABLED:
            logger.warning("ML features are disabled. Install transformers and sentence-transformers to enable.")
            return None
        
        if cls._preprocessor is None and ML_AVAILABLE:
            try:
                config = config or {}
                cls._preprocessor = ProductDataPreprocessor(
                    categories=config.get('categories', AutoProductConfig.DEFAULT_CATEGORIES),
                    use_gpu=config.get('use_gpu', False),
                    embedding_model=config.get('embedding_model', AutoProductConfig.DEFAULT_EMBEDDING_MODEL)
                )
                logger.info("ML ProductEnricher inicializado exitosamente")
            except Exception as e:
                logger.error(f"Error inicializando ML preprocessor: {e}")
                cls._preprocessor = None
        
        return cls._preprocessor
    
    @classmethod
    def enrich_product(
        cls, 
        product_data: Dict[str, Any],
        enable_features: List[str] = None,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Enriquece datos de producto con capacidades ML.
        
        Args:
            product_data: Datos del producto
            enable_features: Lista de features ML a habilitar
            config: Configuración para el preprocesador
            
        Returns:
            Datos enriquecidos
        """
        if not AutoProductConfig.ML_ENABLED:
            return product_data
        
        preprocessor = cls.get_preprocessor(config)
        if not preprocessor:
            return product_data
        
        try:
            # Configurar features a habilitar
            if enable_features is None:
                enable_features = ['category', 'entities', 'tags', 'embedding']
            
            # Extraer texto para procesamiento
            title = product_data.get('title', '')
            description = product_data.get('description', '')
            
            # Si no hay texto suficiente, saltar procesamiento ML
            if not title and not description:
                return product_data
            
            enriched_data = product_data.copy()
            
            # Preparar texto completo
            full_text = f"{title}. {description}".strip()
            
            # 1. Clasificación de categoría (Zero-Shot)
            if 'category' in enable_features and title:
                predicted_category = preprocessor._predict_category_zero_shot(full_text)
                if predicted_category:
                    enriched_data['predicted_category'] = predicted_category
                    
                    # Si no hay categoría principal, usar la predicha
                    if 'main_category' not in enriched_data or not enriched_data['main_category']:
                        enriched_data['main_category'] = predicted_category
            
            # 2. Extracción de entidades (NER)
            if 'entities' in enable_features:
                entities = preprocessor._extract_entities_ner(full_text)
                if entities:
                    enriched_data['extracted_entities'] = entities
                    
                    # Extraer marca y modelo si están disponibles
                    if 'ORG' in entities and entities['ORG']:
                        enriched_data.setdefault('attributes', {})['brand'] = entities['ORG'][0]
                    if 'PRODUCT' in entities and entities['PRODUCT']:
                        enriched_data.setdefault('attributes', {})['model'] = entities['PRODUCT'][0]
            
            # 3. Generación de tags (TF-IDF)
            if 'tags' in enable_features:
                # Nota: TF-IDF requiere entrenamiento previo con fit_tfidf()
                if hasattr(preprocessor, '_tfidf_fitted') and preprocessor._tfidf_fitted:
                    tags = preprocessor._generate_tags_tfidf(full_text)
                    if tags:
                        # Separar tags ML de tags manuales
                        enriched_data['ml_tags'] = tags
                        
                        # Combinar con tags existentes
                        existing_tags = set(enriched_data.get('tags', []))
                        new_tags = [tag for tag in tags if tag not in existing_tags]
                        if new_tags:
                            enriched_data.setdefault('tags', []).extend(new_tags[:5])
            
            # 4. Embeddings semánticos
            if 'embedding' in enable_features:
                embedding = preprocessor._generate_embedding(full_text)
                if embedding is not None:
                    enriched_data['embedding'] = embedding
                    enriched_data['embedding_model'] = preprocessor.embedding_model_name
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"Error en enriquecimiento ML: {e}")
            # En caso de error, devolver datos originales
            return product_data
    
    @classmethod
    def enrich_batch(
        cls,
        products_data: List[Dict[str, Any]],
        enable_features: List[str] = None,
        config: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Enriquece un lote de productos de manera eficiente.
        
        Args:
            products_data: Lista de datos de productos
            enable_features: Features ML a habilitar
            config: Configuración del preprocesador
            
        Returns:
            Lista de productos enriquecidos
        """
        if not AutoProductConfig.ML_ENABLED or not products_data:
            return products_data
        
        preprocessor = cls.get_preprocessor(config)
        if not preprocessor:
            return products_data
        
        try:
            # Usar el método de batch processing del preprocesador
            enriched_batch = preprocessor.preprocess_batch(products_data)
            return enriched_batch
            
        except Exception as e:
            logger.error(f"Error en batch ML enrichment: {e}")
            return products_data
    
    @classmethod
    def fit_tfidf(cls, descriptions: List[str]) -> bool:
        """Entrena el modelo TF-IDF con descripciones."""
        if not AutoProductConfig.ML_ENABLED:
            return False
        
        preprocessor = cls.get_preprocessor()
        if not preprocessor:
            return False
        
        try:
            preprocessor.fit_tfidf(descriptions)
            return True
        except Exception as e:
            logger.error(f"Error entrenando TF-IDF: {e}")
            return False
    
    @classmethod
    def get_metrics(cls) -> Dict[str, Any]:
        """Obtiene métricas del sistema ML."""
        if not AutoProductConfig.ML_ENABLED:
            return {"ml_enabled": False}
        
        preprocessor = cls.get_preprocessor()
        if not preprocessor:
            return {"ml_enabled": False, "preprocessor_loaded": False}
        
        try:
            metrics = {
                "ml_enabled": True,
                "preprocessor_loaded": True,
                "models_loaded": preprocessor.get_model_info(),
                "embedding_cache_size": len(getattr(preprocessor, '_embedding_cache', {})),
                "tfidf_fitted": getattr(preprocessor, '_tfidf_fitted', False)
            }
            return metrics
        except Exception as e:
            logger.error(f"Error obteniendo métricas ML: {e}")
            return {"ml_enabled": True, "error": str(e)}

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
    """Modelo principal de producto con procesamiento automático y ML opcional"""
    
    # Campos principales (existentes)
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
    
    # Nuevos campos ML (opcionales)
    predicted_category: Optional[str] = None
    extracted_entities: Dict[str, List[str]] = Field(default_factory=dict)
    ml_tags: List[str] = Field(default_factory=list)  # Tags generados por ML
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    similar_products: List[Dict[str, Any]] = Field(default_factory=list)
    ml_processed: bool = Field(default=False)  # Flag para indicar si se procesó con ML
    
    # Configuración de ML
    _ml_config: ClassVar[Dict[str, Any]] = {
        'enabled': AutoProductConfig.ML_ENABLED,
        'features': ['category', 'entities', 'tags', 'embedding'],
        'categories': AutoProductConfig.DEFAULT_CATEGORIES
    }
    
    # --------------------------------------------------
    # Validators simplificados
    # --------------------------------------------------
    @model_validator(mode='before')
    @classmethod
    def auto_process_data(cls, data: Any) -> Any:
        """Procesamiento automático completo de datos (incluye ML si está habilitado)"""
        if not isinstance(data, dict):
            return data
        
        processed = data.copy()
        
        # Procesamiento automático base
        processed = cls._auto_enrich_data(processed)
        processed = cls._auto_clean_data(processed)
        
        # Procesamiento ML (opcional)
        if cls._ml_config['enabled']:
            processed = cls._apply_ml_processing(processed)
        
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
        
        # Asegurar que ml_tags sea una lista
        if 'ml_tags' in processed:
            if isinstance(processed['ml_tags'], str):
                processed['ml_tags'] = [processed['ml_tags']]
            elif not isinstance(processed['ml_tags'], list):
                processed['ml_tags'] = []
        
        return processed
    
    @classmethod
    def _apply_ml_processing(cls, data: Dict) -> Dict:
        """Aplica procesamiento ML a los datos del producto"""
        if not cls._ml_config['enabled']:
            return data
        
        try:
            # Usar el enriquecedor ML
            ml_enriched = MLProductEnricher.enrich_product(
                product_data=data,
                enable_features=cls._ml_config['features'],
                config={'categories': cls._ml_config['categories']}
            )
            
            # Marcar como procesado con ML
            ml_enriched['ml_processed'] = True
            
            return ml_enriched
            
        except Exception as e:
            logger.warning(f"ML processing failed, falling back to basic processing: {e}")
            data['ml_processed'] = False
            return data
    
    @classmethod
    def configure_ml(cls, enabled: bool = False, features: Optional[List[str]] = None, 
                    categories: Optional[List[str]] = None):
        """Configura ML una sola vez"""
        # Verificar si ya está configurado
        if hasattr(cls, '_ml_configured') and cls._ml_configured:
            return
        
        cls._ml_config = {
            'enabled': enabled,
            'features': features or ["category", "entities"],
            'categories': categories or cls.DEFAULT_CATEGORIES
        }
        
        # Marcar como configurado
        cls._ml_configured = True
        
        # Loggear solo una vez
        if enabled:
            logger.info(f"✅ ML configuration: {cls._ml_config}")
        else:
            logger.debug(f"ML configuration (disabled): {cls._ml_config}")
    
    @property
    def product_id(self) -> str:
        """
        ID universal del producto, usado por el sistema RAG y RL.
        Busca en varios campos típicos (asin, id, productId, product_type, code)
        y usa el título como último fallback.
        """
        for key in ["asin", "id", "productId", "product_type", "code"]:
            if hasattr(self, key):
                value = getattr(self, key)
                if value:
                    return str(value)
        return self.title or "unknown"
    
    # --------------------------------------------------
    # Métodos estáticos de utilidad (mantenidos del original)
    # --------------------------------------------------
    
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
    def _auto_parse_price(cls, price: Any) -> Optional[float]:
        """Extractor de precios altamente robusto para datos reales de ecommerce."""
        # Mantener la implementación original
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
    # Métodos de instancia
    # --------------------------------------------------
    
    @classmethod
    def from_dict(
        cls, 
        raw: Dict, 
        ml_enrich: bool = None,
        ml_features: List[str] = None
    ) -> "Product":
        """
        Constructor automatizado desde diccionario con ML opcional.
        
        Args:
            raw: Datos crudos del producto
            ml_enrich: Si es True, aplica enriquecimiento ML (usa configuración por defecto si es None)
            ml_features: Lista de features ML a habilitar (solo si ml_enrich=True)
        """
        try:
            # Configurar ML si se especifica
            if ml_enrich is not None:
                original_ml_config = cls._ml_config.copy()
                cls.configure_ml(enabled=ml_enrich)
                if ml_features:
                    cls.configure_ml(features=ml_features)
            
            # Crear producto
            product = cls(**raw)
            
            # Restaurar configuración ML original
            if ml_enrich is not None:
                cls._ml_config = original_ml_config
            
            return product
            
        except Exception as e:
            logger.warning(f"Error creating Product from dict: {e}")
            # Crear producto mínimo con valores por defecto
            return cls(
                title=raw.get('title', 'Unknown Product'),
                id=raw.get('id', str(uuid.uuid4()))
            )
    
    @classmethod
    def batch_create(
        cls,
        raw_list: List[Dict],
        ml_enrich: bool = False,
        batch_size: int = 32
    ) -> List["Product"]:
        """
        Crea múltiples productos con procesamiento optimizado.
        
        Args:
            raw_list: Lista de diccionarios con datos de productos
            ml_enrich: Si es True, aplica enriquecimiento ML por lotes
            batch_size: Tamaño del lote para procesamiento ML
            
        Returns:
            Lista de productos creados
        """
        products = []
        
        # Si ML está habilitado, procesar por lotes
        if ml_enrich and AutoProductConfig.ML_ENABLED:
            try:
                # Enriquecer datos con ML
                enriched_data = MLProductEnricher.enrich_batch(
                    raw_list,
                    enable_features=cls._ml_config['features'],
                    config={'categories': cls._ml_config['categories']}
                )
                
                # Crear productos desde datos enriquecidos
                for data in enriched_data:
                    try:
                        product = cls(**data)
                        products.append(product)
                    except Exception as e:
                        logger.warning(f"Error creating product from enriched data: {e}")
                        # Fallback: crear sin ML
                        product = cls(**data)
                        products.append(product)
                
            except Exception as e:
                logger.error(f"Batch ML enrichment failed: {e}")
                # Fallback: procesamiento individual sin ML
                ml_enrich = False
        
        # Procesamiento sin ML o fallback
        if not ml_enrich or not products:
            for data in raw_list:
                try:
                    product = cls.from_dict(data, ml_enrich=False)
                    products.append(product)
                except Exception as e:
                    logger.warning(f"Error creating product: {e}")
                    # Crear producto mínimo
                    product = cls(
                        title=data.get('title', 'Unknown Product'),
                        id=data.get('id', str(uuid.uuid4()))
                    )
                    products.append(product)
        
        return products
    
    def get_similarity_score(self, other_product: "Product") -> Optional[float]:
        """
        Calcula similitud coseno entre embeddings de dos productos.
        
        Args:
            other_product: Otro producto para comparar
            
        Returns:
            Score de similitud (0-1) o None si no hay embeddings
        """
        if not self.embedding or not other_product.embedding:
            return None
        
        # Verificar que los embeddings sean del mismo modelo
        if self.embedding_model != other_product.embedding_model:
            logger.warning("Embeddings from different models, similarity may not be accurate")
        
        try:
            # Convertir a arrays numpy
            vec1 = np.array(self.embedding)
            vec2 = np.array(other_product.embedding)
            
            # Normalizar vectores
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Calcular similitud coseno
            similarity = np.dot(vec1_norm, vec2_norm)
            
            # Asegurar que esté en rango [0, 1]
            return max(0.0, min(1.0, float(similarity)))
        
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return None
    
    def enrich_with_ml(self, features: List[str] = None) -> "Product":
        """
        Aplica enriquecimiento ML a un producto existente.
        
        Args:
            features: Lista de features ML a aplicar (None para usar configuración)
            
        Returns:
            Producto enriquecido (nueva instancia)
        """
        if not AutoProductConfig.ML_ENABLED:
            logger.warning("ML features are disabled")
            return self
        
        try:
            # Convertir producto a dict
            product_dict = self.model_dump()
            
            # Aplicar enriquecimiento ML
            enriched_dict = MLProductEnricher.enrich_product(
                product_dict,
                enable_features=features or self._ml_config['features'],
                config={'categories': self._ml_config['categories']}
            )
            
            # Crear nuevo producto enriquecido
            enriched_product = self.__class__(**enriched_dict)
            return enriched_product
            
        except Exception as e:
            logger.error(f"ML enrichment failed: {e}")
            return self
    
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
        
        # Agregar entidades ML si existen
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
        """Metadata enriquecida con información automática y ML"""
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
                "ml_tags": json.dumps(self.ml_tags[:5], ensure_ascii=False) if self.ml_tags else "[]",
                "compatible_devices": json.dumps(self.compatible_devices[:5], ensure_ascii=False) if self.compatible_devices else "[]",
                "ml_processed": self.ml_processed,
                "has_embedding": bool(self.embedding),
                "embedding_model": self.embedding_model or "",
                "embedding_dim": len(self.embedding) if self.embedding else 0
            }
            
            # Agregar categoría predicha si existe
            if self.predicted_category:
                metadata['predicted_category'] = self.predicted_category
            
            # Agregar entidades extraídas si existen
            if self.extracted_entities:
                metadata['extracted_entities'] = json.dumps(
                    {k: v[:3] for k, v in self.extracted_entities.items() if v},
                    ensure_ascii=False
                )
            
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
            "ml_tags": "[]",
            "compatible_devices": "[]",
            "ml_processed": False,
            "has_embedding": False,
            "embedding_model": "",
            "embedding_dim": 0
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
            "predicted_category": self.predicted_category,
            "rating": self.average_rating,
            "type": self.product_type,
            "has_description": bool(self.description and self.description != "No description available"),
            "has_images": bool(self.images and (self.images.large or self.images.medium or self.images.small)),
            "feature_count": len(self.details.features) if self.details else 0,
            "tag_count": len(self.tags),
            "ml_tag_count": len(self.ml_tags),
            "ml_enriched": self.ml_processed,
            "has_embedding": bool(self.embedding),
            "entity_count": sum(len(v) for v in self.extracted_entities.values())
        }
    
    def get_ml_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas específicas del enriquecimiento ML"""
        if not self.ml_processed:
            return {"ml_processed": False}
        
        return {
            "ml_processed": True,
            "predicted_category": self.predicted_category,
            "entity_groups": list(self.extracted_entities.keys()),
            "total_entities": sum(len(v) for v in self.extracted_entities.values()),
            "ml_tags_count": len(self.ml_tags),
            "has_embedding": bool(self.embedding),
            "embedding_model": self.embedding_model,
            "embedding_dim": len(self.embedding) if self.embedding else 0
        }
    
    def __str__(self) -> str:
        ml_info = f", ML: {self.ml_processed}" if hasattr(self, 'ml_processed') else ""
        return f"Product(title='{self.title}', price={self.price}, category='{self.main_category}'{ml_info})"
    
    def __repr__(self) -> str:
        return f"Product(id='{self.id}', title='{self.title}', ml_processed={self.ml_processed})"


# Funciones de utilidad para el sistema completo
def create_product_pipeline(
    raw_data: Dict[str, Any],
    enable_ml: bool = True,
    ml_features: List[str] = None
) -> Product:
    """
    Pipeline completo para creación de productos.
    
    Args:
        raw_data: Datos crudos del producto
        enable_ml: Habilitar procesamiento ML
        ml_features: Features ML específicas a habilitar
        
    Returns:
        Producto procesado
    """
    # Configurar ML según parámetros
    Product.configure_ml(enabled=enable_ml)
    if ml_features:
        Product.configure_ml(features=ml_features)
    
    # Crear producto
    product = Product.from_dict(raw_data)
    
    return product


def batch_process_products(
    raw_data_list: List[Dict[str, Any]],
    enable_ml: bool = True,
    batch_size: int = 32
) -> List[Product]:
    """
    Procesa un lote de productos de manera optimizada.
    
    Args:
        raw_data_list: Lista de datos crudos
        enable_ml: Habilitar procesamiento ML
        batch_size: Tamaño del lote para ML
        
    Returns:
        Lista de productos procesados
    """
    # Configurar ML
    Product.configure_ml(enabled=enable_ml)
    
    # Procesar por lotes
    if enable_ml and AutoProductConfig.ML_ENABLED:
        return Product.batch_create(raw_data_list, ml_enrich=True, batch_size=batch_size)
    else:
        return [Product.from_dict(data, ml_enrich=False) for data in raw_data_list]


def get_system_metrics() -> Dict[str, Any]:
    """Obtiene métricas del sistema completo"""
    metrics = {
        "product_model": {
            "ml_enabled": AutoProductConfig.ML_ENABLED,
            "ml_config": Product._ml_config,
            "max_title_length": AutoProductConfig.MAX_TITLE_LENGTH,
            "max_description_length": AutoProductConfig.MAX_DESCRIPTION_LENGTH
        }
    }
    
    # Agregar métricas ML si está disponible
    if AutoProductConfig.ML_ENABLED:
        ml_metrics = MLProductEnricher.get_metrics()
        metrics["ml_system"] = ml_metrics
    
    return metrics


# Aliases para compatibilidad
ProductImage = ProductImage
ProductDetails = ProductDetails