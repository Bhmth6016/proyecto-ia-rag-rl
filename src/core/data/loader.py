# src/core/data/loader.py

import json
import pickle
import re
import zlib
import time
import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import hashlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import Counter

from tqdm import tqdm

# Configurar logging para evitar mensajes verbosos
import logging
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)

# Importaciones simplificadas para evitar dependencias circulares
try:
    from src.core.data.product import Product
    from src.core.config import settings
    from src.core.utils.logger import get_logger
except ImportError:
    # Definiciones básicas para cuando se ejecute independientemente
    import logging
    
    class Product:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        @classmethod
        def from_dict(cls, data):
            return cls(**data)
        
        def clean_image_urls(self):
            if hasattr(self, 'image_urls'):
                if not self.image_urls:
                    self.image_urls = ["https://via.placeholder.com/300"]
        
        def model_dump(self):
            return self.__dict__
    
    class settings:
        RAW_DIR = Path("./data/raw")
        PROC_DIR = Path("./data/processed")
    
    def get_logger(name):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(name)

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Configuración automática
# ------------------------------------------------------------------
class AutoCategoryConfig:
    """Configuración para categorización automática"""
    
    # Modelos pre-entrenados
    SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
    
    # Parámetros de clustering
    MIN_CLUSTER_SIZE = 5  # Reducido para trabajar con menos datos
    MAX_CATEGORIES = 20

class AutomatedDataLoader:
    """
    Cargador automatizado con ML para categorización y tagging
    """

    def __init__(
        self,
        *,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        cache_enabled: bool = True,
        disable_tqdm: bool = False,
        auto_categories: bool = True,
        auto_tags: bool = True,
        min_samples_for_training: int = 10  # Muy reducido para funcionar con pocos datos
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else settings.RAW_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else settings.PROC_DIR
        self.cache_enabled = cache_enabled
        self.disable_tqdm = disable_tqdm
        self.auto_categories = auto_categories
        self.auto_tags = auto_tags
        self.min_samples_for_training = min_samples_for_training
        
        self._ml_models = {}
        self._category_cache = {}
        self._tag_cache = {}
        self._models_initialized = False

        # Crear directorios si no existen
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_ml_models(self):
        """Inicializa modelos ML bajo demanda"""
        if self._models_initialized:
            return
            
        try:
            logger.info("Initializing ML models...")
            
            # Silenciar logs verbosos
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            
            # Modelo de embeddings para similitud semántica
            self._ml_models['embedding'] = SentenceTransformer(
                AutoCategoryConfig.SENTENCE_MODEL_NAME,
                device='cpu'
            )
            
            # Vectorizador TF-IDF
            self._ml_models['tfidf'] = TfidfVectorizer(
                max_features=200,
                stop_words='english',
                ngram_range=(1, 1)
            )
            
            self._models_initialized = True
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
            self._models_initialized = False

    def _extract_text_features(self, products: List[Product]) -> List[str]:
        """Extrae características de texto de los productos"""
        texts = []
        for product in products:
            text_parts = [
                getattr(product, 'title', "") or "",
                getattr(product, 'description', "") or "",
            ]
            
            # Manejar detalles si existen
            details = getattr(product, 'details', None)
            if details:
                if hasattr(details, 'features') and details.features:
                    text_parts.append(' '.join(str(f) for f in details.features))
            
            full_text = ' '.join(filter(None, text_parts))
            if full_text.strip():
                texts.append(full_text)
        
        return texts

    def _auto_discover_categories(self, products: List[Product]) -> Dict[str, List[str]]:
        """Descubre categorías automáticamente usando clustering"""
        if len(products) < self.min_samples_for_training:
            logger.warning(f"Not enough products ({len(products)}) for auto-categorization")
            return self._get_fallback_categories()
        
        self._initialize_ml_models()
        if not self._models_initialized:
            return self._get_fallback_categories()
        
        try:
            # Extraer textos
            texts = self._extract_text_features(products)
            
            if len(texts) < 5:
                return self._get_fallback_categories()
            
            logger.info(f"Generating embeddings for {len(texts)} products...")
            
            # Generar embeddings silenciosamente
            embeddings = self._ml_models['embedding'].encode(
                texts, 
                batch_size=16,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Determinar número óptimo de clusters
            n_clusters = min(
                AutoCategoryConfig.MAX_CATEGORIES,
                max(2, len(texts) // AutoCategoryConfig.MIN_CLUSTER_SIZE)
            )
            
            logger.info(f"Clustering into {n_clusters} categories...")
            
            # Clustering con K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Extraer keywords por cluster
            category_keywords = self._extract_cluster_keywords(texts, cluster_labels, n_clusters)
            
            logger.info(f"Discovered {len(category_keywords)} categories")
            return category_keywords
            
        except Exception as e:
            logger.error(f"Error in auto category discovery: {e}")
            return self._get_fallback_categories()

    def _extract_cluster_keywords(self, texts: List[str], labels: np.ndarray, n_clusters: int) -> Dict[str, List[str]]:
        """Extrae palabras clave representativas para cada cluster"""
        try:
            # Entrenar TF-IDF
            tfidf_matrix = self._ml_models['tfidf'].fit_transform(texts)
            feature_names = self._ml_models['tfidf'].get_feature_names_out()
            
            category_keywords = {}
            
            for cluster_id in range(n_clusters):
                cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
                
                if len(cluster_indices) < 2:
                    continue
                    
                # Calcular scores TF-IDF promedio
                cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
                
                # Obtener top keywords
                top_keyword_indices = cluster_tfidf.argsort()[-5:][::-1]
                top_keywords = [
                    feature_names[i] for i in top_keyword_indices 
                    if cluster_tfidf[i] > 0 and len(feature_names[i]) > 2
                ]
                
                if top_keywords:
                    category_name = self._generate_category_name(top_keywords)
                    category_keywords[category_name] = top_keywords[:3]
            
            return category_keywords if category_keywords else self._get_fallback_categories()
            
        except Exception as e:
            logger.error(f"Error extracting cluster keywords: {e}")
            return self._get_fallback_categories()

    def _generate_category_name(self, keywords: List[str]) -> str:
        """Genera nombre de categoría automáticamente"""
        if not keywords:
            return "general"
        
        base_name = keywords[0]
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', base_name.lower())
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')
        
        return clean_name or "general"

    def _get_fallback_categories(self) -> Dict[str, List[str]]:
        """Categorías de fallback cuando no hay suficientes datos"""
        return {
            "electronics": ["electronic", "device", "tech"],
            "software": ["software", "app", "program"],
            "games": ["game", "gaming", "video"],
            "home": ["home", "household", "domestic"],
            "sports": ["sports", "fitness", "outdoor"],
            "books": ["book", "reading", "education"],
            "clothing": ["clothing", "fashion", "apparel"],
            "general": ["product", "item", "general"]
        }

    def _get_fallback_tags(self) -> Dict[str, List[str]]:
        """Tags de fallback cuando no hay suficientes datos"""
        return {
            "wireless": ["wireless", "bluetooth"],
            "portable": ["portable", "lightweight"],
            "digital": ["digital", "download"],
            "premium": ["premium", "quality"],
            "new": ["new", "latest"]
        }

    def _infer_category_automated(self, product: Product) -> str:
        """Infere categoría automáticamente para un producto"""
        if not self.auto_categories or not self._category_cache or not self._models_initialized:
            return "unknown"
        
        try:
            # Preparar texto del producto
            text_parts = [
                getattr(product, 'title', "") or "",
                getattr(product, 'description', "") or "",
            ]
            
            details = getattr(product, 'details', None)
            if details:
                if hasattr(details, 'features') and details.features:
                    text_parts.append(' '.join(str(f) for f in details.features))
            
            text = ' '.join(filter(None, text_parts))
            
            if not text.strip():
                return "unknown"
            
            # Calcular similitud con cada categoría
            category_scores = {}
            text_embedding = self._ml_models['embedding'].encode([text], show_progress_bar=False)
            
            for category_name, keywords in self._category_cache.items():
                category_text = ' '.join(keywords)
                category_embedding = self._ml_models['embedding'].encode([category_text], show_progress_bar=False)
                similarity = cosine_similarity(text_embedding, category_embedding)[0][0]
                category_scores[category_name] = similarity
            
            best_category = max(category_scores.items(), key=lambda x: x[1])
            
            if best_category[1] > 0.3:
                return best_category[0]
            else:
                return "unknown"
                
        except Exception as e:
            return "unknown"

    def _clean_item_automated(self, item: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Limpieza automatizada con ML"""
        # Validación básica
        title = item.get("title", "").strip()
        if not title:
            raise ValueError("Missing title")
        
        # Limpieza de descripción
        description = item.get("description", "")
        if not description:
            item["description"] = "No description available"
        elif isinstance(description, list):
            item["description"] = " ".join(str(x) for x in description if x)
        
        # Categoría desde archivo (fallback)
        item["main_category"] = self._get_category_from_filename(filename)
        
        # Limpieza de precio
        price = item.get("price")
        if price is None:
            item["price"] = 0.0
        elif isinstance(price, str):
            cleaned_price = re.search(r'\d+\.?\d*', price)
            item["price"] = float(cleaned_price.group()) if cleaned_price else 0.0
        
        # Valores por defecto
        item.setdefault("average_rating", 0.0)
        item.setdefault("tags", [])
        
        # Detalles
        details = item.get("details", {})
        if not isinstance(details, dict):
            details = {}
        
        item["details"] = {
            "features": details.get("features", []),
            "specifications": details.get("specifications", {})
        }
        
        return item

    def _get_category_from_filename(self, filename: str) -> str:
        """Obtiene categoría basada en el nombre del archivo (fallback)"""
        stem = Path(filename).stem.lower()
        
        category_map = {
            'game': 'games',
            'software': 'software', 
            'electronic': 'electronics',
            'beauty': 'beauty',
            'personal': 'beauty',
            'industrial': 'industrial',
            'scientific': 'industrial',
            'book': 'books',
            'clothing': 'clothing',
            'sport': 'sports'
        }
        
        for key, category in category_map.items():
            if key in stem:
                return category
        
        return 'general'

    def _process_json_array_automated(self, raw_file: Path) -> List[Product]:
        """Procesa archivo JSON con array de productos"""
        try:
            with raw_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                data = [data]
            
            products = []
            for item in data:
                try:
                    if isinstance(item, dict):
                        cleaned_item = self._clean_item_automated(item, raw_file.name)
                        product = Product.from_dict(cleaned_item)
                        
                        # Inferir categoría automáticamente si está habilitado
                        if self.auto_categories and not cleaned_item.get("product_type"):
                            auto_category = self._infer_category_automated(product)
                            if auto_category != "unknown":
                                product.product_type = auto_category
                            else:
                                product.product_type = cleaned_item["main_category"]
                        else:
                            product.product_type = cleaned_item["main_category"]
                        
                        product.clean_image_urls()
                        products.append(product)
                except Exception:
                    continue
            
            return products
            
        except Exception as e:
            logger.error(f"Error processing JSON file {raw_file.name}: {e}")
            return []

    def _process_jsonl_file(self, raw_file: Path) -> List[Product]:
        """Procesa archivo JSONL"""
        products = []
        try:
            with raw_file.open("r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:
                            continue
                            
                        item = json.loads(line)
                        if isinstance(item, dict):
                            cleaned_item = self._clean_item_automated(item, raw_file.name)
                            product = Product.from_dict(cleaned_item)
                            
                            # Inferir categoría automáticamente
                            if self.auto_categories and not cleaned_item.get("product_type"):
                                auto_category = self._infer_category_automated(product)
                                if auto_category != "unknown":
                                    product.product_type = auto_category
                                else:
                                    product.product_type = cleaned_item["main_category"]
                            else:
                                product.product_type = cleaned_item["main_category"]
                            
                            product.clean_image_urls()
                            products.append(product)
                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        continue
                        
        except Exception as e:
            logger.error(f"Error processing JSONL file {raw_file.name}: {e}")
        
        return products

    def load_data(self, use_cache: Optional[bool] = None, output_file: Union[str, Path] = None) -> List[Product]:
        """Carga datos con procesamiento automatizado"""
        if use_cache is None:
            use_cache = self.cache_enabled

        if output_file is None:
            output_file = self.processed_dir / "products.json"

        # Verificar caché global
        if use_cache and Path(output_file).exists():
            logger.info("Loading products from cache")
            try:
                return self._load_global_cache(output_file)
            except Exception:
                logger.warning("Cache corrupted, reprocessing data...")

        # Cargar archivos
        files = self._discover_data_files()
        
        if not files:
            logger.warning("No product files found")
            return self._create_sample_data(output_file)

        # Cargar productos iniciales para entrenamiento
        logger.info("Loading sample products for ML training...")
        initial_products = []
        for file_path in files[:2]:  # Usar máximo 2 archivos para entrenamiento
            try:
                products = self._load_single_file_simple(file_path)
                if products:
                    initial_products.extend(products[:30])  # Limitar muestras
            except Exception as e:
                logger.warning(f"Error sampling from {file_path.name}: {e}")

        # Aprender categorías automáticamente
        if self.auto_categories:
            if initial_products and len(initial_products) >= 5:
                logger.info(f"Training ML models with {len(initial_products)} samples...")
                self._category_cache = self._auto_discover_categories(initial_products)
                logger.info(f"Discovered {len(self._category_cache)} categories")
            else:
                self._category_cache = self._get_fallback_categories()
                logger.info("Using fallback categories")

        # Procesar todos los archivos
        logger.info("Processing all files...")
        all_products = []
        for file_path in tqdm(files, desc="Files", disable=self.disable_tqdm):
            try:
                products = self._load_single_file_simple(file_path)[:100]
                if products:
                    all_products.extend(products)
            except Exception as e:
                logger.warning(f"Error processing {file_path.name}: {e}")

        if not all_products:
            logger.error("No products could be loaded")
            return self._create_sample_data(output_file)

        logger.info(f"Successfully loaded {len(all_products)} products")

        # Guardar productos procesados
        self.save_standardized_json(all_products, output_file)

        return all_products

    def _load_single_file_simple(self, raw_file: Path) -> List[Product]:
        """Carga un archivo de manera simple"""
        if not raw_file.exists():
            return []

        try:
            if raw_file.suffix.lower() == ".jsonl":
                return self._process_jsonl_file(raw_file)
            else:
                return self._process_json_array_automated(raw_file)
        except Exception as e:
            logger.error(f"Error loading {raw_file.name}: {e}")
            return []

    def _create_sample_data(self, output_file: Path) -> List[Product]:
        """Crea datos de ejemplo si no hay archivos disponibles"""
        logger.info("Creating sample data...")
        
        sample_products = [
            {
                "title": "Wireless Bluetooth Headphones",
                "description": "High-quality wireless headphones with noise cancellation",
                "price": 99.99,
                "main_category": "electronics",
                "product_type": "electronics",
                "tags": ["wireless", "audio"],
                "image_urls": ["https://via.placeholder.com/300"],
                "details": {
                    "features": ["Noise cancellation", "30h battery", "Bluetooth 5.0"],
                    "specifications": {"color": "black", "weight": "250g"}
                }
            },
            {
                "title": "Python Programming Book",
                "description": "Complete guide to Python programming",
                "price": 45.99,
                "main_category": "books", 
                "product_type": "books",
                "tags": ["digital", "education"],
                "image_urls": ["https://via.placeholder.com/300"],
                "details": {
                    "features": ["500 pages", "Digital download available"],
                    "specifications": {"format": "PDF", "language": "English"}
                }
            },
            {
                "title": "Gaming Mouse RGB",
                "description": "Professional gaming mouse with RGB lighting",
                "price": 59.99,
                "main_category": "electronics",
                "product_type": "electronics", 
                "tags": ["gaming", "rgb"],
                "image_urls": ["https://via.placeholder.com/300"],
                "details": {
                    "features": ["RGB lighting", "High DPI", "6 buttons"],
                    "specifications": {"sensor": "optical", "dpi": "16000"}
                }
            }
        ]
        
        products = []
        for item in sample_products:
            try:
                product = Product.from_dict(item)
                products.append(product)
            except Exception as e:
                logger.warning(f"Error creating sample product: {e}")
        
        # Guardar datos de ejemplo
        self.save_standardized_json(products, output_file)
        logger.info(f"Created {len(products)} sample products")
        
        return products

    def _discover_data_files(self) -> List[Path]:
        """Descubre automáticamente archivos de datos"""
        expected_patterns = ["*.jsonl", "*.json"]
        files = []
        
        for pattern in expected_patterns:
            files.extend(self.raw_dir.glob(pattern))
            files.extend(self.raw_dir.glob(pattern.upper()))
        
        # Filtrar archivos válidos
        valid_files = []
        for f in files:
            if f.exists() and f.stat().st_size > 0:
                valid_files.append(f)
        
        logger.info(f"Found {len(valid_files)} data files")
        return valid_files

    def _load_global_cache(self, cache_file: Path) -> List[Product]:
        """Carga productos desde caché global"""
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)

            products = []
            for item in cached_data:
                try:
                    # Asegurar que details sea un diccionario
                    if "details" not in item or not isinstance(item["details"], dict):
                        item["details"] = {}
                    
                    product = Product.from_dict(item)
                    products.append(product)
                except Exception:
                    continue

            return products
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return []

    def save_standardized_json(self, products: List[Product], output_file: Union[str, Path]) -> None:
        """Guarda productos en formato JSON estandarizado"""
        output_file = Path(output_file)
        try:
            # Convertir productos a diccionarios
            standardized_data = []
            for product in products:
                try:
                    product_dict = {}
                    for key, value in product.__dict__.items():
                        # Manejar objetos anidados
                        if hasattr(value, '__dict__'):
                            product_dict[key] = value.__dict__
                        else:
                            product_dict[key] = value
                    standardized_data.append(product_dict)
                except Exception:
                    continue
            
            # Guardar con estructura ordenada
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(standardized_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved {len(standardized_data)} products to {output_file}")
        except Exception as e:
            logger.error(f"Error saving to {output_file}: {e}")

    def get_automation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la automatización"""
        return {
            "auto_categories_enabled": self.auto_categories,
            "auto_tags_enabled": self.auto_tags,
            "discovered_categories": list(self._category_cache.keys()) if self._category_cache else [],
            "total_categories": len(self._category_cache) if self._category_cache else 0,
            "ml_models_initialized": self._models_initialized
        }

    def print_sample_products(self, count: int = 5):
        """Muestra una muestra de productos cargados"""
        products = self.load_data()
        if not products:
            print("No products available")
            return
            
        print(f"\n=== SAMPLE PRODUCTS (showing {min(count, len(products))} of {len(products)}) ===")
        for i, product in enumerate(products[:count]):
            print(f"\n{i+1}. {getattr(product, 'title', 'No title')}")
            print(f"   Category: {getattr(product, 'product_type', 'Unknown')}")
            print(f"   Price: ${getattr(product, 'price', 0):.2f}")
            print(f"   Description: {getattr(product, 'description', 'No description')[:100]}...")

# Alias para compatibilidad
DataLoader = AutomatedDataLoader

if __name__ == "__main__":
    logger.info("=== Automated Data Loader ===")

    # Crear directorios necesarios
    settings.RAW_DIR.mkdir(parents=True, exist_ok=True)
    settings.PROC_DIR.mkdir(parents=True, exist_ok=True)

    # Inicializar loader
    loader = AutomatedDataLoader(
        raw_dir=settings.RAW_DIR,
        processed_dir=settings.PROC_DIR,
        auto_categories=True,
        auto_tags=True,
        cache_enabled=True
    )

    # Cargar datos
    products = loader.load_data()

    # Mostrar estadísticas
    stats = loader.get_automation_stats()
    print(f"\n=== AUTOMATION STATISTICS ===")
    print(f"Products loaded: {len(products)}")
    print(f"Categories discovered: {stats['total_categories']}")
    print(f"ML models initialized: {stats['ml_models_initialized']}")
    
    if stats['discovered_categories']:
        print(f"Discovered categories: {', '.join(stats['discovered_categories'])}")

    # Mostrar muestra de productos
    loader.print_sample_products(3)

    logger.info("Data loading completed successfully!")