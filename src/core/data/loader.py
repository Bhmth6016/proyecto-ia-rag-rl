# src/core/data/loader.py - VERSIÓN CORREGIDA

import json
import re
import time
import os
import warnings
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ==============================================
# CONFIGURACIÓN DE LOGS Y ADVERTENCIAS
# ==============================================

warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")
warnings.filterwarnings("ignore", message="were not used when initializing")
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('sklearn').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

try:
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()
except ImportError:
    pass

# ==============================================
# IMPORTACIONES DEL SISTEMA - VERSIÓN SIMPLIFICADA
# ==============================================

try:
    # 🔥 IMPORTANTE: Importar SOLO lo esencial
    from src.core.data.product import Product, AutoProductConfig
    from src.core.config import settings
    from src.core.utils.logger import get_logger
    # 🔥 NO importar MLProductEnricher aquí - causa dependencia circular
except ImportError as e:
    # Fallback definitions simplificado
    import logging
    from pathlib import Path
    
    class Product:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
                
        asin: str | None = None
        id: str | None = None
        productId: str | None = None
        product_type: str | None = None
        code: str | None = None
        title: str | None = None
        
        @classmethod
        def from_dict(cls, data, ml_enrich=False, ml_features=None):
            # Implementación simplificada sin ML
            return cls(**data)
        
        @classmethod
        def batch_create(cls, product_dicts, ml_enrich=False, batch_size=16):
            # Implementación simplificada
            return [cls.from_dict(p) for p in product_dicts]
        
        @classmethod
        def configure_ml(cls, enabled=False, features=None, categories=None):
            # Método placeholder
            pass
        
        def clean_image_urls(self):
            if hasattr(self, 'image_urls'):
                if not self.image_urls:
                    self.image_urls = ["https://via.placeholder.com/300"]
        
        @property
        def product_id(self):
            for key in ["asin", "id", "productId", "product_type", "code"]:
                if getattr(self, key, None):
                    return getattr(self, key)
            return self.title or "unknown"
    
    class AutoProductConfig:
        ML_ENABLED = False
        DEFAULT_CATEGORIES = ["Electronics", "Books", "Home", "Clothing"]
    
    class settings:
        RAW_DIR = Path("./data/raw")
        PROC_DIR = Path("./data/processed")
        ML_ENABLED = False
        ML_FEATURES = ["category", "entities"]
        ML_CATEGORIES = ["Electronics", "Books", "Home", "Clothing"]
    
    def get_logger(name):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(name)
    
logger = get_logger(__name__)

# ==============================================
# CONFIGURACIÓN DE CATEGORÍAS AUTOMÁTICAS
# ==============================================

class AutoCategoryConfig:
    """Configuración optimizada para categorización automática"""
    SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
    MIN_CLUSTER_SIZE = 3
    MAX_CATEGORIES = 15

# ==============================================
# DATA LOADER CON INTEGRACIÓN ML COMPLETA
# ==============================================

class FastDataLoader:
    """
    Cargador optimizado con integración ML completa
    """

    def __init__(
        self,
        *,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        cache_enabled: bool = False,
        max_products_per_file: int = 500,
        auto_categories: bool = True,
        auto_tags: bool = False,
        use_progress_bar: bool = True,
        # 🔥 NUEVOS PARÁMETROS ML
        ml_enabled: Optional[bool] = None,
        ml_features: Optional[List[str]] = None,
        ml_batch_size: int = 64
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else settings.RAW_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else settings.PROC_DIR
        self.cache_enabled = cache_enabled
        self.max_products_per_file = max_products_per_file
        self.auto_categories = auto_categories
        self.auto_tags = auto_tags
        self.use_progress_bar = use_progress_bar
        
        # 🔥 CONFIGURACIÓN ML
        self.ml_enabled = ml_enabled if ml_enabled is not None else getattr(settings, "ML_ENABLED", False)
        self.ml_features = ml_features or getattr(settings, "ML_FEATURES", ["category", "entities"])
        self.ml_batch_size = ml_batch_size
        
        self._ml_models = {}
        self._category_cache = {}
        self._models_initialized = False

        # Crear directorios
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # 🔥 CONFIGURAR PRODUCTO PARA USAR ML
        self._configure_product_ml()

    def _configure_product_ml(self):
        """Configura la clase Product para usar ML según los settings"""
        try:
            # Configurar la clase Product para usar ML
            Product.configure_ml(
                enabled=self.ml_enabled,
                features=self.ml_features,
                categories=getattr(settings, "ML_CATEGORIES", AutoProductConfig.DEFAULT_CATEGORIES)
            )
            
            logger.info(f"ML Configuration: enabled={self.ml_enabled}, features={self.ml_features}")
            
        except Exception as e:
            logger.warning(f"Error configuring Product ML: {e}")

    def _initialize_ml_models(self):
        """Inicialización de modelos ML"""
        if self._models_initialized:
            return
        
        try:
            logger.info("Initializing ML models...")
            
            if not self.ml_enabled:
                logger.info("ML is disabled, skipping model initialization")
                return
            
            # Configurar cache
            os.environ['HF_HOME'] = str(Path.home() / ".cache" / "huggingface")
            
            # Configurar para rendimiento
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['OMP_NUM_THREADS'] = '1'
            
            # Modelo de embeddings
            self._ml_models['embedding'] = SentenceTransformer(
                AutoCategoryConfig.SENTENCE_MODEL_NAME,
                device='cpu'
            )
            
            # Vectorizador TF-IDF
            self._ml_models['tfidf'] = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 1)
            )
            
            self._models_initialized = True
            logger.info("ML models initialized successfully")
            if 'tags' in self.ml_features:
                self._train_tfidf_with_existing_data()
            
            self._models_initialized = True
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
            self._models_initialized = False
            
    def _train_tfidf_with_existing_data(self):
        """Entrena TF-IDF con datos de productos existentes."""
        try:
            # Verificar si hay datos procesados
            processed_file = self.processed_dir / "products.json"
            if processed_file.exists():
                with open(processed_file, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                
                if existing_data and isinstance(existing_data, list):
                    # Extraer textos para entrenar TF-IDF
                    texts = []
                    for item in existing_data[:100]:  # Usar máximo 100
                        title = item.get('title', '')
                        desc = item.get('description', '')
                        if title or desc:
                            texts.append(f"{title} {desc}".strip())
                    
                    if len(texts) >= 3:  # Mínimo 3 textos
                        logger.info(f"📊 Entrenando TF-IDF con {len(texts)} descripciones existentes")
                        self._ml_models['tfidf'].fit(texts)
                        self.tfidf_fitted = True
                        logger.info(f"✅ TF-IDF entrenado con vocabulario de {len(self._ml_models['tfidf'].get_feature_names_out())} palabras")
        
        except Exception as e:
            logger.warning(f"No se pudo entrenar TF-IDF con datos existentes: {e}")            
    

    def _clean_item_fast(self, item: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Limpieza rápida de items con configuración ML"""
        # Validación básica
        title = item.get("title", "").strip()
        if not title:
            raise ValueError("Missing title")
        
        # Limpieza mínima
        description = item.get("description", "")
        if not description:
            item["description"] = "No description"
        elif isinstance(description, list):
            item["description"] = " ".join(str(x) for x in description[:1])
        
        # Categoría desde archivo
        item["main_category"] = self._get_category_from_filename_fast(filename)
        
        # Precio
        price = item.get("price")
        if price is None:
            item["price"] = 0.0
        elif isinstance(price, str):
            cleaned_price = re.search(r'(\d+(?:[.,]\d{1,2})?)', price)
            item["price"] = float(cleaned_price.group(1).replace(',', '.')) if cleaned_price else 0.0
        
        # Valores por defecto
        item.setdefault("average_rating", 0.0)
        item.setdefault("tags", [])
        
        # Detalles básicos
        details = item.get("details", {})
        if not isinstance(details, dict):
            details = {}
        
        item["details"] = {
            "features": details.get("features", [])[:2],
            "specifications": details.get("specifications", {})
        }
        
        # 🔥 NUEVO: Añadir metadatos ML
        item["_loader_metadata"] = {
            "ml_enabled": self.ml_enabled,
            "ml_features": self.ml_features,
            "source_file": filename,
            "processing_timestamp": time.time()
        }
        
        return item

    def _get_category_from_filename_fast(self, filename: str) -> str:
        """Categoría rápida desde nombre de archivo"""
        stem = Path(filename).stem.lower()
        
        # Mapeo mínimo
        if 'game' in stem:
            return 'games'
        elif 'software' in stem:
            return 'software'
        elif 'electronic' in stem:
            return 'electronics'
        elif 'book' in stem:
            return 'books'
        else:
            return 'general'

    def _process_jsonl_file_fast(self, raw_file: Path) -> List[Product]:
        """Procesamiento rápido de JSONL con ML"""
        products = []
        line_count = 0
        
        try:
            with raw_file.open("r", encoding="utf-8", errors='ignore') as f:
                for line in f:
                    if line_count >= self.max_products_per_file:
                        break
                        
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            # Limpiar item
                            cleaned_item = self._clean_item_fast(item, raw_file.name)
                            
                            # 🔥 NUEVO: Crear producto con configuración ML
                            product = Product.from_dict(
                                cleaned_item,
                                ml_enrich=self.ml_enabled,
                                ml_features=self.ml_features if self.ml_enabled else None
                            )
                            
                            # Asignar categoría básica
                            product.product_type = cleaned_item["main_category"]
                            
                            product.clean_image_urls()
                            products.append(product)
                            line_count += 1
                            
                    except (json.JSONDecodeError, Exception) as e:
                        logger.debug(f"Skipping invalid line: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error processing JSONL file {raw_file.name}: {e}")
        
        return products

    def _process_json_file_fast(self, raw_file: Path) -> List[Product]:
        """Procesamiento rápido de JSON con ML"""
        try:
            with raw_file.open("r", encoding="utf-8", errors='ignore') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                data = [data]
            
            products = []
            for i, item in enumerate(data):
                if i >= self.max_products_per_file:
                    break
                    
                try:
                    if isinstance(item, dict):
                        # Limpiar item
                        cleaned_item = self._clean_item_fast(item, raw_file.name)
                        
                        # 🔥 NUEVO: Crear producto con configuración ML
                        product = Product.from_dict(
                            cleaned_item,
                            ml_enrich=self.ml_enabled,
                            ml_features=self.ml_features if self.ml_enabled else None
                        )
                        
                        # Asignar categoría básica
                        product.product_type = cleaned_item["main_category"]
                        
                        product.clean_image_urls()
                        products.append(product)
                        
                except Exception as e:
                    logger.debug(f"Skipping invalid item: {e}")
                    continue
            
            return products
            
        except Exception as e:
            logger.error(f"Error processing JSON file {raw_file.name}: {e}")
            return []

    def load_data(self, output_file: Union[str, Path] = None) -> List[Product]:
        """Carga rápida de datos con ML integrado"""
        if output_file is None:
            output_file = self.processed_dir / "products.json"

        start_time = time.time()
        logger.info("=== FAST DATA LOADING WITH ML ===")
        logger.info(f"ML Enabled: {self.ml_enabled}")
        logger.info(f"ML Features: {self.ml_features}")
        
        # 🔥 CONFIGURAR PRODUCTO ANTES DE CARGAR
        self._configure_product_ml()
        
        # Cargar archivos
        files = self._discover_data_files_fast()
        
        if not files:
            logger.warning("No product files found")
            return self._create_sample_data_fast(output_file)

        # Inicializar modelos ML si está habilitado
        if self.ml_enabled:
            self._initialize_ml_models()
        
        # Cargar productos para entrenamiento de categorías
        logger.info("Loading samples for training...")
        initial_products = []
        
        for file_path in files[:1]:
            try:
                products = self._load_single_file_fast(file_path)
                if products:
                    initial_products.extend(products[:15])
                    break
            except Exception as e:
                logger.warning(f"Error sampling from {file_path.name}: {e}")

        # Aprender categorías automáticamente
        if self.auto_categories and initial_products:
            logger.info(f"Training with {len(initial_products)} samples...")
            self._category_cache = self._auto_discover_categories_fast(initial_products)
            logger.info(f"Discovered {len(self._category_cache)} categories")
        else:
            self._category_cache = self._get_fallback_categories()
            logger.info("Using fallback categories")

        # 🔥 PROCESAR TODOS LOS ARCHIVOS CON ML
        logger.info("Processing all files...")
        all_products = []
        
        file_iterator = files
        if self.use_progress_bar:
            file_iterator = tqdm(files, desc="Files")
        
        for file_path in file_iterator:
            try:
                # 🔥 NUEVO: Usar batch processing para ML si está habilitado
                if self.ml_enabled and len(files) > 1:
                    # Cargar todos los productos del archivo primero
                    file_products = self._load_single_file_fast(file_path)
                    if file_products:
                        all_products.extend(file_products)
                else:
                    # Procesamiento normal
                    products = self._load_single_file_fast(file_path)
                    if products:
                        all_products.extend(products)
                        
            except Exception as e:
                logger.warning(f"Error processing {file_path.name}: {e}")

        if not all_products:
            logger.error("No products could be loaded")
            return self._create_sample_data_fast(output_file)

        # 🔥 APLICAR PROCESAMIENTO ML POR LOTES SI ESTÁ HABILITADO
        # 🔥 OPTIMIZACIÓN CRÍTICA: Procesar en lotes más pequeños
        if self.ml_enabled and len(all_products) > 1:
            logger.info(f"Applying ML batch processing to {len(all_products)} products...")
            try:
                # Procesar en lotes pequeños para no sobrecargar memoria
                batch_size = min(self.ml_batch_size, 16)  # Máximo 16 por batch
                processed_products = []
                
                for i in range(0, len(all_products), batch_size):
                    batch = all_products[i:i + batch_size]
                    logger.debug(f"Processing ML batch {i//batch_size + 1}/{(len(all_products)+batch_size-1)//batch_size}")
                    
                    # Convertir solo este batch
                    product_dicts = []
                    for p in batch:
                        try:
                            # Usar to_dict() si existe, si no model_dump()
                            if hasattr(p, 'to_dict'):
                                product_dicts.append(p.to_dict())
                            elif hasattr(p, 'model_dump'):
                                product_dicts.append(p.model_dump())
                            else:
                                # Extraer campos manualmente
                                product_dicts.append({
                                    'id': getattr(p, 'id', ''),
                                    'title': getattr(p, 'title', ''),
                                    'description': getattr(p, 'description', ''),
                                    'price': getattr(p, 'price', 0.0),
                                    'category': getattr(p, 'category', '')
                                })
                        except Exception as e:
                            logger.warning(f"Error converting product: {e}")
                            continue
                    
                    if product_dicts:
                        # Procesar este batch
                        batch_processed = Product.batch_create(
                            product_dicts,
                            ml_enrich=True,
                            batch_size=len(product_dicts)
                        )
                        processed_products.extend(batch_processed)
                
                all_products = processed_products
                logger.info(f"✅ ML processing completed: {len(all_products)} products")
                
            except Exception as e:
                logger.error(f"ML batch processing failed: {e}")
                logger.info("Falling back to no ML processing")
        elif self.auto_categories and self._category_cache:
            # Aplicar categorías aprendidas
            logger.info("Applying learned categories...")
            all_products = self._apply_categories_to_products(all_products)

        # Guardar productos
        self._save_products_fast(all_products, output_file)
        
        elapsed_time = time.time() - start_time
        logger.info(f"🚀 LOADING COMPLETED in {elapsed_time:.1f} seconds")
        logger.info(f"📦 Loaded {len(all_products)} products")
        logger.info(f"🤖 ML Processing: {'✅ Applied' if self.ml_enabled else '❌ Disabled'}")

        return all_products

    def _load_single_file_fast(self, raw_file: Path) -> List[Product]:
        """Carga rápida de archivo individual"""
        if not raw_file.exists():
            return []

        try:
            if raw_file.suffix.lower() == ".jsonl":
                return self._process_jsonl_file_fast(raw_file)
            else:
                return self._process_json_file_fast(raw_file)
        except Exception as e:
            logger.error(f"Error loading {raw_file.name}: {e}")
            return []

    # 🔥 MÉTODOS ML EXISTENTES (mantenidos del código anterior)
    def _extract_text_features_fast(self, products: List[Product]) -> List[str]:
        """Extracción rápida de características de texto"""
        texts = []
        for product in products:
            text_parts = [
                getattr(product, 'title', "") or "",
                getattr(product, 'description', "") or "",
            ]
            
            full_text = ' '.join(filter(None, text_parts))
            if full_text.strip():
                texts.append(full_text)
        
        return texts

    def _auto_discover_categories_fast(self, products: List[Product]) -> Dict[str, List[str]]:
        """Descubrimiento rápido de categorías"""
        if len(products) < 3:
            return self._get_fallback_categories()
        
        self._initialize_ml_models()
        if not self._models_initialized:
            return self._get_fallback_categories()
        
        try:
            texts = self._extract_text_features_fast(products)
            
            if len(texts) < 3:
                return self._get_fallback_categories()
            
            logger.info(f"Fast embedding generation for {len(texts)} products...")
            
            embeddings = self._ml_models['embedding'].encode(
                texts, 
                batch_size=4,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            n_clusters = min(
                AutoCategoryConfig.MAX_CATEGORIES,
                max(2, len(products) // 2)
            )
            
            logger.info(f"Fast clustering into {n_clusters} categories...")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=2, max_iter=20)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            category_keywords = self._extract_cluster_keywords_fast(texts, cluster_labels, n_clusters)
            
            logger.info(f"Discovered {len(category_keywords)} categories")
            return category_keywords
            
        except Exception as e:
            logger.error(f"Error in fast category discovery: {e}")
            return self._get_fallback_categories()

    def _extract_cluster_keywords_fast(self, texts: List[str], labels: np.ndarray, n_clusters: int) -> Dict[str, List[str]]:
        """Extracción rápida de keywords"""
        try:
            tfidf_matrix = self._ml_models['tfidf'].fit_transform(texts)
            feature_names = self._ml_models['tfidf'].get_feature_names_out()
            
            category_keywords = {}
            
            for cluster_id in range(n_clusters):
                cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
                
                if len(cluster_indices) < 1:
                    continue
                    
                cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
                top_keyword_indices = cluster_tfidf.argsort()[-2:][::-1]
                top_keywords = [
                    feature_names[i] for i in top_keyword_indices 
                    if cluster_tfidf[i] > 0
                ]
                
                if top_keywords:
                    category_name = self._generate_category_name_fast(top_keywords)
                    category_keywords[category_name] = top_keywords
            
            return category_keywords if category_keywords else self._get_fallback_categories()
            
        except Exception as e:
            logger.error(f"Error in fast keyword extraction: {e}")
            return self._get_fallback_categories()

    def _generate_category_name_fast(self, keywords: List[str]) -> str:
        """Generación rápida de nombre de categoría"""
        if not keywords:
            return "general"

        base_name = keywords[0].lower()
        name_mapping = {
            'prix': 'price',
            'description': 'general',
            'screen': 'display',
            'headset': 'audio'
        }

        mapped_name = name_mapping.get(base_name, base_name)
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', mapped_name)
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')
        clean_name = clean_name[:20]

        return clean_name or "general"

    def _apply_categories_to_products(self, products: List[Product]) -> List[Product]:
        """Aplica categorías aprendidas a los productos"""
        if not self._models_initialized or not self._category_cache:
            return products
        
        try:
            if len(products) > 10:
                texts = self._extract_text_features_fast(products)
                if texts:
                    embeddings = self._ml_models['embedding'].encode(
                        texts, 
                        batch_size=8,
                        show_progress_bar=False
                    )
                    
                    for i, product in enumerate(products):
                        if i < len(embeddings):
                            best_category = "general"
                            best_similarity = 0.3
                            
                            for category_name, keywords in self._category_cache.items():
                                category_text = ' '.join(keywords)
                                category_embedding = self._ml_models['embedding'].encode([category_text], show_progress_bar=False)
                                similarity = cosine_similarity(embeddings[i:i+1], category_embedding)[0][0]
                                
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_category = category_name
                            
                            product.product_type = best_category
        
        except Exception as e:
            logger.debug(f"Error applying categories: {e}")
        
        return products

    # 🔥 MÉTODOS EXISTENTES (sin cambios)
    def _discover_data_files_fast(self) -> List[Path]:
        """Descubrimiento rápido de archivos"""
        expected_patterns = ["*.jsonl", "*.json"]
        files = []
        
        for pattern in expected_patterns:
            files.extend(self.raw_dir.glob(pattern))
        
        valid_files = []
        for f in files:
            if f.exists() and f.stat().st_size > 100:
                valid_files.append(f)
        
        logger.info(f"Found {len(valid_files)} data files")
        return valid_files[:3]

    def _get_fallback_categories(self) -> Dict[str, List[str]]:
        """Categorías de fallback"""
        return {
            "electronics": ["electronic", "device"],
            "software": ["software", "app"],
            "games": ["game", "gaming"],
            "home": ["home", "household"],
            "books": ["book", "reading"],
            "general": ["product", "general"]
        }

    def _create_sample_data_fast(self, output_file: Path) -> List[Product]:
        """Datos de ejemplo"""
        logger.info("Creating sample data...")
        
        sample_products = [
            {
                "title": "Wireless Bluetooth Headphones",
                "description": "High-quality wireless headphones",
                "price": 99.99,
                "main_category": "electronics",
                "product_type": "electronics",
                "tags": ["wireless"],
                "details": {
                    "features": ["Noise cancellation"],
                    "specifications": {"color": "black"}
                }
            },
            {
                "title": "Python Programming Book",
                "description": "Learn Python programming",
                "price": 45.99,
                "main_category": "books", 
                "product_type": "books",
                "tags": ["education"],
                "details": {
                    "features": ["500 pages"],
                    "specifications": {"format": "PDF"}
                }
            }
        ]
        
        products = []
        for item in sample_products:
            try:
                product = Product.from_dict(item, ml_enrich=self.ml_enabled)
                products.append(product)
            except Exception as e:
                logger.warning(f"Error creating sample product: {e}")
        
        self._save_products_fast(products, output_file)
        logger.info(f"Created {len(products)} sample products")
        
        return products

    def _save_products_fast(self, products: List[Product], output_file: Path):
        """Guardado rápido de productos"""
        try:
            product_dicts = []
            for product in products:
                try:
                    if hasattr(product, 'model_dump'):
                        product_dict = product.model_dump()
                    else:
                        product_dict = product.__dict__.copy()
                    
                    product_dicts.append(product_dict)
                except Exception:
                    continue
            
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(product_dicts, f, ensure_ascii=False, separators=(',', ':'))
            
            logger.info(f"Saved {len(product_dicts)} products to {output_file}")
        except Exception as e:
            logger.error(f"Error saving to {output_file}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas del loader"""
        stats = {
            "total_products_loaded": self._get_total_products(),
            "auto_categories_enabled": self.auto_categories,
            "total_categories": len(self._category_cache) if self._category_cache else 0,
            "categories": list(self._category_cache.keys()) if self._category_cache else [],
            "ml_models_initialized": self._models_initialized,
            # 🔥 NUEVO: Estadísticas ML
            "ml_enabled": self.ml_enabled,
            "ml_features": self.ml_features,
            "ml_batch_size": self.ml_batch_size
        }
        
        # 🔥 MODIFICADO: No intentar acceder a MLProductEnricher si no está disponible
        # Agregar métricas ML si está disponible
        if self.ml_enabled:
            try:
                # Intentar importar MLProductEnricher dinámicamente solo cuando sea necesario
                from src.core.data.product import MLProductEnricher
                ml_metrics = MLProductEnricher.get_metrics()
                stats["ml_metrics"] = ml_metrics
            except ImportError as e:
                logger.debug(f"MLProductEnricher not available: {e}")
                stats["ml_metrics"] = {"ml_enabled": False, "error": "not_available"}
            except Exception as e:
                logger.debug(f"Error getting ML metrics: {e}")
                stats["ml_metrics"] = {"ml_enabled": True, "error": "metrics_unavailable"}
        
        return stats

    def _get_total_products(self) -> int:
        """Obtiene número total de productos"""
        output_file = self.processed_dir / "products.json"
        if output_file.exists():
            try:
                with output_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    return len(data) if isinstance(data, list) else 0
            except Exception:
                return 0
        return 0

    def print_detailed_stats(self):
        """Imprime estadísticas detalladas"""
        stats = self.get_stats()
        products = self._get_sample_products()
        
        print(f"\n{'='*50}")
        print(f"🚀 FAST DATA LOADER - COMPLETED")
        print(f"{'='*50}")
        print(f"📦 Total Products: {stats['total_products_loaded']}")
        print(f"🏷️  Categories Discovered: {stats['total_categories']}")
        print(f"🤖 ML Enabled: {'✅ Yes' if stats['ml_enabled'] else '❌ No'}")
        
        if stats['ml_enabled']:
            print(f"📊 ML Features: {', '.join(stats['ml_features'])}")
            print(f"🔧 ML Batch Size: {stats['ml_batch_size']}")
        
        if stats['categories']:
            print(f"📋 Categories: {', '.join(stats['categories'])}")
        
        if products:
            print(f"\n📋 SAMPLE PRODUCTS:")
            for i, product in enumerate(products[:3]):
                title = getattr(product, 'title', 'No title')
                product_type = getattr(product, 'product_type', 'Unknown')
                price = getattr(product, 'price', 0)
                ml_processed = getattr(product, 'ml_processed', False)
                
                ml_info = " (ML Processed)" if ml_processed else ""
                
                print(f"   {i+1}. {title}{ml_info}")
                print(f"      Type: {product_type}")
                print(f"      Price: ${price:.2f}")
                print()

    def _get_sample_products(self) -> List[Product]:
        """Obtiene muestra de productos"""
        output_file = self.processed_dir / "products.json"
        if output_file.exists():
            try:
                with output_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        sample_data = data[:3]
                        return [Product.from_dict(item) for item in sample_data]
            except Exception:
                pass
        return []


# Aliases para compatibilidad
DataLoader = FastDataLoader
AutomatedDataLoader = FastDataLoader

if __name__ == "__main__":
    logger.info("=== 🚀 FAST DATA LOADER ===")

    # Inicializar loader
    loader = FastDataLoader(
        raw_dir=settings.RAW_DIR,
        processed_dir=settings.PROC_DIR,
        auto_categories=True,
        auto_tags=False,
        max_products_per_file=500,
        cache_enabled=False,
        use_progress_bar=True,
        # 🔥 NUEVO: Configuración ML
        ml_enabled=getattr(settings, "ML_ENABLED", False),
        ml_features=getattr(settings, "ML_FEATURES", ["category", "entities"]),
        ml_batch_size=64
    )

    # Carga rápida
    products = loader.load_data()

    # Mostrar estadísticas
    loader.print_detailed_stats()

    logger.info("🎉 Loading completed successfully!")