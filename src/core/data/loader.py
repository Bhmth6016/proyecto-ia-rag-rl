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
logging.getLogger('sklearn').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)

# Importaciones simplificadas
try:
    from src.core.data.product import Product
    from src.core.config import settings
    from src.core.utils.logger import get_logger
except ImportError:
    # Fallback definitions
    import logging
    
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
        def from_dict(cls, data):
            return cls(**data)
        
        def clean_image_urls(self):
            if hasattr(self, 'image_urls'):
                if not self.image_urls:
                    self.image_urls = ["https://via.placeholder.com/300"]
        @property
        def product_id(self):
            for key in ["asin", "id", "productId", "product_type", "code"]:
                if getattr(self, key, None):
                    return getattr(self, key)

            return self.title 
    
    class settings:
        RAW_DIR = Path("./data/raw")
        PROC_DIR = Path("./data/processed")
    
    def get_logger(name):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(name)

logger = get_logger(__name__)

class AutoCategoryConfig:
    """Configuración optimizada para categorización automática"""
    SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
    MIN_CLUSTER_SIZE = 3
    MAX_CATEGORIES = 15

class FastDataLoader:
    """
    Cargador optimizado para velocidad máxima
    """

    def __init__(
        self,
        *,
        raw_dir: Optional[Union[str, Path]] = None,
        processed_dir: Optional[Union[str, Path]] = None,
        cache_enabled: bool = False,  # Deshabilitado para velocidad
        max_products_per_file: int = 5000,  # Muy limitado
        auto_categories: bool = True,
        auto_tags: bool = False,  # Deshabilitado para velocidad
        use_progress_bar: bool = True
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else settings.RAW_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else settings.PROC_DIR
        self.cache_enabled = cache_enabled
        self.max_products_per_file = max_products_per_file
        self.auto_categories = auto_categories
        self.auto_tags = auto_tags
        self.use_progress_bar = use_progress_bar
        
        self._ml_models = {}
        self._category_cache = {}
        self._models_initialized = False

        # Crear directorios
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_ml_models(self):
        """Inicialización ultra-rápida de modelos"""
        if self._models_initialized:
            return
            
        try:
            logger.info("Initializing optimized ML models...")
            
            # Configurar para máximo rendimiento
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            os.environ['OMP_NUM_THREADS'] = '1'
            
            # Modelo ligero de embeddings
            self._ml_models['embedding'] = SentenceTransformer(
                AutoCategoryConfig.SENTENCE_MODEL_NAME,
                device='cpu'
            )
            
            # Vectorizador minimalista
            self._ml_models['tfidf'] = TfidfVectorizer(
                max_features=50,  # Muy reducido
                stop_words='english',
                ngram_range=(1, 1)
            )
            
            self._models_initialized = True
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
            self._models_initialized = False

    def _extract_text_features_fast(self, products: List[Product]) -> List[str]:
        """Extracción ultra-rápida de características de texto"""
        texts = []
        for product in products:
            # Solo título y descripción básica
            text_parts = [
                getattr(product, 'title', "") or "",
                getattr(product, 'description', "") or "",
            ]
            
            full_text = ' '.join(filter(None, text_parts))
            if full_text.strip():
                texts.append(full_text)
        
        return texts

    def _auto_discover_categories_fast(self, products: List[Product]) -> Dict[str, List[str]]:
        """Descubrimiento ultra-rápido de categorías"""
        if len(products) < 3:
            return self._get_fallback_categories()
        
        self._initialize_ml_models()
        if not self._models_initialized:
            return self._get_fallback_categories()
        
        try:
            # Extraer textos mínimos
            texts = self._extract_text_features_fast(products)
            
            if len(texts) < 3:
                return self._get_fallback_categories()
            
            logger.info(f"Fast embedding generation for {len(texts)} products...")
            
            # Embeddings ultra-rápidos
            embeddings = self._ml_models['embedding'].encode(
                texts, 
                batch_size=4,  # Batch muy pequeño
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Clustering mínimo
            n_clusters = min(
                AutoCategoryConfig.MAX_CATEGORIES,
                max(2, len(products) // 2)  # Clusters muy reducidos
            )
            
            logger.info(f"Fast clustering into {n_clusters} categories...")
            
            # K-means ultra-rápido
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=2, max_iter=20)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Keywords mínimas
            category_keywords = self._extract_cluster_keywords_fast(texts, cluster_labels, n_clusters)
            
            logger.info(f"Discovered {len(category_keywords)} categories")
            return category_keywords
            
        except Exception as e:
            logger.error(f"Error in fast category discovery: {e}")
            return self._get_fallback_categories()

    def _extract_cluster_keywords_fast(self, texts: List[str], labels: np.ndarray, n_clusters: int) -> Dict[str, List[str]]:
        """Extracción ultra-rápida de keywords"""
        try:
            # TF-IDF mínimo
            tfidf_matrix = self._ml_models['tfidf'].fit_transform(texts)
            feature_names = self._ml_models['tfidf'].get_feature_names_out()
            
            category_keywords = {}
            
            for cluster_id in range(n_clusters):
                cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
                
                if len(cluster_indices) < 1:  # Mínimo absoluto
                    continue
                    
                # Scores mínimos
                cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
                
                # Top keywords mínimo
                top_keyword_indices = cluster_tfidf.argsort()[-2:][::-1]  # Solo top 2
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
        """Generación rápida y mejorada de nombre de categoría."""
        if not keywords:
            return "general"

        base_name = keywords[0].lower()

        # Mapeo mejorado de nombres para categorías más limpias
        name_mapping = {
            'prix': 'price',
            'description': 'general',
            'screen': 'display',
            'headset': 'audio'
        }

        # Aplicar el mapeo si existe
        mapped_name = name_mapping.get(base_name, base_name)

        # Limpiar caracteres no alfanuméricos
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', mapped_name)
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')

        # Limitar la longitud a 20 caracteres
        clean_name = clean_name[:20]

        return clean_name or "general"


    def _get_fallback_categories(self) -> Dict[str, List[str]]:
        """Categorías de fallback ultra-optimizadas"""
        return {
            "electronics": ["electronic", "device"],
            "software": ["software", "app"],
            "games": ["game", "gaming"],
            "home": ["home", "household"],
            "books": ["book", "reading"],
            "general": ["product", "general"]
        }

    def _clean_item_fast(self, item: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Limpieza ultra-rápida de items"""
        # Validación básica
        title = item.get("title", "").strip()
        if not title:
            raise ValueError("Missing title")
        
        # Limpieza mínima absoluta
        description = item.get("description", "")
        if not description:
            item["description"] = "No description"
        elif isinstance(description, list):
            item["description"] = " ".join(str(x) for x in description[:1])  # Solo 1 elemento
        
        # Categoría desde archivo (muy básica)
        item["main_category"] = self._get_category_from_filename_fast(filename)
        
        # Precio ultra-rápido
        price = item.get("price")
        if price is None:
            item["price"] = 0.0
        elif isinstance(price, str):
            cleaned_price = re.search(r'(\d+(?:[.,]\d{1,2})?)', price)
            item["price"] = float(cleaned_price.group(1).replace(',', '.')) if cleaned_price else 0.0
        
        # Valores por defecto mínimos
        item.setdefault("average_rating", 0.0)
        item.setdefault("tags", [])
        
        # Detalles ultra-básicos
        details = item.get("details", {})
        if not isinstance(details, dict):
            details = {}
        
        item["details"] = {
            "features": details.get("features", [])[:2],  # Solo 2 features
            "specifications": details.get("specifications", {})
        }
        
        return item

    def _get_category_from_filename_fast(self, filename: str) -> str:
        """Categoría ultra-rápida desde nombre de archivo"""
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
        """Procesamiento ultra-rápido de JSONL"""
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
                            cleaned_item = self._clean_item_fast(item, raw_file.name)
                            product = Product.from_dict(cleaned_item)
                            
                            # Asignar categoría básica
                            product.product_type = cleaned_item["main_category"]
                            
                            product.clean_image_urls()
                            products.append(product)
                            line_count += 1
                    except (json.JSONDecodeError, Exception):
                        continue
                        
        except Exception as e:
            logger.error(f"Error processing JSONL file {raw_file.name}: {e}")
        
        return products

    def _process_json_file_fast(self, raw_file: Path) -> List[Product]:
        """Procesamiento ultra-rápido de JSON"""
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
                        cleaned_item = self._clean_item_fast(item, raw_file.name)
                        product = Product.from_dict(cleaned_item)
                        
                        # Asignar categoría básica
                        product.product_type = cleaned_item["main_category"]
                        
                        product.clean_image_urls()
                        products.append(product)
                except Exception:
                    continue
            
            return products
            
        except Exception as e:
            logger.error(f"Error processing JSON file {raw_file.name}: {e}")
            return []

    def load_data(self, output_file: Union[str, Path] = None) -> List[Product]:
        """Carga ultra-rápida de datos - MÉTODO PRINCIPAL"""
        if output_file is None:
            output_file = self.processed_dir / "products.json"

        start_time = time.time()
        logger.info("=== ULTRA-FAST DATA LOADING ===")
        
        # Cargar archivos
        files = self._discover_data_files_fast()
        
        if not files:
            logger.warning("No product files found")
            return self._create_sample_data_fast(output_file)

        # Cargar productos para entrenamiento (muy limitado)
        logger.info("Loading minimal samples for training...")
        initial_products = []
        for file_path in files[:1]:  # Solo primer archivo
            try:
                products = self._load_single_file_fast(file_path)
                if products:
                    initial_products.extend(products[:15])  # Solo 15 productos
                    break  # Solo un archivo para entrenamiento
            except Exception as e:
                logger.warning(f"Error sampling from {file_path.name}: {e}")

        # Aprender categorías ultra-rápidamente
        if self.auto_categories and initial_products:
            logger.info(f"Ultra-fast training with {len(initial_products)} samples...")
            self._category_cache = self._auto_discover_categories_fast(initial_products)
            logger.info(f"Discovered {len(self._category_cache)} categories")
        else:
            self._category_cache = self._get_fallback_categories()
            logger.info("Using fallback categories")

        # Procesar todos los archivos ultra-rápidamente
        logger.info("Ultra-fast processing all files...")
        all_products = []
        
        # Usar progress bar solo si está habilitado
        file_iterator = files
        if self.use_progress_bar:
            file_iterator = tqdm(files, desc="Files")
        
        for file_path in file_iterator:
            try:
                products = self._load_single_file_fast(file_path)
                if products:
                    all_products.extend(products)
            except Exception as e:
                logger.warning(f"Error processing {file_path.name}: {e}")

        if not all_products:
            logger.error("No products could be loaded")
            return self._create_sample_data_fast(output_file)

        # Aplicar categorías aprendidas si están disponibles
        if self.auto_categories and self._category_cache:
            logger.info("Applying learned categories...")
            all_products = self._apply_categories_to_products(all_products)

        # Guardar ultra-rápidamente
        self._save_products_fast(all_products, output_file)
        
        elapsed_time = time.time() - start_time
        logger.info(f"🚀 ULTRA-FAST LOADING COMPLETED in {elapsed_time:.1f} seconds")
        logger.info(f"📦 Loaded {len(all_products)} products")

        return all_products

    def _apply_categories_to_products(self, products: List[Product]) -> List[Product]:
        """Aplica categorías aprendidas a los productos"""
        if not self._models_initialized or not self._category_cache:
            return products
        
        try:
            # Solo procesar si hay suficientes productos
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
                            # Encontrar categoría más similar
                            best_category = "general"
                            best_similarity = 0.3  # Threshold bajo
                            
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

    def _load_single_file_fast(self, raw_file: Path) -> List[Product]:
        """Carga ultra-rápida de archivo individual"""
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

    def _discover_data_files_fast(self) -> List[Path]:
        """Descubrimiento ultra-rápido de archivos"""
        expected_patterns = ["*.jsonl", "*.json"]
        files = []
        
        for pattern in expected_patterns:
            files.extend(self.raw_dir.glob(pattern))
        
        # Filtrar archivos válidos rápidamente
        valid_files = []
        for f in files:
            if f.exists() and f.stat().st_size > 100:  # Mínimo 100 bytes
                valid_files.append(f)
        
        logger.info(f"Found {len(valid_files)} data files")
        return valid_files[:3]  # Máximo 3 archivos

    def _create_sample_data_fast(self, output_file: Path) -> List[Product]:
        """Datos de ejemplo ultra-rápidos"""
        logger.info("Creating ultra-fast sample data...")
        
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
                product = Product.from_dict(item)
                products.append(product)
            except Exception as e:
                logger.warning(f"Error creating sample product: {e}")
        
        self._save_products_fast(products, output_file)
        logger.info(f"Created {len(products)} sample products")
        
        return products

    def _save_products_fast(self, products: List[Product], output_file: Path):
        """Guardado ultra-rápido de productos"""
        try:
            # Convertir a diccionarios básicos
            product_dicts = []
            for product in products:
                try:
                    # Usar model_dump si está disponible, sino __dict__
                    if hasattr(product, 'model_dump'):
                        product_dict = product.model_dump()
                    else:
                        product_dict = product.__dict__.copy()
                    
                    product_dicts.append(product_dict)
                except Exception:
                    continue
            
            # Guardar sin pretty-printing para máxima velocidad
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(product_dicts, f, ensure_ascii=False, separators=(',', ':'))
            
            logger.info(f"Saved {len(product_dicts)} products to {output_file}")
        except Exception as e:
            logger.error(f"Error saving to {output_file}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Estadísticas ultra-rápidas"""
        return {
            "total_products_loaded": self._get_total_products(),
            "auto_categories_enabled": self.auto_categories,
            "total_categories": len(self._category_cache) if self._category_cache else 0,
            "categories": list(self._category_cache.keys()) if self._category_cache else [],
            "ml_models_initialized": self._models_initialized
        }

    def _get_total_products(self) -> int:
        """Obtiene número total de productos del archivo de salida"""
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
        print(f"🚀 ULTRA-FAST DATA LOADER - COMPLETED")
        print(f"{'='*50}")
        print(f"📦 Total Products: {stats['total_products_loaded']}")
        print(f"🏷️  Categories Discovered: {stats['total_categories']}")
        print(f"🤖 ML Models: {'✅ Ready' if stats['ml_models_initialized'] else '❌ Not ready'}")
        
        if stats['categories']:
            print(f"📊 Categories: {', '.join(stats['categories'])}")
        
        if products:
            print(f"\n📋 SAMPLE PRODUCTS:")
            for i, product in enumerate(products[:3]):
                print(f"   {i+1}. {getattr(product, 'title', 'No title')}")
                print(f"      Type: {getattr(product, 'product_type', 'Unknown')}")
                print(f"      Price: ${getattr(product, 'price', 0):.2f}")
                print()

    def _get_sample_products(self) -> List[Product]:
        """Obtiene muestra de productos para mostrar"""
        output_file = self.processed_dir / "products.json"
        if output_file.exists():
            try:
                with output_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        sample_data = data[:3]  # Solo 3 productos
                        return [Product.from_dict(item) for item in sample_data]
            except Exception:
                pass
        return []


# Aliases para compatibilidad
DataLoader = FastDataLoader
AutomatedDataLoader = FastDataLoader

if __name__ == "__main__":
    logger.info("=== 🚀 ULTRA-FAST DATA LOADER ===")

    # Inicializar loader ultra-rápido
    loader = FastDataLoader(
        raw_dir=settings.RAW_DIR,
        processed_dir=settings.PROC_DIR,
        auto_categories=True,
        auto_tags=False,  # Deshabilitado para máxima velocidad
        max_products_per_file=5000,  # Muy limitado
        cache_enabled=False,  # Sin cache
        use_progress_bar=True
    )

    # Carga ultra-rápida
    products = loader.load_data()

    # Mostrar estadísticas detalladas
    loader.print_detailed_stats()

    logger.info("🎉 Ultra-fast loading completed successfully!")