# src/core/data/loader.py

import json
import pickle
import re
import zlib
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from threading import Lock
import hashlib
from functools import lru_cache
from itertools import islice

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
from collections import Counter, defaultdict

from tqdm import tqdm

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
    
    # Categorías base que se expandirán automáticamente
    BASE_CATEGORIES = {
        "electronics", "software", "games", "home", "sports", 
        "beauty", "books", "clothing", "automotive", "tools"
    }
    
    # Modelos pre-entrenados
    SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
    NER_MODEL_NAME = "en_core_web_sm"
    
    # Parámetros de clustering
    MIN_CLUSTER_SIZE = 10
    MAX_CATEGORIES = 50

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
        max_workers: Optional[int] = None,
        disable_tqdm: bool = False,
        batch_size: int = 1000,
        cache_ttl: int = 3600,
        auto_categories: bool = True,
        auto_tags: bool = True,
        min_samples_for_training: int = 100
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else settings.RAW_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else settings.PROC_DIR
        self.cache_enabled = cache_enabled
        self.max_workers = max_workers or self._get_optimal_workers()
        self.disable_tqdm = disable_tqdm
        self.batch_size = batch_size
        self.cache_ttl = cache_ttl
        self.auto_categories = auto_categories
        self.auto_tags = auto_tags
        self.min_samples_for_training = min_samples_for_training
        
        self._error_lock = Lock()
        self._ml_models = {}
        self._category_cache = {}
        self._tag_cache = {}

        # Inicializar modelos ML bajo demanda
        self._models_initialized = False

        # Crear directorios si no existen
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_ml_models(self):
        """Inicializa modelos ML bajo demanda"""
        if self._models_initialized:
            return
            
        try:
            logger.info("Initializing ML models for automated processing...")
            
            # Modelo de embeddings para similitud semántica
            self._ml_models['embedding'] = SentenceTransformer(
                AutoCategoryConfig.SENTENCE_MODEL_NAME
            )
            
            # Modelo de NLP para NER
            try:
                self._ml_models['nlp'] = spacy.load(AutoCategoryConfig.NER_MODEL_NAME)
            except OSError:
                logger.warning(f"SpaCy model {AutoCategoryConfig.NER_MODEL_NAME} not found. Using minimal processing.")
                # Fallback sin spaCy
                self._ml_models['nlp'] = None
            
            # Vectorizador TF-IDF
            self._ml_models['tfidf'] = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            self._models_initialized = True
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
            # Fallback sin modelos ML
            self._models_initialized = False

    def _get_optimal_workers(self) -> int:
        """Calcula el número óptimo de workers"""
        import os
        cpu_count = os.cpu_count() or 1
        return min(cpu_count, 4)  # Reducido para evitar problemas de memoria

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
                    text_parts.append(' '.join(details.features))
                if hasattr(details, 'specifications') and details.specifications:
                    if isinstance(details.specifications, dict):
                        text_parts.append(' '.join(f"{k} {v}" for k, v in details.specifications.items()))
            
            texts.append(' '.join(filter(None, text_parts)))
        return texts

    def _auto_discover_categories(self, products: List[Product]) -> Dict[str, List[str]]:
        """Descubre categorías automáticamente usando clustering"""
        if len(products) < self.min_samples_for_training:
            logger.warning(f"Not enough products ({len(products)}) for auto-categorization")
            return self._get_fallback_categories()
        
        self._initialize_ml_models()
        
        try:
            # Extraer textos
            texts = self._extract_text_features(products)
            
            # Generar embeddings
            embeddings = self._ml_models['embedding'].encode(texts)
            
            # Determinar número óptimo de clusters
            n_clusters = min(
                AutoCategoryConfig.MAX_CATEGORIES,
                max(2, len(products) // AutoCategoryConfig.MIN_CLUSTER_SIZE)
            )
            
            # Clustering con K-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Extraer keywords por cluster usando TF-IDF
            category_keywords = self._extract_cluster_keywords(texts, cluster_labels, n_clusters)
            
            logger.info(f"Auto-discovered {len(category_keywords)} categories")
            return category_keywords
            
        except Exception as e:
            logger.error(f"Error in auto category discovery: {e}")
            return self._get_fallback_categories()

    def _extract_cluster_keywords(self, texts: List[str], labels: np.ndarray, n_clusters: int) -> Dict[str, List[str]]:
        """Extrae palabras clave representativas para cada cluster"""
        # Entrenar TF-IDF en todos los textos
        tfidf_matrix = self._ml_models['tfidf'].fit_transform(texts)
        feature_names = self._ml_models['tfidf'].get_feature_names_out()
        
        category_keywords = {}
        
        for cluster_id in range(n_clusters):
            # Obtener textos del cluster
            cluster_texts = [texts[i] for i in range(len(texts)) if labels[i] == cluster_id]
            
            if not cluster_texts:
                continue
                
            # Calcular scores TF-IDF promedio para el cluster
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
            cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
            
            # Obtener top keywords
            top_keyword_indices = cluster_tfidf.argsort()[-10:][::-1]
            top_keywords = [feature_names[i] for i in top_keyword_indices if cluster_tfidf[i] > 0]
            
            # Nombrar categoría basado en la keyword más representativa
            category_name = self._generate_category_name(top_keywords, cluster_texts)
            category_keywords[category_name] = top_keywords[:5]  # Top 5 keywords
            
        return category_keywords

    def _generate_category_name(self, keywords: List[str], cluster_texts: List[str]) -> str:
        """Genera nombre de categoría automáticamente"""
        # Usar la keyword más frecuente como base
        if keywords:
            base_name = keywords[0]
        else:
            base_name = "general"
        
        # Refinar usando NER para encontrar entidades comunes
        try:
            if self._ml_models.get('nlp'):
                entities = []
                for text in cluster_texts[:10]:  # Muestra de textos
                    doc = self._ml_models['nlp'](text)
                    entities.extend([ent.text.lower() for ent in doc.ents if ent.label_ in ['PRODUCT', 'ORG', 'GPE']])
                
                if entities:
                    most_common_entity = Counter(entities).most_common(1)[0][0]
                    base_name = most_common_entity
        except Exception as e:
            logger.debug(f"Error in NER for category naming: {e}")
        
        return base_name.replace(' ', '_').lower()

    def _get_fallback_categories(self) -> Dict[str, List[str]]:
        """Categorías de fallback cuando no hay suficientes datos"""
        return {
            "electronics": ["electronic", "device", "tech", "digital", "gadget"],
            "software": ["software", "app", "application", "program", "digital"],
            "home": ["home", "household", "domestic", "kitchen", "living"],
            "sports": ["sports", "fitness", "outdoor", "exercise", "athletic"],
            "general": ["product", "item", "goods", "merchandise", "general"]
        }

    def _get_fallback_tags(self) -> Dict[str, List[str]]:
        """Tags de fallback cuando no hay suficientes datos"""
        return {
            "wireless": ["wireless", "bluetooth"],
            "portable": ["portable", "lightweight"],
            "digital": ["digital", "download"]
        }

    def _infer_category_automated(self, product: Product) -> str:
        """Infere categoría automáticamente para un producto"""
        if not self.auto_categories or not self._category_cache:
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
                    text_parts.append(' '.join(details.features))
            
            text = ' '.join(filter(None, text_parts))
            
            if not text.strip():
                return "unknown"
            
            # Calcular similitud con cada categoría
            category_scores = {}
            text_embedding = self._ml_models['embedding'].encode([text])
            
            for category_name, keywords in self._category_cache.items():
                # Crear embedding de la categoría a partir de sus keywords
                category_text = ' '.join(keywords)
                category_embedding = self._ml_models['embedding'].encode([category_text])
                
                # Calcular similitud del coseno
                similarity = cosine_similarity(text_embedding, category_embedding)[0][0]
                category_scores[category_name] = similarity
            
            # Devolver categoría con mayor similitud
            best_category = max(category_scores.items(), key=lambda x: x[1])
            
            # Solo asignar si la similitud es suficientemente alta
            if best_category[1] > 0.3:
                return best_category[0]
            else:
                return "unknown"
                
        except Exception as e:
            logger.debug(f"Error in automated category inference: {e}")
            return "unknown"

    def _extract_tags_automated(self, product: Product) -> List[str]:
        """Extrae tags automáticamente para un producto"""
        if not self.auto_tags or not self._tag_cache:
            return []
        
        try:
            text_parts = [
                getattr(product, 'title', "") or "",
                getattr(product, 'description', "") or "",
            ]
            
            details = getattr(product, 'details', None)
            if details:
                if hasattr(details, 'features') and details.features:
                    text_parts.append(' '.join(details.features))
            
            text = ' '.join(filter(None, text_parts)).lower()
            
            tags = []
            
            # Buscar coincidencias con patrones de tags
            for tag_name, keywords in self._tag_cache.items():
                for keyword in keywords:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                        tags.append(tag_name)
                        break  # Una coincidencia por tag es suficiente
            
            return list(set(tags))  # Remover duplicados
            
        except Exception as e:
            logger.debug(f"Error in automated tag extraction: {e}")
            return []

    def _clean_item_automated(self, item: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Limpieza automatizada con ML"""
        # Validación básica
        title = item.get("title", "").strip()
        if not title:
            raise ValueError("Missing title")
        
        # Limpieza de descripción
        description = item.get("description")
        if not description:
            item["description"] = "No description available"
        elif isinstance(description, list):
            item["description"] = " ".join(str(x) for x in description if x)
        
        # Categoría desde archivo (fallback)
        item["main_category"] = self._get_category_from_filename(filename)
        
        # Limpieza de precio
        price = item.get("price")
        if price is None:
            item["price"] = "Price not available"
        elif isinstance(price, str):
            cleaned_price = re.search(r'\d+\.?\d*', price)
            item["price"] = float(cleaned_price.group()) if cleaned_price else "Price not available"
        
        # Valores por defecto
        item.setdefault("average_rating", "No rating available")
        
        # Detalles
        details = item.get("details", {})
        if not isinstance(details, dict):
            details = {}
        
        item["details"] = {
            "features": details.get("features", []),
            "specifications": details.get("specifications", {})
        }
        
        # Crear producto temporal para inferencia automática
        temp_product = Product.from_dict(item)
        
        # Inferir categoría automáticamente si está habilitado
        if self.auto_categories and not item.get("product_type"):
            auto_category = self._infer_category_automated(temp_product)
            if auto_category != "unknown":
                item["product_type"] = auto_category
        
        # Extraer tags automáticamente si está habilitado
        if self.auto_tags and not item.get("tags"):
            auto_tags = self._extract_tags_automated(temp_product)
            if auto_tags:
                item["tags"] = auto_tags
        
        return item

    def _get_category_from_filename(self, filename: str) -> str:
        """Obtiene categoría basada en el nombre del archivo (fallback)"""
        stem = Path(filename).stem
        # Inferir del nombre del archivo usando heurísticas simples
        if 'game' in stem.lower():
            return 'video_games'
        elif 'software' in stem.lower():
            return 'software'
        elif 'industrial' in stem.lower() or 'scientific' in stem.lower():
            return 'industrial'
        elif 'electronic' in stem.lower():
            return 'electronics'
        elif 'beauty' in stem.lower() or 'personal' in stem.lower():
            return 'beauty'
        else:
            return stem.lower()

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
                        product.clean_image_urls()
                        products.append(product)
                except Exception as e:
                    logger.debug(f"Error processing item: {e}")
                    continue
            
            return products
            
        except Exception as e:
            logger.error(f"Error processing JSON file {raw_file.name}: {e}")
            return []

    def load_data(self, use_cache: Optional[bool] = None, output_file: Union[str, Path] = None) -> List[Product]:
        """Carga datos con procesamiento automatizado"""
        if use_cache is None:
            use_cache = self.cache_enabled

        if output_file is None:
            output_file = self.processed_dir / "products.json"

        # Verificar caché global
        if use_cache and Path(output_file).exists():
            logger.info("Loading products from cache")
            return self._load_global_cache(output_file)

        # Cargar archivos
        files = self._discover_data_files()
        
        if not files:
            logger.warning("No product files found in %s", self.raw_dir)
            # Crear datos de ejemplo si no hay archivos
            return self._create_sample_data(output_file)

        # Cargar productos iniciales para entrenamiento
        initial_products = []
        for file_path in files[:2]:  # Solo usar 2 archivos para sampling
            try:
                products = self._load_single_file_simple(file_path)
                initial_products.extend(products[:100])  # Limitar a 100 productos por archivo
                if len(initial_products) >= 200:  # Máximo 200 productos para entrenamiento
                    break
            except Exception as e:
                logger.warning(f"Error sampling from {file_path.name}: {e}")
        
        if not initial_products:
            logger.error("No valid products loaded from any files!")
            return self._create_sample_data(output_file)

        # Aprender categorías automáticamente si está habilitado
        if self.auto_categories and len(initial_products) >= 10:
            logger.info("Auto-discovering categories...")
            self._category_cache = self._auto_discover_categories(initial_products)
            logger.info(f"Discovered categories: {list(self._category_cache.keys())}")
        else:
            self._category_cache = self._get_fallback_categories()

        # Usar tags de fallback por ahora (simplificado)
        self._tag_cache = self._get_fallback_tags()

        # Procesar todos los archivos
        all_products = []
        for file_path in tqdm(files, desc="Processing files", disable=self.disable_tqdm):
            try:
                products = self._load_single_file_simple(file_path)
                if products:
                    all_products.extend(products)
            except Exception as e:
                logger.warning(f"Error processing {file_path.name}: {e}")

        if not all_products:
            logger.error("No valid products after processing!")
            all_products = initial_products

        logger.info("Loaded %d products from %d files", len(all_products), len(files))

        # Guardar productos procesados
        self.save_standardized_json(all_products, output_file)

        return all_products

    def _load_single_file_simple(self, raw_file: Path) -> List[Product]:
        """Carga un archivo de manera simple y secuencial"""
        logger.info("Processing file: %s", raw_file.name)
        
        if not raw_file.exists():
            logger.warning(f"File {raw_file} does not exist")
            return []

        products = []
        
        try:
            if raw_file.suffix.lower() == ".jsonl":
                # Procesar JSONL
                with raw_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            if isinstance(item, dict):
                                cleaned_item = self._clean_item_automated(item, raw_file.name)
                                product = Product.from_dict(cleaned_item)
                                product.clean_image_urls()
                                products.append(product)
                        except Exception as e:
                            continue
            else:
                # Procesar JSON array
                products = self._process_json_array_automated(raw_file)
                
        except Exception as e:
            logger.error(f"Error processing file {raw_file.name}: {e}")
        
        logger.info("Loaded %d products from %s", len(products), raw_file.name)
        return products

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
                "details": {
                    "features": ["Noise cancellation", "30h battery", "Bluetooth 5.0"],
                    "specifications": {"color": "black", "weight": "250g"}
                }
            },
            {
                "title": "Programming Book: Python Mastery",
                "description": "Complete guide to Python programming",
                "price": 45.99,
                "main_category": "books", 
                "product_type": "books",
                "tags": ["digital", "education"],
                "details": {
                    "features": ["500 pages", "Digital download available"],
                    "specifications": {"format": "PDF", "language": "English"}
                }
            }
        ]
        
        products = []
        for item in sample_products:
            try:
                product = Product.from_dict(item)
                product.clean_image_urls()
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
        
        # Filtrar solo archivos que existen
        files = [f for f in files if f.exists()]
        
        # Ordenar por tamaño
        files.sort(key=lambda x: x.stat().st_size, reverse=True)
        
        logger.info(f"Discovered {len(files)} data files: {[f.name for f in files]}")
        return files

    def _load_global_cache(self, cache_file: Path) -> List[Product]:
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)

            valid_products = []
            for item in cached_data:
                try:
                    if "details" not in item or item["details"] is None:
                        item["details"] = {}
                    elif not isinstance(item["details"], dict):
                        item["details"] = {}

                    if not item.get("title", "").strip():
                        continue

                    product = Product.from_dict(item)
                    valid_products.append(product)
                except Exception as e:
                    logger.warning(f"Error loading product from cache: {e}")
                    continue

            return valid_products
        except Exception as e:
            logger.error(f"Error loading global cache: {e}")
            return []

    def save_standardized_json(self, products: List[Product], output_file: Union[str, Path]) -> None:
        output_file = Path(output_file)
        try:
            # Convertir productos a diccionarios
            standardized_data = []
            for product in products:
                try:
                    product_dict = product.__dict__.copy()
                    
                    # Asegurar que details sea serializable
                    if 'details' in product_dict and hasattr(product_dict['details'], '__dict__'):
                        product_dict['details'] = product_dict['details'].__dict__
                    
                    standardized_data.append(product_dict)
                except Exception as e:
                    logger.warning(f"Error serializing product: {e}")
                    continue
            
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(standardized_data, f, ensure_ascii=False, indent=2)
            logger.info("Saved standardized JSON to %s", output_file)
        except Exception as e:
            logger.error(f"Error saving JSON to {output_file}: {e}")

    def get_automation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de la automatización"""
        return {
            "auto_categories_enabled": self.auto_categories,
            "auto_tags_enabled": self.auto_tags,
            "discovered_categories": list(self._category_cache.keys()) if self._category_cache else [],
            "generated_tag_categories": list(self._tag_cache.keys()) if self._tag_cache else [],
            "ml_models_initialized": self._models_initialized
        }


# Alias para compatibilidad
DataLoader = AutomatedDataLoader

if __name__ == "__main__":
    logger.info("=== Running Automated Data Loader ===")

    # Crear directorios necesarios
    settings.RAW_DIR.mkdir(parents=True, exist_ok=True)
    settings.PROC_DIR.mkdir(parents=True, exist_ok=True)

    loader = AutomatedDataLoader(
        raw_dir=settings.RAW_DIR,
        processed_dir=settings.PROC_DIR,
        auto_categories=True,
        auto_tags=True,
        cache_enabled=True,
        max_workers=2  # Reducido para evitar problemas
    )

    products = loader.load_data()

    stats = loader.get_automation_stats()
    logger.info(f"Automation Stats: {stats}")

    logger.info(f"Finished. Loaded {len(products)} products.")