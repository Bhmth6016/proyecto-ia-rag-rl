# src/core/data/loader.py

import json
import pickle
import re
import zlib
import time
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from threading import Lock
import hashlib
from functools import lru_cache
from itertools import islice

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
from collections import Counter, defaultdict

from tqdm import tqdm

from src.core.data.product import Product
from src.core.config import settings
from src.core.utils.logger import get_logger

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
                logger.warning(f"SpaCy model {AutoCategoryConfig.NER_MODEL_NAME} not found. Installing...")
                import os
                os.system(f"python -m spacy download {AutoCategoryConfig.NER_MODEL_NAME}")
                self._ml_models['nlp'] = spacy.load(AutoCategoryConfig.NER_MODEL_NAME)
            
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
            self._models_initialized = False

    def _get_optimal_workers(self) -> int:
        """Calcula el número óptimo de workers"""
        cpu_count = mp.cpu_count()
        return min(cpu_count - 1, 8)

    def _extract_text_features(self, products: List[Product]) -> List[str]:
        """Extrae características de texto de los productos"""
        texts = []
        for product in products:
            text_parts = [
                product.title or "",
                product.description or "",
                ' '.join(product.details.features) if product.details else "",
                ' '.join(f"{k} {v}" for k, v in product.details.specifications.items()) if product.details else ""
            ]
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

    def _auto_generate_tags(self, products: List[Product]) -> Dict[str, List[str]]:
        """Genera tags automáticamente usando análisis de texto"""
        if len(products) < self.min_samples_for_training:
            return self._get_fallback_tags()
        
        self._initialize_ml_models()
        
        try:
            texts = self._extract_text_features(products)
            
            # Extraer características comunes usando TF-IDF y patrones
            tag_keywords = self._extract_common_attributes(texts)
            
            # Enriquecer con NER
            tag_keywords.update(self._extract_ner_tags(texts))
            
            logger.info(f"Auto-generated {len(tag_keywords)} tag categories")
            return tag_keywords
            
        except Exception as e:
            logger.error(f"Error in auto tag generation: {e}")
            return self._get_fallback_tags()

    def _extract_common_attributes(self, texts: List[str]) -> Dict[str, List[str]]:
        """Extrae atributos comunes para tags"""
        # Patrones para diferentes tipos de tags
        patterns = {
            'wireless': [r'wireless', r'bluetooth', r'wi.fi', r'wifi'],
            'portable': [r'portable', r'lightweight', r'compact', r'foldable'],
            'waterproof': [r'waterproof', r'water.resistant', r'waterproofing'],
            'gaming': [r'gaming', r'gamer', r'rgb', r'gaming.grade'],
            'fast_charging': [r'fast.charging', r'quick.charge', r'rapid.charge'],
            'eco_friendly': [r'eco', r'recycled', r'biodegradable', r'environmental'],
            'digital': [r'digital', r'download', r'online', r'streaming'],
            'premium': [r'premium', r'luxury', r'exclusive', r'high.end']
        }
        
        tag_keywords = {}
        all_text = ' '.join(texts).lower()
        
        for tag_name, tag_patterns in patterns.items():
            matching_keywords = []
            for pattern in tag_patterns:
                matches = re.findall(pattern, all_text)
                if matches:
                    matching_keywords.extend(matches)
            
            if matching_keywords:
                # Usar las variantes encontradas como keywords
                tag_keywords[tag_name] = list(set(matching_keywords))[:3]
        
        return tag_keywords

    def _extract_ner_tags(self, texts: List[str]) -> Dict[str, List[str]]:
        """Extrae tags usando Named Entity Recognition"""
        tag_categories = {
            'brands': set(),
            'materials': set(),
            'colors': set(),
            'sizes': set()
        }
        
        # Mapeo de entidades de spaCy a nuestras categorías
        entity_mapping = {
            'ORG': 'brands',
            'PRODUCT': 'brands',
            'MATERIAL': 'materials',
            'COLOR': 'colors',
            'QUANTITY': 'sizes'
        }
        
        for text in texts[:100]:  # Procesar muestra para eficiencia
            doc = self._ml_models['nlp'](text)
            
            for ent in doc.ents:
                category = entity_mapping.get(ent.label_)
                if category and len(ent.text.strip()) > 1:
                    tag_categories[category].add(ent.text.lower().strip())
        
        # Convertir a formato de salida
        result = {}
        for category, items in tag_categories.items():
            if items:
                result[category] = list(items)[:5]  # Top 5 items
        
        return result

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
            text = ' '.join([
                product.title or "",
                product.description or "",
                ' '.join(product.details.features) if product.details else ""
            ])
            
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
            text = ' '.join([
                product.title or "",
                product.description or "",
                ' '.join(product.details.features) if product.details else ""
            ]).lower()
            
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
            return []

        # Cargar productos iniciales
        initial_products = self._load_initial_products(files)
        
        if not initial_products:
            logger.error("No valid products loaded from any files!")
            return []

        # Aprender categorías y tags automáticamente si está habilitado
        if self.auto_categories:
            logger.info("Auto-discovering categories...")
            self._category_cache = self._auto_discover_categories(initial_products)
            logger.info(f"Discovered categories: {list(self._category_cache.keys())}")
        
        if self.auto_tags:
            logger.info("Auto-generating tags...")
            self._tag_cache = self._auto_generate_tags(initial_products)
            logger.info(f"Generated tag categories: {list(self._tag_cache.keys())}")

        # Reprocesar productos con modelos aprendidos
        all_products = self._reprocess_with_models(files)
        
        if not all_products:
            logger.error("No valid products after reprocessing!")
            return initial_products  # Fallback a productos iniciales

        logger.info("Loaded %d products from %d files", len(all_products), len(files))

        # Guardar productos procesados
        self.save_standardized_json(all_products, output_file)

        return all_products

    def _discover_data_files(self) -> List[Path]:
        """Descubre automáticamente archivos de datos"""
        expected_patterns = ["*.jsonl", "*.json", "*.csv", "*.parquet"]
        files = []
        
        for pattern in expected_patterns:
            files.extend(self.raw_dir.glob(pattern))
            files.extend(self.raw_dir.glob(pattern.upper()))
        
        # Ordenar por tamaño (archivos más grandes primero para mejor sampling)
        files.sort(key=lambda x: x.stat().st_size if x.exists() else 0, reverse=True)
        
        logger.info(f"Discovered {len(files)} data files: {[f.name for f in files]}")
        return files[:10]  # Limitar a 10 archivos para evitar sobrecarga

    def _load_initial_products(self, files: List[Path]) -> List[Product]:
        """Carga productos iniciales para entrenamiento"""
        sample_products = []
        sample_size = min(1000, self.min_samples_for_training * 2)
        
        for file_path in files:
            if len(sample_products) >= sample_size:
                break
                
            try:
                products = self.load_single_file(file_path)
                sample_products.extend(products[:sample_size - len(sample_products)])
            except Exception as e:
                logger.warning(f"Error sampling from {file_path.name}: {e}")
        
        return sample_products

    def _reprocess_with_models(self, files: List[Path]) -> List[Product]:
        """Reprocesa todos los archivos con los modelos aprendidos"""
        all_products = []
        
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(files))) as exe:
            future_to_file = {exe.submit(self.load_single_file_automated, f): f for f in files}
            for future in tqdm(future_to_file, desc="Processing files", total=len(files), disable=self.disable_tqdm):
                try:
                    products = future.result()
                    if products:
                        all_products.extend(products)
                except Exception as e:
                    file_path = future_to_file[future]
                    logger.warning(f"Error processing {file_path.name}: {e}")

        return all_products

    def load_single_file_automated(self, raw_file: Union[str, Path]) -> List[Product]:
        """Carga un archivo con procesamiento automatizado"""
        raw_file = Path(raw_file)
        
        # Intentar cargar desde caché
        cached_products = self._load_from_cache(raw_file)
        if cached_products is not None:
            return cached_products

        logger.info("Processing file with automation: %s", raw_file.name)
        
        # Procesar archivo
        if raw_file.suffix.lower() == ".jsonl":
            products = self._process_jsonl_automated(raw_file)
        else:
            products = self._process_json_array_automated(raw_file)

        # Guardar en caché
        if products:
            self._save_to_cache(raw_file, products)

        return products

    def _process_jsonl_automated(self, raw_file: Path) -> List[Product]:
        """Procesa JSONL con automatización"""
        with raw_file.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Dividir en chunks
        chunk_size = max(len(lines) // (self.max_workers * 2), 1)
        chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
        
        products = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self._process_chunk_automated, chunk, raw_file.name): chunk 
                for chunk in chunks
            }
            
            for future in tqdm(
                as_completed(future_to_chunk), 
                total=len(chunks),
                desc=f"Processing {raw_file.name}",
                disable=self.disable_tqdm
            ):
                try:
                    chunk_products = future.result()
                    products.extend(chunk_products)
                except Exception as e:
                    logger.warning(f"Error processing chunk: {e}")
        
        return products

    def _process_chunk_automated(self, chunk: List[str], filename: str) -> List[Product]:
        """Procesa un chunk con automatización"""
        products = []
        for line in chunk:
            try:
                item = json.loads(line.strip())
                if isinstance(item, dict):
                    cleaned_item = self._clean_item_automated(item, filename)
                    product = Product.from_dict(cleaned_item)
                    product.clean_image_urls()
                    products.append(product)
            except Exception:
                continue
        return products

    # Métodos de cache y otros helpers (similares a la versión anterior)
    def _get_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_cache_file_path(self, raw_file: Path) -> Path:
        cache_filename = f"{raw_file.stem}_{self._get_file_hash(raw_file)[:8]}.pkl"
        return self.processed_dir / "cache" / cache_filename

    def _save_to_cache(self, raw_file: Path, products: List[Product]) -> None:
        if not self.cache_enabled:
            return
        try:
            cache_file = self._get_cache_file_path(raw_file)
            cache_dir = cache_file.parent
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            cache_data = {
                'products': products,
                'file_hash': self._get_file_hash(raw_file),
                'timestamp': time.time(),
                'version': '2.0'  # Nueva versión para datos automatizados
            }
            
            compressed_data = zlib.compress(pickle.dumps(cache_data))
            with open(cache_file, 'wb') as f:
                f.write(compressed_data)
                
        except Exception as e:
            logger.warning(f"Error saving cache for {raw_file.name}: {e}")

    def _load_from_cache(self, raw_file: Path) -> Optional[List[Product]]:
        if not self.cache_enabled:
            return None
        cache_file = self._get_cache_file_path(raw_file)
        
        if not cache_file.exists():
            return None
        
        try:
            if self.cache_ttl > 0:
                file_age = time.time() - cache_file.stat().st_mtime
                if file_age > self.cache_ttl:
                    cache_file.unlink()
                    return None
            
            with open(cache_file, 'rb') as f:
                compressed_data = f.read()
            
            cache_data = pickle.loads(zlib.decompress(compressed_data))
            
            current_hash = self._get_file_hash(raw_file)
            if (cache_data.get('file_hash') == current_hash and 
                cache_data.get('version') == '2.0'):
                logger.debug(f"Cache hit for {raw_file.name}")
                return cache_data['products']
                
        except Exception as e:
            logger.warning(f"Error loading cache for {raw_file.name}: {e}")
        
        return None

    def _load_global_cache(self, cache_file: Path) -> List[Product]:
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

    def save_standardized_json(self, products: List[Product], output_file: Union[str, Path]) -> None:
        output_file = Path(output_file)
        standardized_data = [product.model_dump() for product in products]
        
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(standardized_data, f, ensure_ascii=False, indent=2)
        logger.info("Saved standardized JSON to %s", output_file)

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

    loader = AutomatedDataLoader(
        raw_dir=settings.RAW_DIR,
        processed_dir=settings.PROC_DIR,
        auto_categories=True,
        auto_tags=True,
        cache_enabled=True
    )

    products = loader.load_data()

    logger.info(f"Finished. Loaded {len(products)} products.")
