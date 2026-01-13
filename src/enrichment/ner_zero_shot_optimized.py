# src/enrichment/ner_zero_shot_optimized.py
"""
NER Zero-Shot OPTIMIZADO - Versión VERIFICADA con tu código
"""
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from tqdm import tqdm
import hashlib

logger = logging.getLogger(__name__)

class OptimizedNERExtractor:
    """
    Extractor NER optimizado con fallback a keywords
    Compatible con tu sistema existente
    """
    
    def __init__(self, use_zero_shot: bool = True, model_name: str = "facebook/bart-large-mnli"):
        self.use_zero_shot = use_zero_shot
        self.classifier = None
        self._init_error = None
        
        if use_zero_shot:
            self._initialize_zero_shot(model_name)
        
        # Keywords curadas específicamente para tu dataset Amazon
        self.keyword_attributes = self._load_keyword_templates()
        
        logger.info(f"✅ NER Extractor inicializado (zero-shot: {use_zero_shot})")
    
    def _initialize_zero_shot(self, model_name: str):
        """Inicializa zero-shot con manejo de errores"""
        try:
            import torch
            from transformers import pipeline
            
            # Verificar si hay GPU
            has_gpu = torch.cuda.is_available()
            device = 0 if has_gpu else -1
            
            if not has_gpu:
                logger.warning("⚠️  Sin GPU disponible, NER zero-shot será lento")
                logger.info("   Considera usar use_zero_shot=False para procesamiento rápido")
            
            # Cargar modelo con configuración segura
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=device,
                batch_size=4 if has_gpu else 2,  # Batch pequeño para evitar OOM
                framework="pt"
            )
            
            logger.info(f"   Modelo: {model_name}")
            logger.info(f"   Dispositivo: {'GPU' if has_gpu else 'CPU'}")
            
        except ImportError as e:
            self._init_error = f"Transformers no instalado: {e}"
            logger.error(f"❌ {self._init_error}")
            logger.info("   Instalar: pip install transformers torch")
            self.use_zero_shot = False
        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"⚠️  Error cargando zero-shot: {e}")
            logger.info("   Usando fallback a keywords")
            self.use_zero_shot = False
    
    def _load_keyword_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Carga templates de keywords optimizados para Amazon"""
        return {
            'Video Games': {
                'platform': ['PC', 'PlayStation', 'Xbox', 'Nintendo', 'Switch', 'mobile', 'console'],
                'genre': ['action', 'adventure', 'RPG', 'strategy', 'sports', 'racing', 
                         'survival', 'shooter', 'simulation', 'puzzle', 'fighting'],
                'features': ['multiplayer', 'single-player', 'online', 'co-op', 'competitive', 'offline']
            },
            'Electronics': {
                'type': ['headphones', 'speaker', 'monitor', 'keyboard', 'mouse', 'camera',
                        'phone', 'tablet', 'laptop', 'processor', 'memory', 'storage'],
                'connectivity': ['wireless', 'bluetooth', 'USB', 'HDMI', 'WiFi', 'ethernet', 'NFC'],
                'power': ['battery', 'rechargeable', 'solar', 'AC', 'DC', 'charger']
            },
            'Automotive': {
                'part': ['tire', 'battery', 'brake', 'filter', 'light', 'mirror', 'seat',
                        'engine', 'transmission', 'exhaust', 'suspension', 'wheel'],
                'type': ['replacement', 'accessory', 'tool', 'upgrade', 'repair', 'maintenance'],
                'material': ['metal', 'plastic', 'rubber', 'ceramic', 'composite', 'aluminum']
            },
            'Books': {
                'genre': ['fiction', 'non-fiction', 'biography', 'educational', 'children',
                         'romance', 'mystery', 'fantasy', 'science fiction', 'history'],
                'format': ['paperback', 'hardcover', 'audiobook', 'ebook', 'kindle', 'digital'],
                'topic': ['science', 'history', 'technology', 'business', 'self-help', 'cooking']
            },
            'Beauty': {
                'type': ['makeup', 'skincare', 'hair', 'perfume', 'nail', 'brush', 'cosmetic'],
                'skin_type': ['oily', 'dry', 'sensitive', 'combination', 'normal', 'all types'],
                'features': ['waterproof', 'long-lasting', 'natural', 'organic', 'vegan', 'cruelty-free']
            },
            'Clothing': {
                'type': ['shirt', 'pants', 'dress', 'shoes', 'jacket', 'hat', 'accessory'],
                'material': ['cotton', 'polyester', 'wool', 'silk', 'leather', 'denim'],
                'style': ['casual', 'formal', 'sports', 'outdoor', 'fashion', 'vintage']
            }
        }
    
    def extract_attributes(self, title: str, category: str = '') -> Dict[str, List[str]]:
        """
        Extrae atributos con fallback automático
        """
        if not title:
            return {}
        
        # Limpiar título
        clean_title = str(title).strip()
        if len(clean_title) < 3:
            return {}
        
        # Determinar template basado en categoría
        template_key = self._map_category_to_template(category)
        
        # Intentar zero-shot primero si está disponible
        if self.use_zero_shot and self.classifier:
            try:
                return self._extract_with_zero_shot(clean_title, template_key)
            except Exception as e:
                logger.debug(f"Zero-shot falló: {e}")
        
        # Fallback a keywords
        return self._extract_with_keywords(clean_title, template_key)
    
    def _map_category_to_template(self, category: str) -> str:
        """Mapea categoría de Amazon a template"""
        if not category:
            return 'General'
        
        category_lower = str(category).lower()
        
        # Mapeo explícito para tus categorías de Amazon
        mapping = {
            'video games': 'Video Games',
            'electronics': 'Electronics',
            'computers': 'Electronics',
            'cell phones': 'Electronics',
            'automotive': 'Automotive',
            'books': 'Books',
            'beauty': 'Beauty',
            'clothing': 'Clothing',
            'sports': 'Clothing',
            'home': 'Home',
            'kitchen': 'Home',
            'toys': 'Toys'
        }
        
        for key, template in mapping.items():
            if key in category_lower:
                return template
        
        return 'General'
    
    def _extract_with_zero_shot(self, title: str, template_key: str) -> Dict[str, List[str]]:
        """Extrae usando zero-shot classification"""
        templates = self.keyword_attributes.get(template_key, {})
        
        extracted = {}
        
        for attr_type, candidates in templates.items():
            try:
                # Limitar número de candidatos para evitar OOM
                limited_candidates = candidates[:15]
                
                result = self.classifier(
                    title,
                    candidate_labels=limited_candidates,
                    multi_label=True,
                    hypothesis_template="This is related to {}."
                )
                
                # Umbral más alto para mayor precisión
                selected = [
                    label for label, score in zip(result['labels'], result['scores'])
                    if score > 0.7  # 70% de confianza
                ]
                
                if selected:
                    extracted[attr_type] = selected[:2]  # Top 2
            
            except Exception as e:
                logger.debug(f"Error en {attr_type}: {e}")
                continue
        
        return extracted
    
    def _extract_with_keywords(self, title: str, template_key: str) -> Dict[str, List[str]]:
        """Extrae usando keyword matching simple y rápido"""
        title_lower = title.lower()
        templates = self.keyword_attributes.get(template_key, {})
        
        extracted = {}
        
        for attr_type, keywords in templates.items():
            found = []
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # Búsqueda exacta de palabra
                if f' {keyword_lower} ' in f' {title_lower} ':
                    found.append(keyword)
                # Búsqueda parcial
                elif keyword_lower in title_lower:
                    found.append(keyword)
            
            if found:
                extracted[attr_type] = found[:3]  # Top 3
        
        return extracted
    
    def enrich_products_batch(self, products: List, batch_size: int = 1000,
                            cache_path: Optional[str] = None) -> List:
        """
        Procesamiento optimizado por lotes
        """
        # Verificar cache
        if cache_path:
            cache_file = Path(cache_path)
            if cache_file.exists():
                logger.info(f"📂 Cargando NER cache: {cache_path}")
                return self._load_from_cache(products, cache_path)
        
        total = len(products)
        logger.info(f"🎯 Enriqueciendo {total:,} productos...")
        
        enriched = []
        stats = {'zero_shot': 0, 'keywords': 0, 'errors': 0}
        
        for i in tqdm(range(0, total, batch_size), desc="NER Processing"):
            batch = products[i:i + batch_size]
            
            for product in batch:
                try:
                    title = getattr(product, 'title', '')
                    category = getattr(product, 'category', '')
                    
                    if title:
                        attributes = self.extract_attributes(title, category)
                        
                        # Estadísticas
                        if attributes:
                            if self.use_zero_shot and self.classifier:
                                stats['zero_shot'] += 1
                            else:
                                stats['keywords'] += 1
                        
                        product.ner_attributes = attributes
                        
                        # Crear texto enriquecido
                        enriched_text = self._create_enriched_text(title, attributes)
                        product.enriched_text = enriched_text
                        
                    else:
                        product.ner_attributes = {}
                        product.enriched_text = ''
                    
                    enriched.append(product)
                    
                except Exception as e:
                    stats['errors'] += 1
                    product.ner_attributes = {}
                    product.enriched_text = getattr(product, 'title', '')
                    enriched.append(product)
            
            # Checkpoint cada 5000 productos
            if cache_path and (i + batch_size) % 5000 == 0:
                temp_cache = f"{cache_path}.temp"
                self._save_to_cache(enriched, temp_cache)
                logger.info(f"  💾 Checkpoint: {len(enriched):,}/{total:,}")
        
        # Log estadísticas
        logger.info(f"✅ Enriquecidos: {total:,} productos")
        logger.info(f"   • Zero-shot: {stats['zero_shot']:,}")
        logger.info(f"   • Keywords: {stats['keywords']:,}")
        logger.info(f"   • Errores: {stats['errors']:,}")
        
        # Guardar cache final
        if cache_path:
            self._save_to_cache(enriched, cache_path)
        
        return enriched
    
    def _create_enriched_text(self, title: str, attributes: Dict) -> str:
        """Crea texto enriquecido optimizado"""
        if not attributes:
            return title
        
        parts = [title]
        
        for attr_type, values in attributes.items():
            if values:
                # Formato compacto
                parts.append(f"{attr_type}:{','.join(values[:2])}")
        
        return " | ".join(parts)
    
    def _save_to_cache(self, products: List, cache_path: str):
        """Guarda cache optimizado"""
        try:
            cache_data = []
            
            for product in products:
                cache_data.append({
                    'id': getattr(product, 'id', ''),
                    'ner_attributes': getattr(product, 'ner_attributes', {}),
                    'enriched_text': getattr(product, 'enriched_text', ''),
                    'title_hash': hashlib.md5(str(getattr(product, 'title', '')).encode()).hexdigest()[:8]
                })
            
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"💾 Cache guardado: {len(cache_data):,} productos")
            
        except Exception as e:
            logger.error(f"❌ Error guardando cache: {e}")
    
    def _load_from_cache(self, products: List, cache_path: str) -> List:
        """Carga desde cache con verificación"""
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Crear diccionario por ID
            cache_dict = {item['id']: item for item in cache_data}
            
            loaded = 0
            for product in products:
                product_id = getattr(product, 'id', '')
                if product_id in cache_dict:
                    data = cache_dict[product_id]
                    
                    # Verificar que el título no haya cambiado
                    current_title = getattr(product, 'title', '')
                    current_hash = hashlib.md5(str(current_title).encode()).hexdigest()[:8]
                    
                    if data.get('title_hash') == current_hash:
                        product.ner_attributes = data.get('ner_attributes', {})
                        product.enriched_text = data.get('enriched_text', current_title)
                        loaded += 1
                    else:
                        # Título cambió, re-procesar
                        product.ner_attributes = {}
                        product.enriched_text = current_title
            
            logger.info(f"📂 Cache cargado: {loaded:,}/{len(products):,}")
            return products
            
        except Exception as e:
            logger.error(f"❌ Error cargando cache: {e}")
            return products

# Función de conveniencia
def enrich_dataset_with_ner(products: List, use_zero_shot: bool = True,
                          cache_file: str = "data/cache/ner_attributes.pkl") -> List:
    """
    Función de alto nivel para enriquecer dataset completo
    """
    extractor = OptimizedNERExtractor(use_zero_shot=use_zero_shot)
    return extractor.enrich_products_batch(
        products, 
        batch_size=500,
        cache_path=cache_file
    )