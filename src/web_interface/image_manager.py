# src/web_interface/image_manager.py
"""
Gestor de imágenes mejorado con seguridad y gestión de caché
"""
import base64
import hashlib
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional
import requests
import logging

logger = logging.getLogger(__name__)

class EnhancedImageManager:
    """Gestor de imágenes con seguridad y gestión de caché"""
    
    def __init__(self, 
                 cache_dir: Path = Path("data/cache/images"),
                 max_cache_size_mb: int = 500,
                 max_image_size_mb: int = 5,
                 cache_days: int = 30):
        
        self.cache_dir = cache_dir
        self.max_cache_size_mb = max_cache_size_mb
        self.max_image_size_mb = max_image_size_mb
        self.cache_days = cache_days
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Inicializar caché
        self._clean_old_cache()
        
    def _clean_old_cache(self):
        """Limpia archivos de caché antiguos"""
        current_time = time.time()
        deleted = 0
        
        for cache_file in self.cache_dir.glob("*.jpg"):
            file_age = current_time - cache_file.stat().st_mtime
            if file_age > self.cache_days * 24 * 3600:
                cache_file.unlink()
                deleted += 1
        
        if deleted > 0:
            logger.info(f"Limpieza caché: {deleted} imágenes antiguas eliminadas")
        
        # Verificar tamaño total
        self._enforce_cache_limit()
    
    def _enforce_cache_limit(self):
        """Asegura que el caché no exceda el límite"""
        cache_files = list(self.cache_dir.glob("*.jpg"))
        
        if not cache_files:
            return
        
        # Calcular tamaño total
        total_size = sum(f.stat().st_size for f in cache_files)
        total_size_mb = total_size / (1024 * 1024)
        
        if total_size_mb <= self.max_cache_size_mb:
            return
        
        # Ordenar por antigüedad (más antiguos primero)
        cache_files.sort(key=lambda f: f.stat().st_mtime)
        
        deleted = 0
        while total_size_mb > self.max_cache_size_mb * 0.8 and cache_files:  # Dejar 80%
            old_file = cache_files.pop(0)
            file_size = old_file.stat().st_size
            old_file.unlink()
            deleted += 1
            total_size_mb -= file_size / (1024 * 1024)
        
        if deleted > 0:
            logger.info(f"Límite caché: {deleted} imágenes eliminadas. Tamaño actual: {total_size_mb:.1f}MB")
    
    def get_product_image(self, product_data: Dict) -> str:
        """Obtiene imagen para un producto con múltiples estrategias"""
        try:
            # 1. Intentar obtener URL de imagen
            image_url = self._extract_image_url(product_data)
            
            if image_url and image_url.startswith(('http://', 'https://')):
                # 2. Intentar descargar desde URL
                cached_path = self._download_and_cache_image(image_url, product_data)
                if cached_path:
                    return f"/api/image/{self._get_cache_key(product_data)}"
            
            # 3. Generar placeholder basado en datos del producto
            return self._generate_smart_placeholder(product_data)
            
        except Exception as e:
            logger.warning(f"Error obteniendo imagen: {e}")
            return self._generate_smart_placeholder(product_data)
    
    def _extract_image_url(self, product_data: Dict) -> Optional[str]:
        """Extrae URL de imagen del producto con múltiples estrategias"""
        # Campos prioritarios
        priority_fields = [
            'imageURLHighRes', 'largeImage', 'hiResImage',
            'imageUrl', 'image_url', 'image_url_large'
        ]
        
        for field in priority_fields:
            if field in product_data:
                url = self._safe_get_url(product_data[field])
                if url:
                    return url
        
        # Campos secundarios
        secondary_fields = ['image', 'images', 'img', 'imgs', 'picture', 'thumbnail']
        for field in secondary_fields:
            if field in product_data:
                url = self._safe_get_url(product_data[field])
                if url:
                    return url
        
        # Buscar en nested structures
        for key, value in product_data.items():
            if isinstance(value, dict) and 'url' in value:
                url = self._safe_get_url(value['url'])
                if url:
                    return url
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and 'url' in item:
                        url = self._safe_get_url(item['url'])
                        if url:
                            return url
        
        return None
    
    def _safe_get_url(self, value) -> Optional[str]:
        """Extrae URL de forma segura de diferentes tipos de datos"""
        if isinstance(value, str) and value.startswith(('http://', 'https://')):
            return value
        elif isinstance(value, dict) and 'url' in value:
            url_val = value['url']
            if isinstance(url_val, str) and url_val.startswith(('http://', 'https://')):
                return url_val
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.startswith(('http://', 'https://')):
                    return item
                elif isinstance(item, dict) and 'url' in item:
                    return self._safe_get_url(item['url'])
        return None
    
    def _get_cache_key(self, product_data: Dict) -> str:
        """Genera clave de caché única para el producto"""
        product_id = product_data.get('id', 'unknown')
        title_hash = hashlib.md5(str(product_data.get('title', '')).encode()).hexdigest()[:8]
        return f"{product_id}_{title_hash}"
    
    def _download_and_cache_image(self, image_url: str, product_data: Dict) -> Optional[Path]:
        """Descarga y cachea una imagen con validación"""
        cache_key = self._get_cache_key(product_data)
        cache_file = self.cache_dir / f"{cache_key}.jpg"
        
        # Verificar si ya está en caché (y es reciente)
        if cache_file.exists():
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < self.cache_days * 24 * 3600:
                return cache_file
            else:
                cache_file.unlink()  # Eliminar caché viejo
        
        try:
            # Descargar con timeout y tamaño máximo
            response = self.session.get(
                image_url,
                timeout=10,
                stream=True
            )
            response.raise_for_status()
            
            # Validar tipo de contenido
            content_type = response.headers.get('content-type', '').lower()
            if 'image' not in content_type:
                logger.warning(f"URL no es imagen: {content_type}")
                return None
            
            # Validar tamaño
            content_length = int(response.headers.get('content-length', 0))
            if content_length > self.max_image_size_mb * 1024 * 1024:
                logger.warning(f"Imagen muy grande: {content_length} bytes")
                return None
            
            # Descargar en chunks
            image_data = BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                image_data.write(chunk)
                
                # Verificar tamaño durante descarga
                if image_data.tell() > self.max_image_size_mb * 1024 * 1024:
                    raise ValueError("Imagen excede tamaño máximo")
            
            # Validar que sea imagen válida
            from PIL import Image
            try:
                image = Image.open(image_data)
                image.verify()  # Verificar integridad
                image_data.seek(0)
            except Exception as e:
                logger.warning(f"Imagen inválida: {e}")
                return None
            
            # Guardar en caché
            with open(cache_file, 'wb') as f:
                f.write(image_data.getvalue())
            
            logger.info(f"Imagen cachéada: {cache_key}")
            return cache_file
            
        except Exception as e:
            logger.warning(f"Error descargando imagen {image_url[:50]}: {e}")
            return None
    
    def _generate_smart_placeholder(self, product_data: Dict) -> str:
        """Genera placeholder SVG inteligente basado en el producto"""
        category = product_data.get('category', 'General').lower()
        title = product_data.get('title', 'Producto')
        product_id = product_data.get('id', 'unknown')
        
        # Colores por categoría
        category_colors = {
            'electronics': '#4facfe', 'computers': '#4facfe', 'phone': '#4facfe',
            'books': '#38b2ac', 'book': '#38b2ac', 'literature': '#38b2ac',
            'clothing': '#ed8936', 'fashion': '#ed8936', 'wear': '#ed8936',
            'home': '#9f7aea', 'kitchen': '#9f7aea', 'furniture': '#9f7aea',
            'sports': '#f56565', 'fitness': '#f56565', 'outdoor': '#f56565',
            'beauty': '#ed64a6', 'cosmetic': '#ed64a6', 'makeup': '#ed64a6',
            'automotive': '#48bb78', 'car': '#48bb78', 'vehicle': '#48bb78',
            'toys': '#ecc94b', 'games': '#ecc94b', 'toy': '#ecc94b',
            'video': '#667eea', 'game': '#667eea'
        }
        
        # Encontrar color por categoría
        color = '#667eea'  # Default
        for cat_key, cat_color in category_colors.items():
            if cat_key in category:
                color = cat_color
                break
        
        # Icono por categoría
        category_icons = {
            'electronics': '💻', 'computers': '💻', 'phone': '📱',
            'books': '📚', 'book': '📚', 'literature': '📚',
            'clothing': '👕', 'fashion': '👕', 'wear': '👕',
            'home': '🏠', 'kitchen': '🏠', 'furniture': '🏠',
            'sports': '⚽', 'fitness': '⚽', 'outdoor': '⚽',
            'beauty': '💄', 'cosmetic': '💄', 'makeup': '💄',
            'automotive': '🚗', 'car': '🚗', 'vehicle': '🚗',
            'toys': '🎮', 'games': '🎮', 'toy': '🎮',
            'video': '🎮', 'game': '🎮'
        }
        
        icon = '📦'  # Default
        for cat_key, cat_icon in category_icons.items():
            if cat_key in category:
                icon = cat_icon
                break
        
        # Obtener palabras clave del título
        title_words = title.split()[:3]
        keywords = ' '.join(title_words)
        
        # Crear SVG placeholder
        svg = f'''
        <svg width="300" height="300" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:{color};stop-opacity:0.1" />
                    <stop offset="100%" style="stop-color:{color};stop-opacity:0.3" />
                </linearGradient>
            </defs>
            
            <rect width="300" height="300" fill="url(#grad1)" rx="15"/>
            
            <rect width="260" height="180" x="20" y="20" fill="white" rx="10" 
                  stroke="{color}" stroke-width="2" opacity="0.9"/>
            
            <text x="150" y="80" font-family="Arial, sans-serif" font-size="48" 
                  fill="{color}" text-anchor="middle" opacity="0.7">
                {icon}
            </text>
            
            <text x="150" y="130" font-family="Arial, sans-serif" font-size="14" 
                  fill="#4a5568" text-anchor="middle" font-weight="600">
                {keywords}
            </text>
            
            <text x="150" y="160" font-family="Arial, sans-serif" font-size="12" 
                  fill="#718096" text-anchor="middle">
                {category.title()}
            </text>
            
            <text x="150" y="250" font-family="Arial, sans-serif" font-size="10" 
                  fill="#a0aec0" text-anchor="middle">
                ID: {product_id[:10]}
            </text>
        </svg>
        '''
        
        return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}"
    
    def get_cached_image(self, cache_key: str) -> Optional[Path]:
        """Obtiene ruta de imagen cachéada si existe"""
        if not cache_key or not isinstance(cache_key, str):
            return None
        
        # Sanitizar cache_key
        safe_key = re.sub(r'[^a-zA-Z0-9_-]', '', cache_key)
        if not safe_key:
            return None
        
        cache_file = self.cache_dir / f"{safe_key}.jpg"
        
        # Verificar path traversal
        try:
            if not cache_file.resolve().is_relative_to(self.cache_dir.resolve()):
                return None
        except:
            return None
        
        return cache_file if cache_file.exists() else None