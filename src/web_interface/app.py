# src/web_interface/app.py
"""
Interfaz Web para Sistema RAG+NER+RLHF - Versión Corregida
"""
import sys
import json
import base64
import re
import time
import urllib.parse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict
from io import BytesIO
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# Configurar paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir.parent))

# Configuración
CONFIG = {
    'image_proxy_enabled': True,
    'image_cache_dir': Path("data/cache/images"),
    'image_cache_max_size': 100,  # MB
    'image_cache_max_age': 30,  # días
    'image_cache_max_files': 1000,
    'max_products': 100000,
    'results_per_page': 10,
    'request_timeout': 10,
    'max_image_size': 5 * 1024 * 1024,  # 5MB
}

# Importar dependencias con manejo de errores
try:
    from flask import Flask, render_template, request, jsonify, send_file, abort
    import requests
    HAS_REQUIREMENTS = True
except ImportError as e:
    print(f"⚠️ Dependencias faltantes: {e}")
    print("   Instalar: pip install flask requests")
    HAS_REQUIREMENTS = False

# Importar PIL opcionalmente
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️ Pillow no instalado. La validación de imágenes será limitada.")
    print("   Instalar: pip install pillow")

if HAS_REQUIREMENTS:
    app = Flask(__name__)
    app.secret_key = 'rag_ner_rlhf_secret_key_2024_v2'
else:
    print("❌ No se puede ejecutar sin Flask y requests")
    sys.exit(1)


class LRUImageCache:
    """Cache LRU para imágenes con límite de tamaño"""
    
    def __init__(self, max_size_mb: int = 100, max_files: int = 1000, max_age_days: int = 30):
        self.cache_dir = CONFIG['image_cache_dir']
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.max_files = max_files
        self.max_age_seconds = max_age_days * 24 * 3600
        self.access_order = OrderedDict()
        self._initialize_cache()
        
    def _initialize_cache(self):
        """Inicializa el cache escaneando directorio"""
        print(f"🔄 Inicializando LRU cache (máx: {self.max_size_mb}MB, {self.max_files} archivos)")
        
        # Escanear archivos existentes
        files = []
        for cache_file in self.cache_dir.glob("*.jpg"):
            try:
                stat = cache_file.stat()
                files.append({
                    'path': cache_file,
                    'size': stat.st_size,
                    'mtime': stat.st_mtime,
                    'atime': stat.st_atime
                })
            except OSError:
                continue
        
        # Ordenar por último acceso
        files.sort(key=lambda x: x['atime'], reverse=True)
        
        # Mantener solo los más recientes
        total_size = 0
        kept_files = 0
        
        for file_info in files:
            # Verificar si el archivo es muy viejo
            if time.time() - file_info['mtime'] > self.max_age_seconds:
                try:
                    file_info['path'].unlink()
                except:
                    pass
                continue
                
            # Verificar límites
            if (total_size + file_info['size'] <= self.max_size_mb * 1024 * 1024 and 
                kept_files < self.max_files):
                # Mantener en cache
                self.access_order[str(file_info['path'].relative_to(self.cache_dir))] = {
                    'size': file_info['size'],
                    'last_access': file_info['atime']
                }
                total_size += file_info['size']
                kept_files += 1
            else:
                # Eliminar archivo excedente
                try:
                    file_info['path'].unlink()
                except:
                    pass
        
        print(f"✅ Cache inicializado: {kept_files} archivos, {total_size/1024/1024:.2f}MB")
    
    def get(self, product_id: str) -> Optional[Path]:
        """Obtiene una imagen del cache"""
        cache_key = f"{product_id}.jpg"
        
        if cache_key in self.access_order:
            # Actualizar orden de acceso
            self.access_order.move_to_end(cache_key)
            self.access_order[cache_key]['last_access'] = time.time()
            
            cache_file = self.cache_dir / cache_key
            
            # Verificar que el archivo existe y no es muy viejo
            try:
                if (cache_file.exists() and 
                    time.time() - cache_file.stat().st_mtime < self.max_age_seconds):
                    return cache_file
            except OSError:
                pass
        
        return None
    
    def put(self, product_id: str, image_data: bytes) -> Path:
        """Agrega una imagen al cache"""
        cache_key = f"{product_id}.jpg"
        cache_file = self.cache_dir / cache_key
        
        # Guardar archivo
        with open(cache_file, 'wb') as f:
            f.write(image_data)
        
        # Actualizar cache LRU
        self.access_order[cache_key] = {
            'size': len(image_data),
            'last_access': time.time()
        }
        
        # Limpiar si excede límites
        self._cleanup()
        
        return cache_file
    
    def _cleanup(self):
        """Limpia el cache si excede límites"""
        current_size = sum(info['size'] for info in self.access_order.values())
        current_count = len(self.access_order)
        
        # Verificar si necesita limpieza
        needs_cleanup = (current_size > self.max_size_mb * 1024 * 1024 or 
                        current_count > self.max_files)
        
        if not needs_cleanup:
            return
        
        print(f"🧹 Limpiando cache: {current_count} archivos, {current_size/1024/1024:.2f}MB")
        
        # Eliminar archivos más antiguos primero
        removed_size = 0
        removed_count = 0
        
        while (self.access_order and 
               (current_size - removed_size > self.max_size_mb * 1024 * 1024 * 0.9 or 
                current_count - removed_count > self.max_files * 0.9)):
            
            cache_key, info = self.access_order.popitem(last=False)
            cache_file = self.cache_dir / cache_key
            
            try:
                cache_file.unlink()
                removed_size += info['size']
                removed_count += 1
            except OSError:
                pass
        
        print(f"✅ Cache limpiado: eliminados {removed_count} archivos, {removed_size/1024/1024:.2f}MB")

# REEMPLAZAR EN app.py - SecureProductImageManager mejorado

class SecureProductImageManager:
    """Gestor seguro de imágenes de productos - VERSIÓN CORREGIDA"""
    
    def __init__(self):
        self.cache = LRUImageCache(
            max_size_mb=CONFIG['image_cache_max_size'],
            max_files=CONFIG['image_cache_max_files'],
            max_age_days=CONFIG['image_cache_max_age']
        )
        self.session = self._create_secure_session()
        self.allowed_content_types = {
            'image/jpeg', 'image/jpg', 'image/png', 
            'image/gif', 'image/webp', 'image/svg+xml'
        }
        
        # ✅ NUEVO: Blacklist en lugar de whitelist
        self.blocked_domains = {
            'localhost', '127.0.0.1', '0.0.0.0',
            'internal', 'admin', 'test'
        }
        
    def _create_secure_session(self):
        """Crea una sesión HTTP segura"""
        session = requests.Session()
        
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        from requests.adapters import HTTPAdapter
        from requests.packages.urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=2,  # Reducido para ser más rápido
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        return session
    
    def get_product_image(self, product_data: Dict) -> str:
        """Obtiene imagen para un producto de forma segura"""
        try:
            product_id = self._sanitize_id(product_data.get('id', 'unknown'))
            cached_file = self.cache.get(product_id)
            
            if cached_file:
                return f"/api/image/{product_id}"
            
            # ✅ MEJORADO: Extracción de imagen más robusta
            image_url = self._extract_image_url(product_data)
            
            if not image_url:
                print(f"⚠️ No se encontró imagen para {product_id}")
                print(f"   Datos disponibles: {list(product_data.keys())[:10]}")
                return self._generate_placeholder(product_data)
            
            print(f"📸 Intentando descargar: {image_url[:80]}...")
            
            image_data = self._download_image_safely(image_url, product_id)
            
            if image_data:
                self.cache.put(product_id, image_data)
                print(f"✅ Imagen cacheada para {product_id}")
                return f"/api/image/{product_id}"
            else:
                print(f"❌ Fallo descargando imagen para {product_id}")
                return self._generate_placeholder(product_data)
                
        except Exception as e:
            print(f"⚠️ Error obteniendo imagen: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_placeholder(product_data)
    
    def _extract_image_url(self, product_data: Dict) -> Optional[str]:
        """
        Extrae URL de imagen del producto - VERSIÓN MEJORADA
        Soporta múltiples formatos de datos
        """
        
        # DEBUG: Mostrar estructura del producto
        print(f"🔍 Buscando imagen en producto:")
        print(f"   Keys disponibles: {list(product_data.keys())}")
        
        # ============ PRIORIDAD 1: raw_data (datos originales) ============
        if 'raw_data' in product_data:
            raw = product_data['raw_data']
            print(f"   raw_data keys: {list(raw.keys()) if isinstance(raw, dict) else 'no dict'}")
            
            # Caso 1: raw_data tiene 'images' como lista de DICTS
            if 'images' in raw and isinstance(raw['images'], list) and len(raw['images']) > 0:
                first_img = raw['images'][0]
                
                if isinstance(first_img, dict):
                    # Buscar en campos comunes
                    for field in ['large', 'hi_res', 'hiRes', 'thumb', 'url', 'src']:
                        if field in first_img and first_img[field]:
                            url = str(first_img[field])
                            if self._is_safe_url(url):
                                print(f"   ✅ Encontrada en raw_data.images[0].{field}")
                                return url
                
                # Caso 2: raw_data tiene 'images' como lista de STRINGS
                elif isinstance(first_img, str):
                    if self._is_safe_url(first_img):
                        print(f"   ✅ Encontrada en raw_data.images[0] (string)")
                        return first_img
            
            # Caso 3: raw_data tiene campo 'image' directo
            for field in ['image', 'imageURL', 'image_url', 'main_image', 'primaryImage']:
                if field in raw and raw[field]:
                    url = str(raw[field])
                    if self._is_safe_url(url):
                        print(f"   ✅ Encontrada en raw_data.{field}")
                        return url
        
        # ============ PRIORIDAD 2: Campo 'images' directo ============
        if 'images' in product_data and isinstance(product_data['images'], list):
            for image_info in product_data['images']:
                if isinstance(image_info, dict):
                    for field in ['large', 'hi_res', 'hiRes', 'thumb', 'url', 'src']:
                        if field in image_info and image_info[field]:
                            url = str(image_info[field])
                            if self._is_safe_url(url):
                                print(f"   ✅ Encontrada en images[].{field}")
                                return url
                
                elif isinstance(image_info, str):
                    if self._is_safe_url(image_info):
                        print(f"   ✅ Encontrada en images[] (string)")
                        return image_info
        
        # ============ PRIORIDAD 3: Campos directos ============
        image_fields = [
            'image', 'image_url', 'imageURL', 'thumbnail', 
            'primary_image', 'primaryImage', 'main_image',
            'picture', 'photo', 'img'
        ]
        
        for field in image_fields:
            if field in product_data and product_data[field]:
                url = str(product_data[field])
                if self._is_safe_url(url):
                    print(f"   ✅ Encontrada en {field}")
                    return url
        
        # ============ PRIORIDAD 4: En 'details' ============
        if 'details' in product_data and isinstance(product_data['details'], dict):
            details = product_data['details']
            for field in image_fields:
                if field in details and details[field]:
                    url = str(details[field])
                    if self._is_safe_url(url):
                        print(f"   ✅ Encontrada en details.{field}")
                        return url
        
        print(f"   ❌ No se encontró URL de imagen válida")
        return None
    
    def _is_safe_url(self, url: str) -> bool:
        """
        Verifica si una URL es segura - VERSIÓN MEJORADA (blacklist)
        """
        try:
            if not isinstance(url, str) or len(url) < 10:
                return False
            
            if not url.startswith(('http://', 'https://')):
                return False
            
            parsed = urllib.parse.urlparse(url)
            if not parsed.netloc:
                return False
            
            domain = parsed.netloc.lower()
            
            # ✅ BLACKLIST: Rechazar dominios peligrosos
            if any(blocked in domain for blocked in self.blocked_domains):
                print(f"   ⚠️ Dominio bloqueado: {domain}")
                return False
            
            # ✅ Verificar que no sea IP local
            if domain.startswith(('192.168.', '10.', '172.')):
                print(f"   ⚠️ IP privada bloqueada: {domain}")
                return False
            
            # ✅ ACEPTAR: Cualquier otro dominio público
            print(f"   ✅ URL segura: {domain}")
            return True
            
        except Exception as e:
            print(f"   ⚠️ Error validando URL: {e}")
            return False
    
    def _download_image_safely(self, image_url: str, product_id: str) -> Optional[bytes]:
        """Descarga una imagen de forma segura - VERSIÓN MEJORADA"""
        try:
            timeout = (3, 8)  # Reducido: (connect, read)
            
            response = self.session.get(
                image_url,
                stream=True,
                timeout=timeout,
                allow_redirects=True,
                verify=False  # ✅ TEMPORAL: Desactivar verificación SSL para testing
            )
            
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            print(f"   Content-Type: {content_type}")
            
            # Aceptar cualquier tipo de imagen
            if not ('image' in content_type or 'octet-stream' in content_type):
                print(f"   ⚠️ Content-type sospechoso: {content_type}")
            
            image_data = BytesIO()
            size = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    image_data.write(chunk)
                    size += len(chunk)
                    
                    if size > CONFIG['max_image_size']:
                        print(f"   ⚠️ Imagen muy grande: {size} bytes")
                        return None
            
            image_bytes = image_data.getvalue()
            
            if size < 100:
                print(f"   ⚠️ Imagen muy pequeña: {size} bytes")
                return None
            
            print(f"   ✅ Descargada: {size} bytes")
            
            # Validar con PIL si está disponible
            if HAS_PIL:
                try:
                    img = Image.open(BytesIO(image_bytes))
                    img.verify()
                    print(f"   ✅ Formato válido: {img.format}")
                except Exception as e:
                    print(f"   ⚠️ PIL validation falló: {e}")
                    # Aún así retornar la imagen
            
            return image_bytes
            
        except requests.exceptions.Timeout:
            print(f"   ❌ Timeout")
        except requests.exceptions.SSLError as e:
            print(f"   ❌ Error SSL: {e}")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Error de red: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        return None
    
    def _sanitize_id(self, product_id: str) -> str:
        """Sanitiza un ID de producto para uso seguro en rutas"""
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', str(product_id))
        return sanitized[:50] if sanitized else 'unknown'
    
    def _generate_placeholder(self, product_data: Dict) -> str:
        """Genera un placeholder SVG seguro basado en el producto"""
        try:
            category = str(product_data.get('category', 'General')).lower()[:20]
            product_id = self._sanitize_id(product_data.get('id', 'unknown'))[:10]
            title = str(product_data.get('title', ''))[:30]
            
            category_colors = {
                'electronics': '#4facfe',
                'books': '#38b2ac', 
                'clothing': '#ed8936',
                'home': '#9f7aea',
                'sports': '#f56565',
                'beauty': '#ed64a6',
                'automotive': '#48bb78',
                'toys': '#ecc94b',
                'video games': '#ed8936',
                'general': '#667eea'
            }
            
            color = category_colors.get(category, '#667eea')
            for cat_key in category_colors:
                if cat_key in category:
                    color = category_colors[cat_key]
                    break
            
            svg_template = '''<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300" viewBox="0 0 300 300">
                <rect width="300" height="300" fill="{color}" opacity="0.1"/>
                <rect x="20" y="20" width="260" height="180" rx="10" fill="white"/>
                <text x="150" y="110" font-family="Arial, sans-serif" font-size="14" 
                      fill="#666" text-anchor="middle">{category}</text>
                <text x="150" y="230" font-family="Arial, sans-serif" font-size="12" 
                      fill="#888" text-anchor="middle">{title}</text>
                <text x="150" y="250" font-family="Arial, sans-serif" font-size="10" 
                      fill="#aaa" text-anchor="middle">{product_id}</text>
            </svg>'''
            
            svg = svg_template.format(
                color=color,
                category=category.upper(),
                title=title,
                product_id=f"ID: {product_id}"
            )
            
            return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}"
            
        except Exception:
            svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300">
                <rect width="100%" height="100%" fill="#f8f9fa"/>
                <text x="150" y="150" font-family="Arial" font-size="16" 
                      fill="#666" text-anchor="middle">No image</text>
            </svg>'''
            return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}"
class WebUnifiedSystem:
    """Sistema unificado para web con mejoras"""
    
    def __init__(self):
        self.system = None
        self.image_manager = SecureProductImageManager()
        self.user_sessions = {}
        self.interactions = []
        self.stats = {
            'total_queries': 0,
            'total_feedback': 0,
            'active_users': set(),
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    def load_system(self):
        """Carga el sistema unificado"""
        try:
            if self.system is not None:
                return True
            
            from unified_system_v2 import UnifiedSystemV2
            
            system_cache = Path("data/cache/unified_system_v2.pkl")
            if system_cache.exists():
                print("🔄 Cargando sistema desde cache...")
                self.system = UnifiedSystemV2.load_from_cache()
            else:
                print("⚠️ Sistema no encontrado en cache")
                return False
            
            if self.system and hasattr(self.system, 'canonical_products'):
                print(f"✅ Sistema cargado: {len(self.system.canonical_products):,} productos")
                
                # Verificar datos de ejemplo
                if self.system.canonical_products:
                    sample = self.system.canonical_products[0]
                    print(f"   • Ejemplo: {getattr(sample, 'title', 'N/A')[:50]}...")
                    print(f"   • Categoría: {getattr(sample, 'category', 'N/A')}")
                    print(f"   • ID: {getattr(sample, 'id', 'N/A')}")
                
                return True
            
        except ImportError as e:
            print(f"❌ Error importando: {e}")
        except Exception as e:
            print(f"❌ Error cargando sistema: {e}")
            import traceback
            traceback.print_exc()
        
        return False
    
    def search_products(self, query: str, method: str = 'full_hybrid', k: int = 20) -> Dict:
        """Busca productos"""
        self.stats['total_queries'] += 1
        
        if not self.system:
            return {
                'success': False, 
                'error': 'Sistema no inicializado',
                'code': 'SYSTEM_NOT_LOADED'
            }
        
        if not query or len(query) < 2:
            return {
                'success': False,
                'error': 'Query demasiado corta',
                'code': 'QUERY_TOO_SHORT'
            }
        
        try:
            # Usar el sistema para buscar
            results = self.system.query_four_methods(query, k=k)
            
            if method not in results['methods']:
                method = 'baseline'
            
            products = results['methods'].get(method, [])
            
            # Formatear productos para la web
            formatted_products = []
            for i, product in enumerate(products[:CONFIG['results_per_page']]):
                product_data = self._format_product_for_web(product, i+1)
                
                # Añadir imagen
                product_data['image'] = self.image_manager.get_product_image(product_data)
                
                formatted_products.append(product_data)
            
            return {
                'success': True,
                'query': query,
                'method': method,
                'products': formatted_products,
                'stats': {
                    'total_found': len(products),
                    'shown': len(formatted_products),
                    'methods_available': list(results['methods'].keys()),
                    'timing': results.get('timing', {})
                }
            }
            
        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False, 
                'error': 'Error interno del sistema',
                'code': 'INTERNAL_ERROR'
            }
    
    def _format_product_for_web(self, product, position: int) -> Dict:
        """Formatea un producto para la web"""
        try:
            # Obtener atributos básicos
            title = getattr(product, 'title', 'Sin título')
            description = getattr(product, 'description', '')
            category = getattr(product, 'category', 'General')
            product_id = getattr(product, 'id', f'prod_{position}')
            price = getattr(product, 'price', None)
            rating = getattr(product, 'rating', None)
            brand = getattr(product, 'brand', '')
            
            # Extraer más datos si existen
            additional_data = {}
            if hasattr(product, 'raw_data'):
                try:
                    additional_data = getattr(product, 'raw_data', {})
                except:
                    pass
            
            # Construir datos del producto
            product_dict = {
                'position': position,
                'id': str(product_id),
                'title': str(title)[:100],
                'description': self._truncate_text(str(description), 150),
                'price': self._format_price(price),
                'category': str(category)[:30],
                'rating': self._format_rating(rating),
                'rating_count': getattr(product, 'rating_count', 0),
                'brand': str(brand)[:50],
                'has_ner': hasattr(product, 'ner_attributes') and bool(getattr(product, 'ner_attributes', {})),
                'features': getattr(product, 'features_dict', {})
            }
            
            # Añadir atributos NER si existen
            if product_dict['has_ner']:
                ner_attrs = getattr(product, 'ner_attributes', {})
                product_dict['ner_attributes'] = ner_attrs
                product_dict['ner_tags'] = [
                    f"{key}: {', '.join(values[:2])}" 
                    for key, values in ner_attrs.items() 
                    if values and isinstance(values, list)
                ][:5]
            
            # Añadir datos adicionales para extracción de imágenes
            if additional_data:
                product_dict.update({
                    'images': additional_data.get('images', []),
                    'details': additional_data.get('details', {}),
                    'raw_data': additional_data  # Para depuración
                })
            # 🔍 DEBUG temporal
            if position == 1:  # Solo el primer producto
                print("=" * 80)
                print("ESTRUCTURA DEL PRODUCTO:")
                print(f"  Atributos: {dir(product)}")
                if hasattr(product, 'raw_data'):
                    print(f"  raw_data keys: {list(product.raw_data.keys())}")
                    if 'images' in product.raw_data:
                        print(f"  images type: {type(product.raw_data['images'])}")
                        print(f"  images value: {product.raw_data['images'][:200]}")
                print("=" * 80)
            return product_dict
            
        except Exception as e:
            print(f"⚠️ Error formateando producto: {e}")
            # Producto mínimo
            return {
                'position': position,
                'id': f'prod_{position}',
                'title': 'Producto',
                'description': '',
                'price': 'N/A',
                'category': 'General',
                'rating': 'Sin rating',
                'brand': '',
                'image': ''
            }
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Trunca texto"""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        
        truncated = text[:max_length].rsplit(' ', 1)[0]
        return truncated + '...' if truncated else text[:max_length-3] + '...'
    
    def _format_price(self, price) -> str:
        """Formatea precio"""
        if price is None:
            return "No disponible"
        
        try:
            price_float = float(price)
            if price_float < 0:
                return "No disponible"
            return f"${price_float:.2f}"
        except:
            return "No disponible"
    
    def _format_rating(self, rating) -> str:
        """Formatea rating"""
        if rating is None:
            return "Sin rating"
        
        try:
            rating_float = float(rating)
            if rating_float < 0 or rating_float > 5:
                return "Sin rating"
            
            full_stars = int(rating_float)
            half_star = rating_float - full_stars >= 0.5
            
            stars = '★' * full_stars
            if half_star:
                stars += '½'
            
            return f"{stars} ({rating_float:.1f}/5)"
        except:
            return "Sin rating"
    
    def record_feedback(self, user_id: str, interaction_data: Dict) -> Tuple[bool, str]:
        """Registra feedback del usuario"""
        try:
            # Validar feedback
            required_fields = ['product_id', 'rating', 'query']
            for field in required_fields:
                if field not in interaction_data:
                    return False, f"Campo requerido faltante: {field}"
            
            # Validar rating
            try:
                rating = float(interaction_data['rating'])
                if rating < 1 or rating > 5:
                    return False, "Rating debe estar entre 1 y 5"
            except:
                return False, "Rating inválido"
            
            # Crear interacción
            interaction = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'product_id': interaction_data['product_id'],
                'rating': rating,
                'query': interaction_data['query'],
                'product_category': interaction_data.get('product_category', 'General'),
                'type': 'feedback'
            }
            
            self.interactions.append(interaction)
            self.stats['total_feedback'] += 1
            self.stats['active_users'].add(user_id)
            
            # Guardar en archivo
            feedback_file = Path("data/interactions/web_feedback.jsonl")
            feedback_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(feedback_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(interaction, ensure_ascii=False) + '\n')
            
            return True, "Feedback registrado exitosamente"
            
        except Exception as e:
            print(f"❌ Error registrando feedback: {e}")
            return False, "Error interno del sistema"
    
    def get_system_stats(self) -> Dict:
        """Obtiene estadísticas del sistema"""
        if not self.system:
            return {
                'system_loaded': False,
                'total_products': 0,
                'active_users': len(self.stats['active_users']),
                'total_interactions': len(self.interactions)
            }
        
        try:
            stats = self.system.get_system_stats()
            
            return {
                'system_loaded': True,
                'total_products': len(self.system.canonical_products) if hasattr(self.system, 'canonical_products') else 0,
                'has_ner': stats.get('has_ner_ranker', False),
                'has_rlhf': stats.get('has_learned_rlhf', False),
                'ner_enriched': stats.get('ner_enriched_count', 0),
                'active_users': len(self.stats['active_users']),
                'total_queries': self.stats['total_queries'],
                'total_feedback': self.stats['total_feedback'],
                'total_interactions': len(self.interactions),
                'methods_available': ['baseline', 'ner_enhanced', 'rlhf', 'full_hybrid']
            }
        except Exception as e:
            print(f"❌ Error obteniendo estadísticas: {e}")
            return {'system_loaded': False, 'error': str(e)}
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Obtiene estadísticas de usuario"""
        session = self.user_sessions.get(user_id, {})
        
        return {
            'user_id': user_id,
            'session_created': session.get('created', datetime.now().isoformat()),
            'feedback_count': session.get('feedback_count', 0),
            'preferences': session.get('preferences', {}),
            'total_interactions': len([i for i in self.interactions if i.get('user_id') == user_id])
        }


# Inicializar sistema global
web_system = WebUnifiedSystem()

# Rutas de la API
@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def api_search():
    """API para buscar productos"""
    if not request.is_json:
        return jsonify({
            'success': False, 
            'error': 'Content-Type debe ser application/json'
        }), 415
    
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({
            'success': False, 
            'error': 'Query requerida'
        }), 400
    
    query = data['query'].strip()
    method = data.get('method', 'full_hybrid')
    user_id = data.get('user_id', 'anonymous')
    
    if not query:
        return jsonify({
            'success': False, 
            'error': 'Query vacía'
        }), 400
    
    # Cargar sistema si no está cargado
    if not web_system.system:
        if not web_system.load_system():
            return jsonify({
                'success': False, 
                'error': 'Sistema no disponible'
            }), 503
    
    # Buscar productos
    result = web_system.search_products(query, method)
    
    return jsonify(result)

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """API para recibir feedback"""
    if not request.is_json:
        return jsonify({
            'success': False, 
            'error': 'Content-Type debe ser application/json'
        }), 415
    
    data = request.get_json()
    
    if not data:
        return jsonify({
            'success': False, 
            'error': 'JSON inválido'
        }), 400
    
    required_fields = ['product_id', 'rating', 'query']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return jsonify({
            'success': False, 
            'error': f'Campos requeridos faltantes: {", ".join(missing_fields)}'
        }), 400
    
    product_id = data['product_id']
    rating = data['rating']
    query = data['query']
    user_id = data.get('user_id', 'anonymous')
    product_category = data.get('product_category', 'General')
    
    # Registrar feedback
    success, message = web_system.record_feedback(user_id, {
        'product_id': product_id,
        'rating': rating,
        'query': query,
        'product_category': product_category
    })
    
    if success:
        return jsonify({
            'success': True,
            'message': message,
            'user_stats': web_system.get_user_stats(user_id)
        })
    else:
        return jsonify({
            'success': False,
            'error': message
        }), 400

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """API para obtener estadísticas del sistema"""
    stats = web_system.get_system_stats()
    return jsonify({'success': True, 'stats': stats})

@app.route('/api/user/<user_id>', methods=['GET'])
def api_user(user_id):
    """API para obtener estadísticas de usuario"""
    if not user_id:
        return jsonify({
            'success': False,
            'error': 'User ID requerido'
        }), 400
    
    stats = web_system.get_user_stats(user_id)
    return jsonify({'success': True, 'user': stats})

@app.route('/api/methods', methods=['GET'])
def api_methods():
    """API para obtener métodos disponibles"""
    methods = [
        {
            'id': 'baseline',
            'name': 'Baseline (FAISS)',
            'description': 'Búsqueda vectorial básica',
            'icon': '🔍',
            'color': '#4facfe'
        },
        {
            'id': 'ner_enhanced',
            'name': 'NER Enhanced',
            'description': 'Mejorado con reconocimiento de entidades',
            'icon': '🏷️',
            'color': '#38b2ac'
        },
        {
            'id': 'rlhf',
            'name': 'RLHF',
            'description': 'Aprendizaje por refuerzo con feedback humano',
            'icon': '🧠',
            'color': '#9f7aea'
        },
        {
            'id': 'full_hybrid',
            'name': 'Full Hybrid',
            'description': 'Combinación de NER + RLHF',
            'icon': '⚡',
            'color': '#f56565'
        }
    ]
    
    return jsonify({'success': True, 'methods': methods})

@app.route('/api/image/<product_id>', methods=['GET'])
def api_image(product_id):
    """API para servir imágenes de productos"""
    try:
        # Sanitizar product_id
        product_id = re.sub(r'[^a-zA-Z0-9_-]', '', str(product_id))
        
        if not product_id:
            return generate_placeholder_svg(), 200, {'Content-Type': 'image/svg+xml'}
        
        # Verificar cache
        cache_file = CONFIG['image_cache_dir'] / f"{product_id}.jpg"
        
        # Verificar path traversal
        try:
            cache_file = cache_file.resolve()
            cache_dir = CONFIG['image_cache_dir'].resolve()
            
            if not cache_file.is_relative_to(cache_dir):
                return generate_placeholder_svg(), 200, {'Content-Type': 'image/svg+xml'}
        except:
            return generate_placeholder_svg(), 200, {'Content-Type': 'image/svg+xml'}
        
        # Servir archivo si existe
        if cache_file.exists():
            return send_file(cache_file, mimetype='image/jpeg')
        else:
            return generate_placeholder_svg(), 200, {'Content-Type': 'image/svg+xml'}
            
    except Exception as e:
        print(f"❌ Error sirviendo imagen: {e}")
        return generate_placeholder_svg(), 200, {'Content-Type': 'image/svg+xml'}

def generate_placeholder_svg() -> str:
    """Genera un placeholder SVG"""
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300">
        <rect width="100%" height="100%" fill="#f8f9fa"/>
        <text x="150" y="150" font-family="Arial, sans-serif" font-size="16" 
              fill="#666" text-anchor="middle">No image</text>
    </svg>'''
    return svg

@app.route('/api/init_system', methods=['POST'])
def api_init_system():
    """API para inicializar el sistema"""
    try:
        if web_system.load_system():
            return jsonify({
                'success': True,
                'message': 'Sistema inicializado correctamente',
                'stats': web_system.get_system_stats()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Error inicializando sistema'
            }), 500
    except Exception as e:
        print(f"❌ Error en inicialización: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def api_health():
    """API de salud del sistema"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0',
        'system_loaded': web_system.system is not None
    })

if __name__ == '__main__':
    print("🚀 Iniciando servidor web RAG+NER+RLHF...")
    print("📊 Accede en: http://localhost:5000")
    
    # Mostrar configuración
    print(f"\n⚙️ Configuración:")
    print(f"   • Cache imágenes: {CONFIG['image_cache_max_size']}MB")
    print(f"   • Tiempo vida cache: {CONFIG['image_cache_max_age']} días")
    print(f"   • Máximo productos: {CONFIG['max_products']:,}")
    
    # Intentar cargar el sistema
    if web_system.load_system():
        stats = web_system.get_system_stats()
        print(f"\n✅ Sistema listo:")
        print(f"   • Productos: {stats['total_products']:,}")
        print(f"   • NER: {'✅' if stats.get('has_ner') else '❌'}")
        print(f"   • RLHF: {'✅' if stats.get('has_rlhf') else '❌'}")
    else:
        print("\n⚠️ Sistema no cargado. Usa la opción 'Inicializar Sistema' en la web.")
    
    print("\n📈 Rutas disponibles:")
    print("   • GET  /                    - Interfaz web")
    print("   • POST /api/search          - Buscar productos")
    print("   • POST /api/feedback        - Enviar feedback")
    print("   • GET  /api/stats           - Estadísticas del sistema")
    print("   • GET  /api/health          - Salud del sistema")
    
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000,
        threaded=True
    )