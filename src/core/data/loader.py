# src/core/data/loader.py - VERSIÓN SIMPLIFICADA FINAL

import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FastDataLoader:
    """
    Cargador optimizado que usa settings como única fuente de verdad.
    Elimina toda duplicación de configuración ML.
    """

    def __init__(
        self,
        *,
        raw_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
        cache_enabled: bool = False,
        max_products_per_file: int = 500000,
        use_progress_bar: bool = True,
        # 🔥 ELIMINADO: Parámetros ML redundantes
        # La configuración ML viene de settings automáticamente
    ):
        # Importar settings después de definir la clase
        from src.core.config import settings
        
        self.raw_dir = Path(raw_dir) if raw_dir else settings.RAW_DIR
        self.processed_dir = Path(processed_dir) if processed_dir else settings.PROC_DIR
        self.cache_enabled = cache_enabled
        self.max_products_per_file = max_products_per_file
        self.use_progress_bar = use_progress_bar
        
        # 🔥 ELIMINADO: No más configuración ML duplicada
        # Todo viene de settings automáticamente
        
        # Crear directorios
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📂 FastDataLoader inicializado")
        logger.info(f"   • Raw dir: {self.raw_dir}")
        logger.info(f"   • Processed dir: {self.processed_dir}")
        logger.info(f"   • Cache: {'✅' if cache_enabled else '❌'}")

    # --------------------------------------------------
    # Método principal simplificado
    # --------------------------------------------------
    
    def load_data(self, output_file: Optional[Path] = None) -> List[Any]:
        """
        Carga datos usando configuración global desde settings.
        
        Args:
            output_file: Archivo de salida opcional
            
        Returns:
            Lista de productos
        """
        from src.core.data.product import Product, create_product
        
        start_time = time.time()
        
        if output_file is None:
            output_file = self.processed_dir / "products.json"
        
        logger.info("📊 Iniciando carga de datos...")
        
        # Cargar archivos disponibles
        files = self._discover_data_files()
        
        if not files:
            logger.warning("⚠️ No se encontraron archivos de datos")
            return self._create_sample_data(output_file)
        
        logger.info(f"📁 Archivos encontrados: {len(files)}")
        
        # Procesar archivos
        all_products = []
        for file_path in files:
            try:
                file_products = self._process_file(file_path)
                if file_products:
                    all_products.extend(file_products)
                    logger.debug(f"   • {file_path.name}: {len(file_products)} productos")
            except Exception as e:
                logger.warning(f"⚠️ Error procesando {file_path.name}: {e}")
        
        if not all_products:
            logger.error("❌ No se pudieron cargar productos")
            return self._create_sample_data(output_file)
        
        # 🔥 IMPORTANTE: La configuración ML está en settings
        # Product.from_dict() usará automáticamente esta configuración
        
        # Guardar productos
        self._save_products(all_products, output_file)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Carga completada en {elapsed_time:.1f}s")
        logger.info(f"📦 Productos cargados: {len(all_products)}")
        
        return all_products
    
    # --------------------------------------------------
    # Métodos de procesamiento de archivos
    # --------------------------------------------------
    
    def _discover_data_files(self) -> List[Path]:
        """Descubre archivos de datos en el directorio raw"""
        extensions = [".json", ".jsonl"]
        files = []
        
        for ext in extensions:
            files.extend(self.raw_dir.glob(f"*{ext}"))
        
        # Filtrar archivos válidos
        valid_files = []
        for f in files:
            if f.exists() and f.stat().st_size > 0:
                valid_files.append(f)
        
        # Ordenar por tamaño (más grandes primero)
        valid_files.sort(key=lambda x: x.stat().st_size, reverse=True)
        
        return valid_files[:25]  # Limitar a 5 archivos
    
    def _process_file(self, file_path: Path) -> List[Any]:
        """Procesa un archivo individual"""
        from src.core.data.product import Product, create_product
        
        try:
            if file_path.suffix.lower() == ".jsonl":
                return self._process_jsonl(file_path)
            else:
                return self._process_json(file_path)
        except Exception as e:
            logger.error(f"❌ Error procesando {file_path.name}: {e}")
            return []
    
    def _process_json(self, file_path: Path) -> List[Any]:
        """Procesa archivo JSON"""
        from src.core.data.product import Product, create_product
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                data = [data]
            
            products = []
            for i, item in enumerate(data):
                if i >= self.max_products_per_file:
                    break
                
                try:
                    # 🔥 USAR CONFIGURACIÓN GLOBAL: create_product() usa settings automáticamente
                    product = create_product(item)
                    products.append(product)
                except Exception as e:
                    logger.debug(f"   Saltando item inválido: {e}")
                    continue
            
            return products
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON inválido en {file_path.name}: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Error procesando JSON {file_path.name}: {e}")
            return []
    
    def _process_jsonl(self, file_path: Path) -> List[Any]:
        """Procesa archivo JSONL"""
        from src.core.data.product import Product, create_product
        
        products = []
        line_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line_count >= self.max_products_per_file:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            # 🔥 USAR CONFIGURACIÓN GLOBAL
                            product = create_product(item)
                            products.append(product)
                            line_count += 1
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.debug(f"   Error en línea: {e}")
                        continue
            
            return products
            
        except Exception as e:
            logger.error(f"❌ Error procesando JSONL {file_path.name}: {e}")
            return []
    
    # --------------------------------------------------
    # Métodos de utilidad
    # --------------------------------------------------
    
    def _create_sample_data(self, output_file: Path) -> List[Any]:
        """Crea datos de muestra si no hay datos reales"""
        from src.core.data.product import create_product
        
        logger.info("📝 Creando datos de muestra...")
        
        sample_data = [
            {
                "title": "Audífonos Bluetooth Inalámbricos",
                "description": "Audífonos de alta calidad con cancelación de ruido",
                "price": 89.99,
                "main_category": "Electronics",
                "product_type": "Headphones",
                "tags": ["wireless", "bluetooth", "noise-cancelling"]
            },
            {
                "title": "Libro de Programación Python",
                "description": "Aprende Python desde cero hasta avanzado",
                "price": 39.99,
                "main_category": "Books",
                "product_type": "Programming",
                "tags": ["python", "programming", "education"]
            },
            {
                "title": "Mouse Inalámbrico para Computadora",
                "description": "Mouse ergonómico con sensor óptico de alta precisión",
                "price": 29.99,
                "main_category": "Electronics",
                "product_type": "Computer Accessories",
                "tags": ["wireless", "mouse", "ergonomic"]
            }
        ]
        
        products = []
        for item in sample_data:
            try:
                product = create_product(item)
                products.append(product)
            except Exception as e:
                logger.warning(f"⚠️ Error creando producto de muestra: {e}")
        
        self._save_products(products, output_file)
        
        logger.info(f"✅ Datos de muestra creados: {len(products)} productos")
        return products
    
    def _save_products(self, products: List[Any], output_file: Path) -> None:
        """Guarda productos en archivo"""
        try:
            # Convertir productos a diccionarios
            product_dicts = []
            for product in products:
                try:
                    if hasattr(product, 'model_dump'):
                        product_dicts.append(product.model_dump())
                    elif hasattr(product, 'dict'):
                        product_dicts.append(product.dict())
                    else:
                        # Intentar extraer atributos básicos
                        product_dict = {
                            'id': getattr(product, 'id', ''),
                            'title': getattr(product, 'title', ''),
                            'description': getattr(product, 'description', ''),
                            'price': getattr(product, 'price', 0.0),
                            'main_category': getattr(product, 'main_category', ''),
                            'product_type': getattr(product, 'product_type', ''),
                        }
                        product_dicts.append(product_dict)
                except Exception as e:
                    logger.debug(f"   Error convirtiendo producto: {e}")
                    continue
            
            # Guardar en archivo
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(product_dicts, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Productos guardados: {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Error guardando productos: {e}")
    
    def _clean_product_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Limpia datos básicos del producto"""
        cleaned = item.copy()
        
        # Asegurar campos requeridos
        cleaned.setdefault('title', 'Producto sin nombre')
        cleaned.setdefault('description', 'Sin descripción')
        cleaned.setdefault('price', 0.0)
        cleaned.setdefault('main_category', 'General')
        
        # Limpiar título
        title = cleaned['title']
        if isinstance(title, str):
            cleaned['title'] = title.strip()[:200]
        
        # Limpiar descripción
        description = cleaned['description']
        if isinstance(description, list):
            cleaned['description'] = ' '.join(str(x) for x in description[:3])
        elif not isinstance(description, str):
            cleaned['description'] = str(description)[:5000]
        
        # Asegurar que price sea numérico
        try:
            price = cleaned['price']
            if isinstance(price, str):
                # Extraer números
                import re
                match = re.search(r'(\d+(?:[.,]\d+)?)', price)
                if match:
                    cleaned['price'] = float(match.group(1).replace(',', '.'))
                else:
                    cleaned['price'] = 0.0
            elif not isinstance(price, (int, float)):
                cleaned['price'] = 0.0
        except (ValueError, TypeError):
            cleaned['price'] = 0.0
        
        return cleaned
    
    # --------------------------------------------------
    # Métodos de información y estadísticas
    # --------------------------------------------------
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del loader"""
        from src.core.config import settings
        
        stats = {
            "raw_dir": str(self.raw_dir),
            "processed_dir": str(self.processed_dir),
            "max_products_per_file": self.max_products_per_file,
            "cache_enabled": self.cache_enabled,
            "total_products": self._get_total_products(),
            
            # 🔥 CONFIGURACIÓN ML DESDE SETTINGS
            "ml_config": {
                "ml_enabled": settings.ML_ENABLED,
                "ml_features": list(settings.ML_FEATURES),
                "ml_categories": settings.ML_CATEGORIES[:5] if settings.ML_CATEGORIES else []
            }
        }
        
        return stats
    
    def _get_total_products(self) -> int:
        """Obtiene número total de productos procesados"""
        output_file = self.processed_dir / "products.json"
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return len(data) if isinstance(data, list) else 0
            except Exception:
                return 0
        return 0
    
    def print_summary(self) -> None:
        """Imprime resumen del loader"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("📊 RESUMEN DEL FAST DATA LOADER")
        print("="*60)
        print(f"📂 Directorio raw: {stats['raw_dir']}")
        print(f"📁 Directorio procesado: {stats['processed_dir']}")
        print(f"📦 Productos totales: {stats['total_products']}")
        print(f"⚙️  Máximo por archivo: {stats['max_products_per_file']}")
        print(f"💾 Cache: {'✅ Activado' if stats['cache_enabled'] else '❌ Desactivado'}")
        
        ml_config = stats['ml_config']
        print(f"\n🤖 CONFIGURACIÓN ML:")
        print(f"   • Habilitado: {'✅ Sí' if ml_config['ml_enabled'] else '❌ No'}")
        if ml_config['ml_enabled']:
            print(f"   • Características: {', '.join(ml_config['ml_features'])}")
            print(f"   • Categorías: {', '.join(ml_config['ml_categories'])}")
        print("="*60)


# ----------------------------------------------------------
# Alias para compatibilidad
# ----------------------------------------------------------

DataLoader = FastDataLoader


# ----------------------------------------------------------
# Función de conveniencia
# ----------------------------------------------------------

def load_products(
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    max_products: int = 500000
) -> List[Any]:
    """
    Función de conveniencia para cargar productos.
    
    Args:
        raw_dir: Directorio de datos crudos
        processed_dir: Directorio de datos procesados
        max_products: Máximo de productos a cargar
        
    Returns:
        Lista de productos
    """
    from src.core.config import settings
    
    loader = FastDataLoader(
        raw_dir=raw_dir or settings.RAW_DIR,
        processed_dir=processed_dir or settings.PROC_DIR,
        max_products_per_file=max_products,
        cache_enabled=settings.CACHE_ENABLED
    )
    
    return loader.load_data()


# ----------------------------------------------------------
# Ejecución directa (para pruebas)
# ----------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    # Configurar logging básico
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 FastDataLoader - Prueba directa")
    print("="*50)
    
    try:
        # Cargar productos
        products = load_products(max_products=100)
        
        # Mostrar resumen
        if products:
            print(f"\n✅ Carga exitosa: {len(products)} productos")
            
            # Mostrar primeros 3 productos
            print("\n📋 Primeros 3 productos:")
            for i, product in enumerate(products[:3]):
                title = getattr(product, 'title', 'Sin título')
                price = getattr(product, 'price', 0.0)
                category = getattr(product, 'main_category', 'General')
                
                print(f"   {i+1}. {title}")
                print(f"      Precio: ${price:.2f}")
                print(f"      Categoría: {category}")
                print()
        else:
            print("❌ No se pudieron cargar productos")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)