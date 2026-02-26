"""
init_esci_system.py
===================
Inicializa el sistema con el corpus ESCI construido por build_esci_corpus.py.

Qué hace:
    1. Carga el corpus ESCI (products.parquet + embeddings.npz + faiss.index)
    2. Construye objetos de producto compatibles con UnifiedSystemV2
    3. Reemplaza el corpus del sistema
    4. Reconstruye el índice FAISS del sistema con los nuevos embeddings
    5. Actualiza la caché del sistema (unified_system_v2.pkl)
    6. Actualiza product_embeddings.npz (usado por pretrain_reward_esci.py)

Prerequisitos:
    python verify_and_split_esci.py    <- splits limpios
    python build_esci_corpus.py        <- corpus construido

Salidas:
    data/cache/unified_system_v2.pkl   (reemplazado)
    data/cache/product_embeddings.npz  (reemplazado)
    data/cache/system_backup_TIMESTAMP.pkl  (backup del original)

Uso:
    python init_esci_system.py
    python init_esci_system.py --no-backup   # sin backup (más rápido)
"""
import argparse
import json
import logging
import pickle
import shutil
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

ESCI_CORPUS_DIR = Path("data/esci_corpus")
CACHE_DIR       = Path("data/cache")
SYSTEM_CACHE    = CACHE_DIR / "unified_system_v2.pkl"
EMB_CACHE       = CACHE_DIR / "product_embeddings.npz"


# ---------------------------------------------------------------------
# Verificación de prerequisitos
# ---------------------------------------------------------------------
def check_prerequisites():
    errors = []
    if not (ESCI_CORPUS_DIR / "products.parquet").exists():
        errors.append("data/esci_corpus/products.parquet -> ejecuta: python build_esci_corpus.py")
    if not (ESCI_CORPUS_DIR / "embeddings.npz").exists():
        errors.append("data/esci_corpus/embeddings.npz -> ejecuta: python build_esci_corpus.py")
    if not (ESCI_CORPUS_DIR / "faiss.index").exists():
        errors.append("data/esci_corpus/faiss.index -> ejecuta: python build_esci_corpus.py")
    if not SYSTEM_CACHE.exists():
        errors.append("data/cache/unified_system_v2.pkl -> ejecuta: python main.py init")
    if errors:
        print("\n  [ERR] Prerequisitos faltantes:")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    print("  [OK] Prerequisitos OK")


# ---------------------------------------------------------------------
# Backup del sistema original
# ---------------------------------------------------------------------
def backup_system():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = CACHE_DIR / f"system_backup_{ts}.pkl"
    shutil.copy2(SYSTEM_CACHE, backup_path)
    logger.info(f"  [OK] Backup: {backup_path}")

    # Backup de embeddings también
    if EMB_CACHE.exists():
        emb_backup = CACHE_DIR / f"product_embeddings_backup_{ts}.npz"
        shutil.copy2(EMB_CACHE, emb_backup)
        logger.info(f"  [OK] Backup embeddings: {emb_backup}")

    return backup_path


# ---------------------------------------------------------------------
# Carga del corpus ESCI
# ---------------------------------------------------------------------
def load_esci_corpus():
    import pandas as pd

    logger.info("Cargando corpus ESCI...")

    df = pd.read_parquet(ESCI_CORPUS_DIR / "products.parquet")
    logger.info(f"  Productos: {len(df):,}")

    data = np.load(ESCI_CORPUS_DIR / "embeddings.npz", allow_pickle=True)
    embeddings = data['embeddings'].astype(np.float32)
    ids        = data['ids'].tolist()
    logger.info(f"  Embeddings: {embeddings.shape}")

    with open(ESCI_CORPUS_DIR / "corpus_manifest.json") as f:
        manifest = json.load(f)
    logger.info(f"  Manifest: {manifest['n_products']:,} productos")

    return df, embeddings, ids, manifest


# ---------------------------------------------------------------------
# Construcción de objetos de producto
# ---------------------------------------------------------------------
def build_product_objects(df, system):
    """
    Construye objetos de producto compatibles con el sistema existente.
    Usa el producto ejemplo del sistema para determinar la clase correcta.
    """
    logger.info(f"Construyendo {len(df):,} objetos de producto...")

    # Detectar clase del producto existente
    if system.canonical_products:
        example = system.canonical_products[0]
        product_class = type(example)
        example_attrs = vars(example) if hasattr(example, '__dict__') else {}
        logger.info(f"  Clase detectada: {product_class.__name__}")
        logger.info(f"  Atributos: {list(example_attrs.keys())[:10]}")
    else:
        product_class = None
        example_attrs = {}

    products = []
    for _, row in df.iterrows():
        asin  = str(row['asin'])
        title = str(row.get('product_title', row.get('title', asin)) or asin)
        text  = str(row.get('embedding_text', title))
        brand = str(row.get('product_brand', '') or '')
        desc  = str(row.get('product_description', text) or text)

        prod = _make_product(
            product_class, example_attrs, asin, title, text, brand, desc
        )
        products.append(prod)

    logger.info(f"  [OK] {len(products):,} objetos construidos")
    return products


def _make_product(product_class, example_attrs: dict,
                  asin: str, title: str, text: str, brand: str, desc: str):
    """
    Crea un objeto de producto con todos los atributos necesarios.
    Intenta múltiples estrategias para compatibilidad máxima.
    """
    # Mapa de valores por atributo
    value_map = {
        'id': asin, 'product_id': asin, 'asin': asin,
        'title': title, 'name': title, 'product_title': title,
        'text': text, 'embedding_text': text,
        'description': desc, 'product_description': desc,
        'brand': brand, 'product_brand': brand,
        'price': 0.0, 'rating': 0.0, 'reviews': 0,
        'category': 'General', 'source': 'esci_corpus',
        'product_bullet_point': '', 'product_color': '',
        'product_locale': 'us', 'small_version': 1, 'large_version': 1,
    }

    # Estrategia 1: instanciar con __new__ y setear atributos
    if product_class is not None:
        try:
            prod = object.__new__(product_class)
            # Copiar atributos del ejemplo, luego sobrescribir con nuestros valores
            for attr, example_val in example_attrs.items():
                if attr in value_map:
                    setattr(prod, attr, value_map[attr])
                elif isinstance(example_val, str):
                    setattr(prod, attr, '')
                elif isinstance(example_val, (int, float)):
                    setattr(prod, attr, 0)
                elif isinstance(example_val, list):
                    setattr(prod, attr, [])
                elif isinstance(example_val, dict):
                    setattr(prod, attr, {})
                elif example_val is None:
                    setattr(prod, attr, None)
                else:
                    setattr(prod, attr, example_val)
            # Asegurar atributos clave
            for attr, val in value_map.items():
                if not hasattr(prod, attr):
                    setattr(prod, attr, val)
            return prod
        except Exception as e:
            logger.debug(f"Estrategia 1 falló para {asin}: {e}")

    # Estrategia 2: objeto genérico compatible
    class ESCIProduct:
        __slots__ = []  # Permitir atributos dinámicos

    prod = ESCIProduct.__new__(ESCIProduct)
    prod.__class__ = product_class if product_class else ESCIProduct
    for attr, val in value_map.items():
        object.__setattr__(prod, attr, val)
    return prod


# ---------------------------------------------------------------------
# Reemplazo del corpus en el sistema
# ---------------------------------------------------------------------
def replace_corpus_in_system(system, products, embeddings, ids):
    """
    Reemplaza el corpus del sistema con los productos ESCI.
    También reconstruye el índice FAISS.
    """
    n_before = len(system.canonical_products)
    logger.info(f"Reemplazando corpus: {n_before:,} -> {len(products):,} productos")

    # Reemplazar lista de productos
    system.canonical_products = products

    # Reconstruir índice FAISS
    logger.info("Reconstruyendo índice FAISS...")
    try:
        import faiss

        # Intentar también cargar el índice pre-construido
        esci_index_path = ESCI_CORPUS_DIR / "faiss.index"
        if esci_index_path.exists():
            index = faiss.read_index(str(esci_index_path))
            logger.info(f"  [OK] Índice FAISS cargado: {index.ntotal:,} vectores")
        else:
            dim   = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)
            logger.info(f"  [OK] Índice FAISS construido: {index.ntotal:,} vectores")

        # Reemplazar en el sistema
        if hasattr(system, 'vector_store'):
            system.vector_store.index = index
            # Actualizar también el mapeo id->índice si existe
            if hasattr(system.vector_store, 'id_to_idx'):
                system.vector_store.id_to_idx = {str(pid): i for i, pid in enumerate(ids)}
            if hasattr(system.vector_store, 'idx_to_id'):
                system.vector_store.idx_to_id = {i: str(pid) for i, pid in enumerate(ids)}
            if hasattr(system.vector_store, 'products'):
                system.vector_store.products = products
        else:
            logger.warning("  [WARN] system.vector_store no encontrado — índice no reemplazado")

    except ImportError:
        logger.error("pip install faiss-cpu --break-system-packages")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reconstruyendo FAISS: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    return system


# ---------------------------------------------------------------------
# Guardar sistema actualizado
# ---------------------------------------------------------------------
def save_updated_system(system):
    """Guarda el sistema actualizado en el caché principal."""
    logger.info(f"Guardando sistema actualizado...")
    try:
        # Usar el método propio si existe
        if hasattr(system, 'save_to_cache'):
            system.save_to_cache()
        else:
            with open(SYSTEM_CACHE, 'wb') as f:
                pickle.dump(system, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"  [OK] {SYSTEM_CACHE}")
    except Exception as e:
        logger.error(f"Error guardando sistema: {e}")
        sys.exit(1)


def save_updated_embeddings(ids, embeddings):
    """
    Actualiza product_embeddings.npz que usa pretrain_reward_esci.py.
    """
    logger.info(f"Actualizando caché de embeddings...")
    np.savez_compressed(
        EMB_CACHE,
        ids        = np.array(ids),
        embeddings = embeddings.astype(np.float32),
    )
    logger.info(f"  [OK] {EMB_CACHE} ({len(ids):,} embeddings)")


# ---------------------------------------------------------------------
# Verificación post-instalación
# ---------------------------------------------------------------------
def verify_system(system, ids):
    """
    Verifica que el sistema funciona con el nuevo corpus.
    Hace una búsqueda de prueba.
    """
    logger.info("\nVerificando sistema...")
    try:
        # Verificar que el corpus tiene el tamaño correcto
        n = len(system.canonical_products)
        assert n == len(ids), f"Tamaño incorrecto: {n} vs {len(ids)}"
        logger.info(f"  [OK] Corpus: {n:,} productos")

        # Verificar acceso al id del primer producto
        p = system.canonical_products[0]
        pid = getattr(p, 'id', None) or getattr(p, 'product_id', None)
        assert pid is not None, "Producto sin ID"
        logger.info(f"  [OK] Primer producto: {pid}")

        # Hacer una búsqueda de prueba
        if hasattr(system, 'vector_store') and system.vector_store.index is not None:
            q_emb = np.random.randn(system.vector_store.index.d).astype(np.float32)
            q_emb /= np.linalg.norm(q_emb)
            results = system.vector_store.search(q_emb, k=5)
            assert len(results) > 0, "Búsqueda sin resultados"
            logger.info(f"  [OK] Búsqueda de prueba: {len(results)} resultados")

        logger.info("  [OK] Sistema verificado correctamente")
        return True

    except Exception as e:
        logger.error(f"  [ERR] Verificación falló: {e}")
        return False


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-backup', action='store_true',
                        help='Sin backup del sistema original')
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  INICIALIZACIÓN DEL SISTEMA CON CORPUS ESCI")
    print("=" * 65 + "\n")

    t0 = time.time()

    # 1. Prerrequisitos
    check_prerequisites()

    # 2. Cargar sistema existente
    logger.info("Cargando sistema existente...")
    from src.unified_system_v2 import UnifiedSystemV2
    system = UnifiedSystemV2.load_from_cache()
    logger.info(f"  Sistema original: {len(system.canonical_products):,} productos")

    # 3. Backup
    if not args.no_backup:
        logger.info("Creando backup...")
        backup_path = backup_system()
    else:
        backup_path = None
        logger.info("  [SKIP] Backup omitido")

    # 4. Cargar corpus ESCI
    df, embeddings, ids, manifest = load_esci_corpus()

    # 5. Construir objetos de producto
    products = build_product_objects(df, system)

    # 6. Reemplazar corpus en el sistema
    system = replace_corpus_in_system(system, products, embeddings, ids)

    # 7. Guardar sistema y embeddings
    save_updated_system(system)
    save_updated_embeddings(ids, embeddings)

    # 8. Verificar
    ok = verify_system(system, ids)

    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print("  SISTEMA INICIALIZADO CON CORPUS ESCI")
    print("=" * 65)
    print(f"  Corpus original:     {manifest.get('n_products', '?'):,} -> reemplazado")
    print(f"  Corpus nuevo:        {len(products):,} productos de ESCI")
    print(f"  Intersección ESCI:   100% (todo el corpus está en ESCI)")
    print(f"  Verificación:        {'[OK] OK' if ok else '[ERR] FALLÓ'}")
    if backup_path:
        print(f"  Backup:              {backup_path}")
    print(f"  Tiempo:              {elapsed:.0f}s")

    if not ok:
        print(f"\n  [ERR] Verificación falló. Revisar los logs.")
        if backup_path:
            print(f"  Para revertir: cp {backup_path} {SYSTEM_CACHE}")
        sys.exit(1)

    # Leer proyección de pares
    proj = manifest.get('training_projection', {})
    if proj:
        print(f"\n  PROYECCIÓN DE TRAINING:")
        print(f"    Pares estimados:     {proj.get('estimated_pairs', '?'):,}")
        print(f"    Queries con pares:   {proj.get('queries_with_pairs', '?'):,}")

    print(f"\n  SIGUIENTE PASO:")
    print(f"    python pretrain_reward_esci.py")
    print(f"    python build_esci_ground_truth.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()