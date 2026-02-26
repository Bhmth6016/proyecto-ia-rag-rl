"""
build_esci_corpus.py
====================
CAMINO 2 — Construir corpus desde productos ESCI.

PROBLEMA RAÍZ:
    Tu corpus original (89,990 productos) y ESCI (982,641 ASINs)
    son universos casi disjuntos: solo 1,330 en común (0.14%).
    No hay forma de entrenar el reward si no hay intersección.

SOLUCIÓN:
    Construir un corpus nuevo usando directamente los productos de ESCI.
    ESCI tiene título, descripción, bullets -> suficiente para embeddings.
    Intersección resultante: 100% (todo el corpus existe en ESCI).

ESTRATEGIA DE CONSTRUCCIÓN:
    1. Tomar ESCI train (1.4M filas)
    2. Filtrar locale='us'
    3. Deduplicar por ASIN
    4. Filtrar: solo ASINs que aparecen con esci_label='Exact' en ≥1 query
       -> Garantiza que el corpus tiene productos realmente relevantes
    5. Añadir también Substitute y Complement (cobertura)
    6. Tomar los N más frecuentes (más señal de training)
    7. Generar embeddings con el mismo modelo del sistema
    8. Construir índice FAISS
    9. Guardar como nuevo corpus

SEPARACIÓN LIMPIA (intocable):
    ESCI train -> corpus + training del reward       [OK] (este script)
    ESCI test  -> evaluación exclusivamente          <- NO tocar

PARÁMETROS:
    --size 100000    ASINs en el corpus nuevo (default: 100,000)
    --min-queries 1  mínimo de queries donde aparece el ASIN (default: 1)
    --batch 256      batch size para embeddings (default: 256)
    --dry-run        solo reportar, no crear nada

Uso:
    python build_esci_corpus.py
    python build_esci_corpus.py --size 200000
    python build_esci_corpus.py --dry-run

Salidas:
    data/esci_corpus/products.parquet       <- tabla de productos
    data/esci_corpus/embeddings.npz         <- embeddings
    data/esci_corpus/faiss.index            <- índice FAISS
    data/esci_corpus/corpus_manifest.json   <- metadatos
    data/esci_corpus/asins.txt              <- lista de ASINs
"""
import argparse
import json
import logging
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

ESCI_LABEL_SCORE = {"Exact": 3, "Substitute": 2, "Complement": 1, "Irrelevant": 0}
OUTPUT_DIR = Path("data/esci_corpus")


# ---------------------------------------------------------------------
# Carga y filtrado de ESCI
# ---------------------------------------------------------------------
def load_esci_train() -> "pd.DataFrame":
    import pandas as pd
    train_path = Path("data/esci/esci_train.parquet")
    if not train_path.exists():
        logger.error("Ejecuta primero: python verify_and_split_esci.py")
        sys.exit(1)
    logger.info(f"Cargando ESCI train...")
    df = pd.read_parquet(train_path)
    # Normalizar nombre de columna ASIN
    if 'product_id' in df.columns and 'asin' not in df.columns:
        df = df.rename(columns={'product_id': 'asin'})
    logger.info(f"  {len(df):,} filas, {df['asin'].nunique():,} ASINs únicos")
    return df


def select_corpus_asins(df, target_size: int, min_queries: int = 1) -> "pd.DataFrame":
    """
    Selecciona los mejores ASINs para el corpus nuevo.

    Priorización:
        1. ASINs con max_score=Exact (score=3) — siempre relevantes
        2. ASINs con max_score=Substitute (score=2)
        3. Dentro de cada nivel: los que aparecen en más queries
           (más útiles para generar pares de entrenamiento)
        4. Límite: target_size ASINs

    Garantía: todo ASIN en el corpus estará en ESCI train
    -> intersección = 100%
    """
    import pandas as pd

    logger.info(f"\nSeleccionando {target_size:,} ASINs para corpus nuevo...")

    df['score'] = df['esci_label'].map(ESCI_LABEL_SCORE)

    # Estadísticas por ASIN
    asin_stats = df.groupby('asin').agg(
        max_score   = ('score', 'max'),
        mean_score  = ('score', 'mean'),
        n_queries   = ('query', 'nunique'),
        n_exact     = ('score', lambda x: (x == 3).sum()),
        n_substitute= ('score', lambda x: (x == 2).sum()),
        n_irrelevant= ('score', lambda x: (x == 0).sum()),
    ).reset_index()

    # Filtro mínimo de queries
    if min_queries > 1:
        n_before = len(asin_stats)
        asin_stats = asin_stats[asin_stats['n_queries'] >= min_queries]
        logger.info(f"  Filtro min_queries≥{min_queries}: {n_before:,} -> {len(asin_stats):,}")

    # Ordenar: max_score desc, n_queries desc, n_exact desc
    asin_stats = asin_stats.sort_values(
        ['max_score', 'n_queries', 'n_exact'],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    # Seleccionar top N
    selected = asin_stats.head(target_size).copy()

    # Log distribución
    n_exact   = (selected['max_score'] == 3).sum()
    n_sub     = (selected['max_score'] == 2).sum()
    n_comp    = (selected['max_score'] == 1).sum()
    n_irrel   = (selected['max_score'] == 0).sum()
    avg_q     = selected['n_queries'].mean()

    logger.info(f"\n  ASINs seleccionados: {len(selected):,}")
    logger.info(f"    con Exact en alguna query:     {n_exact:,} ({n_exact/len(selected)*100:.1f}%)")
    logger.info(f"    con Substitute en alguna query:{n_sub:,} ({n_sub/len(selected)*100:.1f}%)")
    logger.info(f"    con Complement en alguna query:{n_comp:,} ({n_comp/len(selected)*100:.1f}%)")
    logger.info(f"    solo Irrelevant:               {n_irrel:,} ({n_irrel/len(selected)*100:.1f}%)")
    logger.info(f"    media queries por ASIN:        {avg_q:.1f}")

    return selected


def build_product_table(df_train, selected_asins: "pd.DataFrame") -> "pd.DataFrame":
    """
    Construye tabla de productos con toda la información disponible.
    Un ASIN puede aparecer en múltiples filas (queries distintas):
    tomamos la primera ocurrencia para los campos de producto.
    """
    import pandas as pd

    asin_set = set(selected_asins['asin'].tolist())

    # Columnas de producto (no de query)
    product_cols = [
        'asin', 'product_title', 'product_description',
        'product_bullet_point', 'product_brand', 'product_color',
        'product_text',  # campo combinado si existe
    ]
    available_cols = ['asin'] + [c for c in product_cols[1:] if c in df_train.columns]

    df_products = (
        df_train[df_train['asin'].isin(asin_set)][available_cols]
        .drop_duplicates('asin')
        .merge(selected_asins[['asin', 'max_score', 'n_queries', 'n_exact']], on='asin', how='left')
        .reset_index(drop=True)
    )

    # Construir campo de texto combinado para embedding
    def make_text(row):
        parts = []
        title = str(row.get('product_title', '') or '').strip()
        brand = str(row.get('product_brand', '') or '').strip()
        color = str(row.get('product_color', '') or '').strip()
        bullets = str(row.get('product_bullet_point', '') or '').strip()
        desc = str(row.get('product_description', '') or '').strip()
        text = str(row.get('product_text', '') or '').strip()

        if title:   parts.append(title)
        if brand:   parts.append(brand)
        if color:   parts.append(color)
        if bullets: parts.append(bullets.split('\n')[0][:300])
        if desc and not text:
            parts.append(desc[:400])
        if text and not desc:
            parts.append(text[:400])
        return ' | '.join(p for p in parts if p) or f"Product {row['asin']}"

    df_products['embedding_text'] = df_products.apply(make_text, axis=1)
    df_products['title']          = df_products.get('product_title', df_products['asin'])

    logger.info(f"Tabla de productos: {len(df_products):,} ASINs con texto")
    return df_products


# ---------------------------------------------------------------------
# Generación de embeddings
# ---------------------------------------------------------------------
def generate_embeddings(df_products, batch_size: int = 256) -> np.ndarray:
    """
    Genera embeddings usando el mismo modelo que usa el sistema.
    Carga el modelo desde el sistema existente para garantizar compatibilidad.
    """
    logger.info(f"\nCargando modelo de embeddings del sistema...")
    emb_model = _load_embedding_model()

    texts = df_products['embedding_text'].tolist()
    logger.info(f"Generando embeddings para {len(texts):,} productos...")

    all_embs = []
    t0 = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embs  = emb_model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embs.append(embs)

        if i % (batch_size * 10) == 0:
            elapsed = time.time() - t0
            pct = (i + len(batch)) / len(texts) * 100
            eta = elapsed / max(i + len(batch), 1) * (len(texts) - i - len(batch))
            logger.info(f"  [{i+len(batch):,}/{len(texts):,}] {pct:.1f}% — ETA: {eta:.0f}s")

    embeddings = np.vstack(all_embs).astype(np.float32)
    logger.info(f"  [OK] Embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
    return embeddings


def _load_embedding_model():
    """
    Carga el modelo de embeddings del sistema existente.
    Intenta distintas formas de acceder al modelo.
    """
    # Intento 1: desde el sistema cargado
    try:
        from src.unified_system_v2 import UnifiedSystemV2
        system = UnifiedSystemV2.load_from_cache()
        model = system.canonicalizer.embedding_model
        logger.info(f"  [OK] Modelo cargado desde sistema: {type(model).__name__}")
        return model
    except Exception as e:
        logger.debug(f"  Intento 1 falló: {e}")

    # Intento 2: desde el pipeline RLHF
    try:
        from src.unified_system_v2 import UnifiedSystemV2
        from src.rlhf_integration import add_rlhf_to_system
        system   = UnifiedSystemV2.load_from_cache()
        pipeline = add_rlhf_to_system(system)
        model    = pipeline.emb_model
        logger.info(f"  [OK] Modelo cargado desde pipeline: {type(model).__name__}")
        return model
    except Exception as e:
        logger.debug(f"  Intento 2 falló: {e}")

    # Intento 3: cargar directamente sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        # Detectar el modelo usado por el sistema
        model_name = _detect_model_name()
        model = SentenceTransformer(model_name)
        logger.info(f"  [OK] Modelo cargado directamente: {model_name}")
        return model
    except Exception as e:
        logger.error(f"No se pudo cargar el modelo de embeddings: {e}")
        logger.error("Instala: pip install sentence-transformers --break-system-packages")
        sys.exit(1)


def _detect_model_name() -> str:
    """
    Intenta detectar el nombre del modelo usado por el sistema
    buscando en archivos de configuración o usando el default.
    """
    config_paths = [
        Path("src/config.py"),
        Path("config.py"),
        Path("src/unified_system_v2.py"),
    ]
    for p in config_paths:
        if p.exists():
            content = p.read_text(encoding='utf-8', errors='ignore')
            for line in content.split('\n'):
                if 'sentence' in line.lower() and ('model' in line.lower() or 'paraphrase' in line.lower()):
                    for candidate in [
                        'paraphrase-multilingual-MiniLM-L12-v2',
                        'all-MiniLM-L6-v2',
                        'all-mpnet-base-v2',
                        'multi-qa-MiniLM-L6-cos-v1',
                    ]:
                        if candidate in line:
                            return candidate

    # El sistema usa embeddings de dim=384 -> MiniLM
    logger.warning("No se detectó el modelo. Usando paraphrase-multilingual-MiniLM-L12-v2 (dim=384).")
    return 'paraphrase-multilingual-MiniLM-L12-v2'


# ---------------------------------------------------------------------
# Construcción del índice FAISS
# ---------------------------------------------------------------------
def build_faiss_index(embeddings: np.ndarray, output_dir: Path):
    """
    Construye y guarda el índice FAISS.
    Usa IndexFlatIP (producto interior) = coseno sobre vectores normalizados.
    """
    try:
        import faiss
    except ImportError:
        logger.error("pip install faiss-cpu --break-system-packages")
        sys.exit(1)

    logger.info(f"\nConstruyendo índice FAISS ({embeddings.shape[0]:,} vectores, dim={embeddings.shape[1]})...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"  [OK] Índice: {index.ntotal:,} vectores")

    index_path = output_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    logger.info(f"  [OK] Guardado: {index_path}")
    return index


# ---------------------------------------------------------------------
# Guardar corpus
# ---------------------------------------------------------------------
def save_corpus(df_products, embeddings: np.ndarray, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tabla de productos
    products_path = output_dir / "products.parquet"
    df_products.to_parquet(products_path, index=False)
    logger.info(f"  [OK] Productos: {products_path} ({len(df_products):,} filas)")

    # 2. Embeddings
    emb_path = output_dir / "embeddings.npz"
    np.savez_compressed(
        emb_path,
        ids        = np.array(df_products['asin'].tolist()),
        embeddings = embeddings,
    )
    logger.info(f"  [OK] Embeddings: {emb_path} ({embeddings.shape})")

    # 3. Lista de ASINs
    asins_path = output_dir / "asins.txt"
    asins_path.write_text('\n'.join(df_products['asin'].tolist()))
    logger.info(f"  [OK] ASINs: {asins_path}")

    # 4. Manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'n_products': len(df_products),
        'embedding_dim': int(embeddings.shape[1]),
        'source': 'esci_train',
        'split': 'train_only',
        'test_split_preserved': True,
        'columns': list(df_products.columns),
        'score_distribution': {
            'max_score_3_exact': int((df_products['max_score'] == 3).sum()),
            'max_score_2_substitute': int((df_products['max_score'] == 2).sum()),
            'max_score_1_complement': int((df_products['max_score'] == 1).sum()),
            'max_score_0_irrelevant': int((df_products['max_score'] == 0).sum()),
        },
    }
    manifest_path = output_dir / "corpus_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  [OK] Manifest: {manifest_path}")


# ---------------------------------------------------------------------
# Verificación: cuántos pares de training tendremos
# ---------------------------------------------------------------------
def project_training_pairs(df_train, corpus_asins: set) -> dict:
    """
    Proyecta cuántos pares de entrenamiento tendremos con el nuevo corpus.
    """
    df = df_train.copy()
    if 'product_id' in df.columns and 'asin' not in df.columns:
        df = df.rename(columns={'product_id': 'asin'})
    df['score'] = df['esci_label'].map(ESCI_LABEL_SCORE)

    df_in = df[df['asin'].isin(corpus_asins)].copy()
    n_common = df_in['asin'].nunique()

    pair_count = 0
    queries_with_pairs = 0
    for _, gdf in df_in.groupby('query'):
        scores = sorted(gdf['score'].unique().tolist(), reverse=True)
        if len(scores) >= 2:
            has_multi = any(s > scores[-1] for s in scores)
            if has_multi:
                queries_with_pairs += 1
                by_score = gdf.groupby('score')['asin'].count().to_dict()
                for s_h in scores:
                    for s_l in scores:
                        if s_h > s_l and (s_h - s_l) >= 1:
                            n_pairs = by_score.get(s_h, 0) * by_score.get(s_l, 0)
                            pair_count += min(20, n_pairs)

    return {
        'corpus_size': len(corpus_asins),
        'esci_rows_in_corpus': len(df_in),
        'common_asins': n_common,
        'queries_with_pairs': queries_with_pairs,
        'estimated_pairs': pair_count,
        'avg_pairs_per_query': pair_count / max(queries_with_pairs, 1),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size',        type=int, default=100_000,
                        help='Número de ASINs en el corpus nuevo (default: 100,000)')
    parser.add_argument('--min-queries', type=int, default=1,
                        help='Mínimo de queries donde aparece el ASIN (default: 1)')
    parser.add_argument('--batch',       type=int, default=256,
                        help='Batch size para embeddings (default: 256)')
    parser.add_argument('--dry-run',     action='store_true',
                        help='Solo reportar, no crear archivos')
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  CAMINO 2 — CORPUS DESDE PRODUCTOS ESCI")
    print("=" * 65)
    print(f"\n  Tamaño objetivo:    {args.size:,} ASINs")
    print(f"  Min queries/ASIN:   {args.min_queries}")
    print(f"  Intersección:       100% (todo el corpus estará en ESCI)")
    print(f"  Split preservado:   ESCI test intocable\n")

    t0 = time.time()

    # 1. Cargar ESCI train
    df_train = load_esci_train()

    # 2. Seleccionar ASINs
    selected = select_corpus_asins(df_train, args.size, args.min_queries)

    # 3. Tabla de productos
    df_products = build_product_table(df_train, selected)

    # Verificar que tenemos texto válido
    n_empty = (df_products['embedding_text'].str.len() < 5).sum()
    if n_empty > 0:
        logger.warning(f"  {n_empty:,} productos con texto casi vacío — se incluyen de todas formas")

    # 4. Proyección de pares
    corpus_asins = set(df_products['asin'].tolist())
    proj = project_training_pairs(df_train, corpus_asins)

    print("\n" + "-" * 65)
    print("  PROYECCIÓN DE TRAINING CON EL NUEVO CORPUS")
    print("-" * 65)
    print(f"  Corpus size:                 {proj['corpus_size']:,}")
    print(f"  Filas ESCI en corpus:        {proj['esci_rows_in_corpus']:,}")
    print(f"  ASINs en común:              {proj['common_asins']:,} (≈100%)")
    print(f"  Queries con pares válidos:   {proj['queries_with_pairs']:,}")
    print(f"  Pares estimados:             {proj['estimated_pairs']:,}")
    print(f"  Media pares/query:           {proj['avg_pairs_per_query']:.1f}")
    print("-" * 65)

    if args.dry_run:
        print("\n  DRY-RUN: no se creó ningún archivo.")
        print(f"  Ejecuta sin --dry-run para construir el corpus.")
        return

    # 5. Verificar si ya existe
    manifest_path = OUTPUT_DIR / "corpus_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            existing = json.load(f)
        print(f"\n  [WARN] Corpus existente encontrado ({existing.get('n_products', '?'):,} productos).")
        resp = input("  ¿Sobreescribir? [s/N] ").strip().lower()
        if resp not in ('s', 'si', 'sí', 'y', 'yes'):
            print("  Cancelado.")
            return

    # 6. Generar embeddings
    embeddings = generate_embeddings(df_products, args.batch)

    # 7. Guardar corpus
    print("\nGuardando corpus...")
    save_corpus(df_products, embeddings, OUTPUT_DIR)

    # 8. Construir índice FAISS
    build_faiss_index(embeddings, OUTPUT_DIR)

    # 9. Manifest final con proyección
    manifest_path = OUTPUT_DIR / "corpus_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest['training_projection'] = proj
    manifest['elapsed_seconds'] = time.time() - t0
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print("  CORPUS ESCI CONSTRUIDO")
    print("=" * 65)
    print(f"  Productos:          {len(df_products):,}")
    print(f"  Embedding dim:      {embeddings.shape[1]}")
    print(f"  Pares estimados:    {proj['estimated_pairs']:,}")
    print(f"  Tiempo:             {elapsed:.0f}s")
    print(f"\n  Archivos:")
    print(f"    {OUTPUT_DIR}/products.parquet")
    print(f"    {OUTPUT_DIR}/embeddings.npz")
    print(f"    {OUTPUT_DIR}/faiss.index")
    print(f"    {OUTPUT_DIR}/corpus_manifest.json")
    print(f"\n  SIGUIENTE PASO:")
    print(f"    python init_esci_system.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()