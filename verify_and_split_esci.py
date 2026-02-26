"""
verify_and_split_esci.py
========================
Verifica la estructura del dataset ESCI y crea splits train/test limpios.

EJECUTAR PRIMERO — antes de cualquier otro script.

Salidas:
    data/esci/esci_train.parquet        <- para pretrain_reward_esci.py
    data/esci/esci_test.parquet         <- para evaluación ÚNICAMENTE
    data/esci/split_verification.json

Garantías:
    - Ninguna query del test aparece en train
    - La intersección de queries entre splits es CERO
    - Los splits son reproducibles (seed fijo)
"""
import json
import logging
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

ESCI_DIR       = Path("data/esci")
TEST_FRACTION  = 0.25
SEED           = 42


def main():
    print("\n" + "=" * 60)
    print("  VERIFICACIÓN Y SEPARACIÓN DE ESCI")
    print("=" * 60 + "\n")

    ESCI_DIR.mkdir(parents=True, exist_ok=True)

    train_path = ESCI_DIR / "esci_train.parquet"
    test_path  = ESCI_DIR / "esci_test.parquet"

    if train_path.exists() and test_path.exists():
        logger.info("Splits ya existen. Verificando integridad...")
        _verify_existing_splits(train_path, test_path)
        return

    try:
        from datasets import load_dataset
        import pandas as pd
    except ImportError:
        logger.error("pip install datasets pandas pyarrow")
        sys.exit(1)

    logger.info("Descargando ESCI desde HuggingFace...")
    try:
        ds = load_dataset("tasksource/esci")
        logger.info(f"Estructura del dataset: {ds}")

        if 'train' in ds and 'test' in ds:
            logger.info("[OK] ESCI tiene splits nativos train/test")
            df_train = ds['train'].to_pandas()
            df_test  = ds['test'].to_pandas()
            split_source = 'native_hf_splits'

        elif 'train' in ds:
            df_all = ds['train'].to_pandas()
            logger.info(f"Dataset completo: {len(df_all):,} filas")
            logger.info(f"Columnas: {list(df_all.columns)}")

            if 'split' in df_all.columns:
                splits_found = df_all['split'].unique().tolist()
                logger.info(f"Columna 'split' encontrada. Valores: {splits_found}")

                if 'train' in splits_found and 'test' in splits_found:
                    df_train = df_all[df_all['split'] == 'train'].copy()
                    df_test  = df_all[df_all['split'] == 'test'].copy()
                    split_source = 'internal_split_column'
                    logger.info(f"[OK] Split interno: train={len(df_train):,}, test={len(df_test):,}")
                else:
                    df_train, df_test = _manual_split_by_query(df_all)
                    split_source = 'manual_by_query'
            else:
                logger.warning("Sin columna 'split'. Haciendo split manual por query.")
                df_train, df_test = _manual_split_by_query(df_all)
                split_source = 'manual_by_query'
        else:
            key = list(ds.keys())[0]
            df_all = ds[key].to_pandas()
            df_train, df_test = _manual_split_by_query(df_all)
            split_source = 'manual_by_query'

    except Exception as e:
        logger.error(f"Error descargando ESCI: {e}")
        sys.exit(1)

    # Normalizar columnas
    for df in [df_train, df_test]:
        if 'product_id' in df.columns and 'asin' not in df.columns:
            df.rename(columns={'product_id': 'asin'}, inplace=True)

    # Filtrar locale inglés
    for name, df in [('train', df_train), ('test', df_test)]:
        for lc in ['product_locale', 'locale']:
            if lc in df.columns:
                n_before = len(df)
                filtered = df[df[lc] == 'us'].copy()
                if name == 'train':
                    df_train = filtered
                else:
                    df_test = filtered
                logger.info(f"  {name} locale='us': {n_before:,} -> {len(filtered):,}")
                break

    # VERIFICACIÓN CRÍTICA: sin overlap de queries
    queries_train = set(df_train['query'].astype(str).unique())
    queries_test  = set(df_test['query'].astype(str).unique())
    overlap       = queries_train & queries_test

    if overlap:
        logger.warning(f"[WARN] {len(overlap)} queries compartidas. Limpiando...")
        df_test       = df_test[~df_test['query'].astype(str).isin(overlap)].copy()
        queries_test  = set(df_test['query'].astype(str).unique())
        assert len(queries_train & queries_test) == 0
        logger.info("  [OK] Overlap removido")

    assert len(queries_train & queries_test) == 0, \
        "ERROR CRÍTICO: queries compartidas entre train y test"

    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)

    logger.info(f"\n  [OK] Train: {train_path} ({len(df_train):,} filas)")
    logger.info(f"  [OK] Test:  {test_path}  ({len(df_test):,} filas)")

    verification = {
        'timestamp': datetime.now().isoformat(),
        'split_source': split_source,
        'train_rows': len(df_train),
        'test_rows': len(df_test),
        'train_queries': len(queries_train),
        'test_queries': len(queries_test),
        'query_overlap': 0,
        'leakage_free': True,
        'test_fraction': len(queries_test) / (len(queries_train) + len(queries_test)),
    }
    with open(ESCI_DIR / "split_verification.json", 'w') as f:
        json.dump(verification, f, indent=2)

    print("\n" + "=" * 60)
    print("  SPLITS ESCI VERIFICADOS")
    print("=" * 60)
    print(f"  Método de split:     {split_source}")
    print(f"  Train filas:         {len(df_train):,}")
    print(f"  Test filas:          {len(df_test):,}")
    print(f"  Train queries:       {len(queries_train):,}")
    print(f"  Test queries:        {len(queries_test):,}")
    print(f"  Overlap de queries:  0  <- debe ser 0")
    print(f"  Leakage-free:        [OK] SÍ")
    print(f"\n  Siguiente paso:")
    print(f"    python pretrain_reward_esci.py")


def _manual_split_by_query(df, test_fraction=TEST_FRACTION, seed=SEED):
    import pandas as pd
    logger.info(f"  Split manual: {1-test_fraction:.0%} train / {test_fraction:.0%} test")
    queries = df['query'].astype(str).unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(queries)
    n_test        = int(len(queries) * test_fraction)
    test_queries  = set(queries[:n_test])
    train_queries = set(queries[n_test:])
    df_test  = df[df['query'].astype(str).isin(test_queries)].copy()
    df_train = df[df['query'].astype(str).isin(train_queries)].copy()
    logger.info(f"  Train: {len(train_queries):,} queries, {len(df_train):,} filas")
    logger.info(f"  Test:  {len(test_queries):,} queries, {len(df_test):,} filas")
    return df_train, df_test


def _verify_existing_splits(train_path, test_path):
    import pandas as pd
    df_train = pd.read_parquet(train_path)
    df_test  = pd.read_parquet(test_path)
    queries_train = set(df_train['query'].astype(str).unique())
    queries_test  = set(df_test['query'].astype(str).unique())
    overlap       = queries_train & queries_test

    print(f"\n  Train: {len(df_train):,} filas, {len(queries_train):,} queries")
    print(f"  Test:  {len(df_test):,} filas,  {len(queries_test):,} queries")
    print(f"  Overlap: {len(overlap)} queries")

    if overlap:
        print("  [WARN] LEAKAGE DETECTADO. Borra los splits y vuelve a correr.")
        sys.exit(1)
    else:
        print("  [OK] Sin leakage — splits limpios")
        print(f"\n  Siguiente paso:")
        print(f"    python pretrain_reward_esci.py")


if __name__ == "__main__":
    main()