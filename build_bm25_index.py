"""
build_bm25_index.py
Construye índice BM25 sobre títulos de productos del corpus.
Guarda en data/cache/bm25_index.pkl
"""
import pickle, sys, logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def tokenize(text: str):
    """Tokenización simple: lowercase + split"""
    return text.lower().split() if text else []

def build_bm25(system):
    from rank_bm25 import BM25Okapi
    
    products = getattr(system, 'canonical_products', [])
    logger.info(f"Indexando {len(products):,} productos...")
    
    corpus, pids = [], []
    for p in products:
        pid = str(getattr(p, 'id', '') or getattr(p, 'product_id', ''))
        title = getattr(p, 'title', '') or ''
        if pid and title:
            corpus.append(tokenize(title))
            pids.append(pid)
    
    logger.info(f"Construyendo BM25Okapi sobre {len(corpus):,} documentos...")
    bm25 = BM25Okapi(corpus)
    
    out = Path("data/cache/bm25_index.pkl")
    out.parent.mkdir(exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump({'bm25': bm25, 'pids': pids}, f)
    
    logger.info(f"Guardado: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return bm25, pids

if __name__ == "__main__":
    from src.unified_system_v2 import UnifiedSystemV2
    system = UnifiedSystemV2.load_from_cache()
    bm25, pids = build_bm25(system)
    
    # Test rápido
    test_query = "laptop computer"
    tokens = tokenize(test_query)
    scores = bm25.get_scores(tokens)
    top5 = np.argsort(scores)[::-1][:5]
    logger.info(f"\nTest '{test_query}':")
    for i in top5:
        logger.info(f"  {pids[i]}: score={scores[i]:.3f}")