# demo/pipeline.py
"""
End-to-end data pipeline:

1.  Load raw JSONL files
2.  Validate & normalise → `Product` objects
3.  Export clean JSON (optional)
4.  Build Chroma or FAISS index
5.  Cache intermediate results

Usage
-----
$ python -m src.core.data.pipeline
$ python -m src.core.data.pipeline --backend faiss --clear-cache
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

from tqdm import tqdm

from src.core.config import settings
from src.core.data.loader import DataLoader
from src.core.data.product import Product
from src.core.rag.basic.indexer import Indexer
from src.core.retrieval.retriever import Retriever
from src.core.utils.logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Pipeline orchestrator
# ------------------------------------------------------------------
class Pipeline:
    def __init__(
        self,
        *,
        raw_dir: Path = settings.RAW_DIR,
        processed_dir: Path = settings.PROC_DIR,
        index_dir: Path = settings.VEC_DIR / settings.INDEX_NAME,
        backend: str = settings.VECTOR_BACKEND,
        device: str = settings.RLHF_CONFIG.get("device", "cpu"),
    ):
        self.loader = DataLoader(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            cache_enabled=settings.CACHE_ENABLED,
        )
        self.index_dir = Path(index_dir)
        self.backend = backend.lower()
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, *, clear_cache: bool = False) -> int:
        """Execute the full pipeline and return #indexed docs."""
        if clear_cache:
            self.loader.clear_cache()

        products = self.loader.load_data(use_cache=True)
        if not products:
            logger.error("No products to process.")
            return 0

        logger.info("Indexing %d products with %s...", len(products), self.backend)
        indexer = Retriever(
            index_path=self.index_dir,
            vectorstore_type=self.backend,
            device=self.device,
        )
        # index creation is handled internally in Retriever
        return len(products)

    def export_clean_json(self, outfile: str | Path) -> None:
        """Write the **validated** product list to a single JSON file."""
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)

        products = self.loader.load_data(use_cache=True)
        with outfile.open("w", encoding="utf-8") as f:
            json.dump(
                [p.dict() for p in tqdm(products, desc="Exporting JSON")],
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        logger.info("Exported %d products to %s", len(products), outfile)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build vector index from raw JSONL")
    parser.add_argument("--backend", choices=("chroma", "faiss"), default=settings.VECTOR_BACKEND)
    parser.add_argument("--clear-cache", action="store_true", help="Delete existing pickles")
    parser.add_argument("--export-json", type=Path, help="Write clean JSON snapshot")
    parser.add_argument("--device", default=settings.RLHF_CONFIG.get("device", "cpu"), help="CPU/CUDA for embeddings")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)

    from src.core.utils.logger import configure_root_logger

    configure_root_logger(
        level=logging.DEBUG if args.verbose else logging.INFO,
        module_levels={"urllib3": logging.WARNING, "transformers": logging.WARNING},
    )

    pipeline = Pipeline(backend=args.backend, device=args.device)

    if args.export_json:
        pipeline.export_clean_json(args.export_json)

    total = pipeline.run(clear_cache=args.clear_cache)
    print(f"\n✅ Pipeline complete – {total} products indexed with {args.backend}.\n")


if __name__ == "__main__":
    main()
