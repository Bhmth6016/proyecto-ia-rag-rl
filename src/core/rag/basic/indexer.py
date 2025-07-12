# src/core/rag/basic/indexer.py

"""
Unified indexer that ingests raw JSON/JSONL Amazon products and
builds either a Chroma or FAISS vector-store with category and
filter-aware metadata.

The indexer now works with the domain Product model
(src.core.data.product.Product) so downstream code (CategoryTree,
ProductFilter, RLHF, etc.) stay consistent.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.core.utils.logger import get_logger
from src.core.data.product import Product  # ← domain model

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Public utilities
# ------------------------------------------------------------------
class Indexer:
    """
    Build / overwrite vector-stores from raw product files.

    Supports:
    • JSONL  (one product per line)
    • JSON   (list of products)
    • Chroma or FAISS back-end
    • Batch processing & memory-friendly
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vectorstore_backend: str = "chroma",
        device: str = "cpu",
    ):
        if vectorstore_backend not in {"chroma", "faiss"}:
            raise ValueError("vectorstore_backend must be 'chroma' or 'faiss'")

        self.embedder = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.backend = vectorstore_backend

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------
    def index_products(
        self,
        products: List[Union[dict, Product]],
        persist_dir: Union[str, Path],
        batch_size: int = 1_000,
    ) -> None:
        """
        Build or overwrite the vector-store.

        Args
        ----
        products     : list[dict] OR list[Product]
        persist_dir  : directory where the store will be written
        batch_size   : #docs to embed at once (memory control)
        """
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Normalise to list[Product]
        validated = []
        for p in products:
            if isinstance(p, Product):
                validated.append(p)
            elif isinstance(p, dict):
                prod = Product.from_dict(p)
                if prod:
                    validated.append(prod)
                else:
                    logger.warning("Skipping invalid product: %s", p.get("title"))
            else:
                logger.warning("Skipping unknown type: %s", type(p))

        if not validated:
            raise ValueError("No valid products to index.")

        docs = [Document(**p.to_document()) for p in validated]
        logger.info("Indexing %d products with %s…", len(docs), self.backend.upper())

        if self.backend == "chroma":
            self._build_chroma(docs, persist_dir, batch_size)
        else:
            self._build_faiss(docs, persist_dir, batch_size)

    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: Union[str, Path],
        persist_dir: Union[str, Path],
        max_products: Optional[int] = None,
        **indexer_kwargs,
    ) -> "Indexer":
        """
        Factory that reads a JSONL and runs the indexing pipeline in one call.

        Returns the Indexer instance for inspection (e.g. `.embedder`).
        """
        products = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_products and idx >= max_products:
                    break
                try:
                    products.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Invalid line %d: %s", idx, e)
                    continue

        indexer = cls(**indexer_kwargs)
        indexer.index_products(products, persist_dir)
        return indexer

    # ------------------------------------------------------------------
    # Back-end specific builders
    # ------------------------------------------------------------------
    def _build_chroma(
        self,
        docs: List[Document],
        persist_dir: Path,
        batch_size: int,
    ) -> None:
        texts = [d.page_content for d in docs]
        metas = [d.metadata for d in docs]

        embeddings = []
        for i in range(0, len(texts), batch_size):
            embeddings.extend(self.embedder.embed_documents(texts[i : i + batch_size]))

        Chroma.from_texts(
            texts=texts,
            embedding=self.embedder,
            metadatas=metas,
            ids=[str(i) for i in range(len(texts))],
            persist_directory=str(persist_dir),
        )
        logger.info("Chroma index saved at %s", persist_dir)

    def _build_faiss(
        self,
        docs: List[Document],
        persist_dir: Path,
        batch_size: int,
    ) -> None:
        texts = [d.page_content for d in docs]

        embeddings = []
        for i in range(0, len(texts), batch_size):
            embeddings.extend(self.embedder.embed_documents(texts[i : i + batch_size]))

        arr = np.array(embeddings).astype(np.float32)
        dim = arr.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(arr)

        faiss.write_index(index, str(persist_dir / "faiss.index"))
        with open(persist_dir / "docs.json", "w", encoding="utf-8") as f:
            json.dump([{"content": d.page_content, "metadata": d.metadata} for d in docs], f)
        logger.info("FAISS index saved at %s", persist_dir)


# ------------------------------------------------------------------
# Thin CLI for `python -m src.core.data.indexer`
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index Amazon products")
    parser.add_argument("jsonl", help="Path to products.jsonl")
    parser.add_argument("-o", "--out", default="chroma_index", help="Output folder")
    parser.add_argument("-b", "--backend", choices=("chroma", "faiss"), default="chroma")
    parser.add_argument("-n", "--max", type=int, help="Max products to index")
    args = parser.parse_args()

    Indexer.from_jsonl(args.jsonl, args.out, backend=args.backend, max_products=args.max)