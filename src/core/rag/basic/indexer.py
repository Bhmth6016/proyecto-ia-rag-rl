# src/core/rag/basic/indexer.py
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.core.utils.logger import get_logger
from src.core.data.product import AmazonProduct, products_to_documents, validate_product_data

logger = get_logger(__name__)

class Indexer:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vectorstore_type: str = "chroma",
        device: str = "cpu"
    ):
        self.embedder = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vectorstore_type = vectorstore_type.lower() 
        if self.vectorstore_type not in ["chroma", "faiss"]:
            raise ValueError("Vectorstore must be 'chroma' or 'faiss'")

    def build_index_from_products(
        self,
        products: List[Union[Dict, AmazonProduct]],
        persist_dir: Union[str, Path],
        batch_size: int = 1000
    ) -> None:
        """Main indexing method that handles both raw dicts and AmazonProduct objects"""
        persist_dir = Path(persist_dir)
        os.makedirs(persist_dir, exist_ok=True)
        
        # Convert to documents
        if isinstance(products[0], AmazonProduct):
            documents = [product.to_document() for product in products]
        else:
            documents = products_to_documents([p for p in products if validate_product_data(p)])
        
        logger.info(f"Indexing {len(documents)} documents...")
        
        if self.vectorstore_type == "chroma":
            self._build_chroma_index(documents, persist_dir, batch_size)
        else:
            self._build_faiss_index(documents, persist_dir, batch_size)

    def _build_chroma_index(
        self,
        documents: List[Document],
        persist_dir: Path,
        batch_size: int
    ) -> None:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Batch embeddings to avoid memory issues
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings.extend(self.embedder.embed_documents(batch))
        
        Chroma.from_texts(
            texts=texts,
            embedding=self.embedder,
            metadatas=metadatas,
            persist_directory=str(persist_dir)
        )
        logger.info(f"Chroma index built at {persist_dir}")

    def _build_faiss_index(
        self,
        documents: List[Document],
        persist_dir: Path,
        batch_size: int
    ) -> None:
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings in batches
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings.extend(self.embedder.embed_documents(batch))
        
        # Convert to numpy array
        embeddings_np = np.array(embeddings).astype('float32')
        dim = embeddings_np.shape[1]
        
        # Build and save index
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings_np)
        faiss.write_index(index, str(persist_dir / "faiss_index.index"))
        
        # Save documents with metadata
        self._save_faiss_documents(documents, persist_dir)
        logger.info(f"FAISS index built at {persist_dir}")

    def _save_faiss_documents(self, documents: List[Document], persist_dir: Path) -> None:
        docs_path = persist_dir / "faiss_documents.json"
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump([
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in documents
            ], f)

    @classmethod
    def from_product_jsonl(
        cls,
        jsonl_path: Union[str, Path],
        persist_dir: Union[str, Path],
        max_products: Optional[int] = None,
        **kwargs
    ) -> None:
        """Convenience method to index directly from JSONL file"""
        indexer = cls(**kwargs)
        
        # Load and validate products
        products = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_products and i >= max_products:
                    break
                try:
                    product = json.loads(line)
                    if validate_product_data(product):
                        products.append(product)
                except json.JSONDecodeError:
                    continue
        
        indexer.build_index_from_products(products, persist_dir)