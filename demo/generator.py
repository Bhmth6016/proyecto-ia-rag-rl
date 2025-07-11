# demo/generator.py
#!/usr/bin/env python3
"""
Data Generator and Indexing Pipeline

This script handles:
1. Conversion of raw JSONL to processed JSON
2. Cleaning and validation of product data
3. Generation of vector embeddings
4. Creation of ChromaDB indexes
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.core.data.loader import DataLoader
from src.core.utils.logger import get_logger
from src.core.utils.validators import validate_product

logger = get_logger(__name__)

class DataGenerator:
    def __init__(self, data_dir: str = None):
        self.base_dir = Path(__file__).parent.parent.parent.resolve()
        self.data_dir = Path(data_dir) if data_dir else (self.base_dir / "data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.index_dir = self.data_dir / "chroma_indexes"
        
        self._ensure_directories()
        self.embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def _ensure_directories(self):
        """Create required directories if they don't exist"""
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

    def _convert_jsonl_to_json(self, input_file: Path) -> Optional[Path]:
        """
        Convert JSONL file to formatted JSON with validation
        Returns path to output file
        """
        output_file = self.processed_dir / f"{input_file.stem}.json"
        valid_products = []
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in tqdm(lines, desc=f"Processing {input_file.name}"):
                try:
                    product = json.loads(line)
                    if validate_product(product):
                        valid_products.append(product)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON line in {input_file.name}")
                    continue
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(valid_products, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Converted {len(valid_products)} products to {output_file.name}")
            return output_file
            
        except Exception as e:
            logger.error(f"Failed to convert {input_file.name}: {str(e)}")
            return None

    def _product_to_document(self, product: Dict) -> Document:
        """Convert product dict to LangChain Document"""
        details = product.get('details', {})
        
        # Build page content
        content_parts = [
            f"Title: {product.get('title', 'N/A')}",
            f"Category: {product.get('main_category', 'N/A')}",
            f"Price: ${product.get('price', 'N/A')}",
            f"Rating: {product.get('average_rating', 'N/A')}"
        ]
        
        if details:
            content_parts.append("Details:")
            for k, v in details.items():
                content_parts.append(f"- {k}: {v}")
        
        # Build metadata
        metadata = {
            'title': product.get('title', ''),
            'category': product.get('main_category', ''),
            'price': product.get('price'),
            'rating': product.get('average_rating')
        }
        
        return Document(
            page_content="\n".join(content_parts),
            metadata=metadata
        )

    def _generate_index(self, json_file: Path) -> int:
        """Generate vector index from JSON file"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                products = json.load(f)
            
            documents = []
            invalid_count = 0
            
            for product in tqdm(products, desc="Creating documents"):
                try:
                    if validate_product(product):
                        documents.append(self._product_to_document(product))
                    else:
                        invalid_count += 1
                except Exception as e:
                    logger.warning(f"Invalid product skipped: {str(e)}")
                    invalid_count += 1
            
            if invalid_count > 0:
                logger.warning(f"Skipped {invalid_count} invalid products")
            
            if not documents:
                logger.error("No valid documents to index")
                return 0
                
            # Create index per category
            collection_name = json_file.stem.replace("meta_", "").replace("_processed", "")
            
            vectordb = Chroma.from_documents(
                documents=documents,
                embedding=self.embedder,
                persist_directory=str(self.index_dir),
                collection_name=collection_name
            )
            
            logger.info(f"Created index for {collection_name} with {len(documents)} documents")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Indexing failed for {json_file.name}: {str(e)}")
            return 0

    def run_pipeline(self) -> int:
        """Run complete data processing pipeline"""
        total_docs = 0
        
        # Process all JSONL files in raw directory
        for file in self.raw_dir.glob("*.jsonl"):
            logger.info(f"Processing {file.name}...")
            
            # Step 1: Convert JSONL to cleaned JSON
            json_file = self._convert_jsonl_to_json(file)
            if not json_file:
                continue
                
            # Step 2: Generate vector index
            docs_indexed = self._generate_index(json_file)
            total_docs += docs_indexed
            
        return total_docs

def run_generator(data_dir: str = None) -> int:
    """
    Run the data generation and indexing pipeline
    
    Args:
        data_dir: Path to data directory (optional)
        
    Returns:
        Total number of documents indexed
    """
    try:
        generator = DataGenerator(data_dir)
        return generator.run_pipeline()
    except Exception as e:
        logger.critical(f"Data generation failed: {str(e)}", exc_info=True)
        return 0

if __name__ == "__main__":
    # For direct script execution
    total = run_generator()
    print(f"\nProcessing complete. Total documents indexed: {total}")